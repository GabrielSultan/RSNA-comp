# -*- coding: utf-8 -*-
"""Pod : telechargement NDA par lots + conversion OAI -> shards crop130 / 320.

Un thread telecharge le lot k+1 (downloadcmd -t) pendant que le pool de
processus convertit le lot k. Au plus 2 lots bruts sur le disque conteneur.
Un seul processus ecrit les zips (ZIP_STORED), d'abord en .partial puis
renommes : un shard present est complet. Noms sans "part"
(oai_crop130_bNN.zip) pour ne pas tromper _index_from_manifests.
Reprise : un lot dont les deux shards existent est saute.

    nohup python runpod/oai/pipeline.py > /workspace/logs/oai_pipeline.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.oai_convert import convert_series, npz_bytes  # noqa: E402

GB = 1e9
RUN_ID = time.strftime("%m%d%H%M%S")
DL_PROCS: set[subprocess.Popen] = set()
MAX_MISSING = 0.02
MIN_OK = 0.95


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / GB


def shard_paths(out224: Path, out320: Path, b: int) -> tuple[Path, Path]:
    return out224 / f"oai_crop130_b{b:02d}.zip", out320 / f"oai_crop130_320_b{b:02d}.zip"


def _present(path: Path, expected: int) -> bool:
    """FILE_SIZE vaut 1 pour ~2,5 % des archives NDA (taille inconnue)."""
    if not path.is_file():
        return False
    size = path.stat().st_size
    return size == expected or (expected <= 1 and size > 0)


def missing_archives(rows: pd.DataFrame, raw: Path) -> pd.DataFrame:
    ok = [_present(raw / f, int(s)) for f, s in zip(rows.image_file, rows.file_size)]
    return rows[[not x for x in ok]]


def est_bytes(rows: pd.DataFrame) -> float:
    return float(rows.file_size.where(rows.file_size > 1, 8.2e6).sum())


def download_batch(b: int, rows: pd.DataFrame, raw: Path, args) -> int:
    """-> nombre d'archives toujours manquantes apres les essais."""
    raw.mkdir(parents=True, exist_ok=True)
    need_gb = est_bytes(rows) / GB * 1.1 + 2
    while free_gb(raw) < need_gb:
        log(f"lot {b:02d} : {free_gb(raw):.1f} Go libres sur {raw}, "
            f"{need_gb:.1f} requis, attente")
        time.sleep(60)
    todo = missing_archives(rows, raw)
    for attempt in range(1, args.dl_attempts + 1):
        if todo.empty:
            break
        # downloadcmd saute ce qu'un job (dossier + nom du fichier de liens)
        # a deja enregistre, meme efface du disque : un nom neuf par essai.
        links = raw / f"links_b{b:02d}_{RUN_ID}_a{attempt}.txt"
        links.write_text("\n".join(todo.s3_url) + "\n", encoding="utf-8")
        t0, before = time.time(), est_bytes(todo)
        with open(args.log_dir / f"downloadcmd_b{b:02d}.log", "a", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                [sys.executable, "-m", "NDATools.clientscripts.downloadcmd",
                 "-dp", str(args.package), "-t", str(links),
                 "-d", str(raw), "-wt", str(args.dl_threads)],
                stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
            DL_PROCS.add(proc)
            rc = proc.wait()
            DL_PROCS.discard(proc)
        todo = missing_archives(rows, raw)
        dt = time.time() - t0
        log(f"lot {b:02d} : telechargement essai {attempt} rc={rc} en {dt:.0f} s, "
            f"{len(rows) - len(todo)}/{len(rows)} archives, "
            f"{(before - est_bytes(todo)) / 1e6 / max(dt, 1):.0f} Mo/s")
    return len(todo)


def _convert_one(job: dict) -> tuple[dict, bytes | None, bytes | None]:
    t0 = time.time()
    rec = {k: job[k] for k in ("StudyInstanceUID", "SeriesInstanceUID", "plane", "fluid", "batch")}
    try:
        outs, meta = convert_series(job["path"])
        b224, b320 = npz_bytes(outs[224]), npz_bytes(outs[320])
        rec.update(ok=1, error="", nbytes224=len(b224), nbytes320=len(b320), **meta)
    except Exception as e:  # noqa: BLE001
        b224 = b320 = None
        rec.update(ok=0, error=f"{type(e).__name__}: {e}"[:300], nbytes224=0, nbytes320=0)
    rec["dt"] = round(time.time() - t0, 3)
    return rec, b224, b320


def _worker_init() -> None:
    import cv2
    cv2.setNumThreads(1)


def convert_batch(b: int, rows: pd.DataFrame, raw: Path, pool, args) -> dict:
    z224, z320 = shard_paths(args.out224, args.out320, b)
    while min(free_gb(args.out224), free_gb(args.out320)) < args.min_free_volume_gb:
        log(f"lot {b:02d} : moins de {args.min_free_volume_gb} Go libres sur le volume, attente")
        time.sleep(120)
    p224, p320 = z224.with_suffix(".zip.partial"), z320.with_suffix(".zip.partial")
    jobs = [{"StudyInstanceUID": r.StudyInstanceUID, "SeriesInstanceUID": r.SeriesInstanceUID,
             "plane": r.Anatomical_Plane, "fluid": int(r.Fluid_Sensitive), "batch": b,
             "path": str(raw / r.image_file)} for r in rows.itertuples()]
    recs, t0 = [], time.time()
    with zipfile.ZipFile(p224, "w", zipfile.ZIP_STORED, allowZip64=True) as f224, \
         zipfile.ZipFile(p320, "w", zipfile.ZIP_STORED, allowZip64=True) as f320:
        futs = [pool.submit(_convert_one, j) for j in jobs if Path(j["path"]).is_file()]
        for j in jobs:
            if not Path(j["path"]).is_file():
                recs.append({**{k: v for k, v in j.items() if k != "path"},
                             "ok": 0, "error": "archive absente", "nbytes224": 0, "nbytes320": 0})
        for i, fut in enumerate(as_completed(futs), 1):
            rec, b224, b320 = fut.result()
            recs.append(rec)
            if rec["ok"]:
                f224.writestr(f"{rec['SeriesInstanceUID']}.npz", b224)
                f320.writestr(f"{rec['SeriesInstanceUID']}.npz", b320)
            if i % 200 == 0:
                log(f"lot {b:02d} : {i}/{len(futs)} series converties")
    man = pd.DataFrame(recs)
    for d in (args.out224, args.out320):
        man.to_csv(d / f"oai_manifest_b{b:02d}.csv", index=False)
    n_ok = int(man.ok.sum())
    if n_ok < MIN_OK * len(man):
        for err in man[man.ok == 0].error.head(5):
            log(f"   echec : {err}")
        raise SystemExit(f"lot {b:02d} : {n_ok}/{len(man)} ok, shards laisses en .partial")
    os.replace(p224, z224)
    os.replace(p320, z320)
    log(f"lot {b:02d} : {n_ok}/{len(man)} ok en {time.time() - t0:.0f} s | "
        f"{z224.stat().st_size / GB:.2f} + {z320.stat().st_size / GB:.2f} Go")
    for err in man[man.ok == 0].error.head(5):
        log(f"   echec : {err}")
    return {"batch": int(b), "n": len(man), "ok": n_ok,
            "gb224": z224.stat().st_size / GB, "gb320": z320.stat().st_size / GB}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=str(ROOT / "data/oai/derived/oai_series.csv"))
    ap.add_argument("--package", type=int, default=1249964)
    ap.add_argument("--raw", default="/root/oai_raw", help="disque conteneur, pas le volume")
    ap.add_argument("--out224", default=str(ROOT / "data/crop130"))
    ap.add_argument("--out320", default=str(ROOT / "data/crop130_320"))
    ap.add_argument("--log-dir", default="/workspace/logs")
    ap.add_argument("--workers", type=int, default=max(1, min(24, (os.cpu_count() or 8) - 4)))
    ap.add_argument("--dl-threads", type=int, default=16)
    ap.add_argument("--dl-attempts", type=int, default=3)
    ap.add_argument("--min-free-volume-gb", type=float, default=6.0)
    ap.add_argument("--batches", default="", help="ex. 0,1 (defaut : tous)")
    args = ap.parse_args()
    for k in ("out224", "out320", "log_dir"):
        setattr(args, k, Path(getattr(args, k)))
        getattr(args, k).mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw)
    raw_root.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(args.series)
    batches = sorted(int(b) for b in series.batch.unique())
    if args.batches:
        batches = [b for b in batches if b in {int(x) for x in args.batches.split(",")}]
    todo = [b for b in batches if not all(p.is_file() for p in shard_paths(args.out224, args.out320, b))]
    log(f"{len(series)} series, {len(batches)} lots, {len(todo)} a faire, "
        f"{args.workers} workers, raw={raw_root} ({free_gb(raw_root):.0f} Go libres)")

    ready: Queue = Queue()
    slots = threading.Semaphore(2)
    failed = threading.Event()

    def downloader() -> None:
        try:
            for b in todo:
                slots.acquire()
                rows = series[series.batch == b]
                n_miss = download_batch(b, rows, raw_root / f"b{b:02d}", args)
                ready.put((b, n_miss))
        except Exception:  # noqa: BLE001
            log("ERREUR telechargement :\n" + traceback.format_exc())
            failed.set()
        finally:
            ready.put(None)

    th = threading.Thread(target=downloader, daemon=True)
    th.start()
    summary, t_all = [], time.time()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            while (item := ready.get()) is not None:
                b, n_miss = item
                n_rows = int((series.batch == b).sum())
                if n_miss > MAX_MISSING * n_rows:
                    raise SystemExit(f"lot {b:02d} : {n_miss}/{n_rows} archives manquantes, arret")
                if n_miss:
                    log(f"lot {b:02d} : {n_miss} archives manquantes, converti sans elles")
                raw_b = raw_root / f"b{b:02d}"
                summary.append(convert_batch(b, series[series.batch == b], raw_b, pool, args))
                shutil.rmtree(raw_b, ignore_errors=True)
                slots.release()
                done = sum(s["n"] for s in summary)
                log(f"cumul : {len(summary)}/{len(todo)} lots, {done} series, "
                    f"{(time.time() - t_all) / 60:.0f} min")
    finally:
        for proc in list(DL_PROCS):
            proc.terminate()
    th.join()

    out = {"batches": summary, "n_ok": sum(s["ok"] for s in summary),
           "n": sum(s["n"] for s in summary), "minutes": round((time.time() - t_all) / 60, 1)}
    if failed.is_set():
        raise SystemExit("arret : echec du telechargement (voir plus haut)")
    (args.log_dir / "oai_pipeline_done.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"termine : {out['n_ok']}/{out['n']} series ok en {out['minutes']} min")


if __name__ == "__main__":
    main()
