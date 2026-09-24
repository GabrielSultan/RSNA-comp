# -*- coding: utf-8 -*-
"""Push 3 kernels T4 v20h lat (ep 8/9/10), attend COMPLETE, soumet, telecharge."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
MAKE_NB = KERNEL_DIR / "_make_nb_v20h.py"
COMPETITION = "rsna-knee-abnormality-detection"
POLL_EVERY_S = 60
MAX_WAIT_S = 8 * 3600
TIMEOUT_S = "32400"
ACCEL = "NvidiaTeslaT4"
EPOCHS = (8, 9, 10)
GOLD = {7: 0.958, 8: 0.960, 9: 0.965, 10: 0.968, 11: 0.972, 12: 0.974}


def _kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _kernel_id(epoch: int) -> str:
    if epoch == 0:
        return "sultent/rsna-knee-v20h-lat-solo-swa"
    return f"sultent/rsna-knee-v20h-lat-solo-ep{epoch}"


def _push_dir(epoch: int) -> Path:
    if epoch == 0:
        return KERNEL_DIR / "_push_v20h_swa"
    return KERNEL_DIR / f"_push_v20h_ep{epoch}"


def _out_dir(epoch: int) -> Path:
    if epoch == 0:
        return KERNEL_DIR / "_out_v20h_swa"
    if epoch == 8:
        return KERNEL_DIR / "_out_v1"
    return KERNEL_DIR / f"_out_v20h_ep{epoch}"


def _parse_version(out: str) -> str:
    import re
    if "successfully pushed" not in out.lower():
        raise RuntimeError(f"push kernel echoue : {out.strip()!r}")
    m = re.search(r"Kernel version (\d+) successfully pushed", out, flags=re.I)
    if not m:
        raise RuntimeError(f"version kernel introuvable dans : {out!r}")
    return m.group(1)


def _status_upper(kernel: str) -> tuple[str, str]:
    raw = _kaggle("kernels", "status", kernel).strip()
    return raw, raw.upper()


def _is_failed(status_u: str) -> bool:
    return "FAILED" in status_u or "ERROR" in status_u


def _is_complete(status_u: str) -> bool:
    running = "RUNNING" in status_u
    done = ("COMPLETE" in status_u) or ("SUCCEEDED" in status_u)
    return done and not running


def try_push(epoch: int) -> str | None:
    push = _push_dir(epoch)
    meta = push / "kernel-metadata.json"
    nb = push / "submit.ipynb"
    if not meta.is_file() or not nb.is_file():
        raise FileNotFoundError(push)
    print("push", _kernel_id(epoch), "T4", TIMEOUT_S, "s", flush=True)
    try:
        out = _kaggle(
            "kernels", "push",
            "-p", str(push),
            "--accelerator", ACCEL,
            "-t", TIMEOUT_S,
        )
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        if "Maximum batch GPU" in text:
            print("quota GPU, retry plus tard ep", epoch, flush=True)
            return None
        raise
    print(out, flush=True)
    ver = _parse_version(out)
    print("kernel version", _kernel_id(epoch), ver, flush=True)
    return ver


def download_one(epoch: int) -> None:
    kernel = _kernel_id(epoch)
    dest = _out_dir(epoch)
    dest.mkdir(parents=True, exist_ok=True)
    out = _kaggle("kernels", "output", kernel, "-p", str(dest))
    print(out, flush=True)
    csv = dest / "submission.csv"
    print("downloaded", csv, "exists", csv.exists(), flush=True)
    if not csv.exists():
        raise SystemExit(f"submission.csv absent dans {dest}")


def submit_one(epoch: int, version: str, *, submit: bool = True) -> None:
    kernel = _kernel_id(epoch)
    if submit:
        if epoch == 0:
            msg = "v20h oofmask lat SWA ep7-10 (moyenne de poids, canon L, T4)"
        else:
            msg = (
                f"v20h oofmask v20b lat full solo epoch {epoch} "
                f"(gold in-sample {GOLD[epoch]:.3f}, canon L, T4)"
            )
        submit_out = _kaggle(
            "competitions", "submit", COMPETITION,
            "-k", kernel, "-v", version, "-f", "submission.csv", "-m", msg,
        )
        print(submit_out, flush=True)
        print("submitted", kernel, "version", version, flush=True)
    else:
        print("pas de soumission competition (quota) :", kernel, "v", version,
              flush=True)
    download_one(epoch)


def wait_all_and_submit(
    versions: dict[int, str],
    to_push: list[int] | None = None,
    *,
    submit: bool = True,
) -> dict[int, str]:
    pending = set(versions)
    leftover = list(to_push or [])
    seen_running = {ep: False for ep in versions}
    t0 = time.time()
    last_push_try = 0.0
    while pending or leftover:
        progressed = False
        if leftover and (time.time() - last_push_try) >= 60:
            last_push_try = time.time()
            still = []
            for ep in leftover:
                ver = try_push(ep)
                if ver is None:
                    still.append(ep)
                else:
                    versions[ep] = ver
                    pending.add(ep)
                    seen_running[ep] = False
                    progressed = True
            leftover = still
        for ep in sorted(pending):
            kernel = _kernel_id(ep)
            raw, status_u = _status_upper(kernel)
            print(f"[ep{ep}] {raw}", flush=True)
            if _is_failed(status_u):
                raise SystemExit(f"kernel ep{ep} failed")
            if "RUNNING" in status_u or "QUEUED" in status_u or "PENDING" in status_u:
                seen_running[ep] = True
            ready = _is_complete(status_u) and (
                seen_running[ep] or (time.time() - t0) > 120
            )
            if ready:
                submit_one(ep, versions[ep], submit=submit)
                pending.remove(ep)
                progressed = True
                last_push_try = 0.0
                break
        if not pending and not leftover:
            break
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(
                f"timeout waiting pending={sorted(pending)} leftover={leftover}"
            )
        if not progressed:
            time.sleep(POLL_EVERY_S)
    return versions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, nargs="*", default=list(EPOCHS))
    p.add_argument("--swa", action="store_true",
                   help="SWA ep7-10 (slug lat-solo-swa, epoch interne 0)")
    p.add_argument("--push-only", action="store_true")
    p.add_argument("--no-submit", action="store_true",
                   help="attend COMPLETE et telecharge, sans soumettre a la competition")
    p.add_argument("--submit-only", action="store_true",
                   help="ne pousse pas ; attend+soumet les versions deja lancees")
    p.add_argument("--version", action="append", default=[],
                   help="ep:version (ex. 8:1), pour --submit-only")
    args = p.parse_args()
    epochs = (0,) if args.swa else tuple(args.epoch)

    versions: dict[int, str] = {}
    for item in args.version:
        ep_s, _, ver = item.partition(":")
        versions[int(ep_s)] = ver

    leftover: list[int] = []
    if not args.submit_only:
        make_cmd = [sys.executable, "-X", "utf8", str(MAKE_NB)]
        if args.swa:
            make_cmd.append("--swa")
        else:
            make_cmd.extend(["--epoch", *[str(e) for e in epochs]])
        subprocess.check_call(make_cmd)
        for ep in epochs:
            if ep in versions:
                continue
            ver = try_push(ep)
            if ver is None:
                leftover.append(ep)
            else:
                versions[ep] = ver

    if args.push_only:
        print("versions", versions, "leftover", leftover, flush=True)
        return

    running = {ep: versions[ep] for ep in epochs if ep in versions}
    leftover.extend(
        ep for ep in epochs if ep not in versions and ep not in leftover
    )
    if not running and not leftover:
        raise SystemExit("rien a attendre ni a pousser")
    wait_all_and_submit(running, leftover, submit=not args.no_submit)
    print("DONE", versions, flush=True)


if __name__ == "__main__":
    main()
