#!/usr/bin/env bash
# File de trains d'une voie GPU : bash runpod/oai/queue.sh LANE cfg1.yaml [cfg2.yaml ...]
# Pour chaque config : attente des donnees (tailles exactes + shards OAI),
# controle a blanc, puis train. Run deja fini (epoch3.pt) : saute. last.pt
# present : --resume. Echec : la voie s'arrete (diagnostic avant la suite).
# Pas d'OMP_NUM_THREADS ni de CUBLAS_WORKSPACE_CONFIG exportes (cf. docs).
# Lancer detache :  setsid nohup bash runpod/oai/queue.sh 224 ... > /dev/null 2>&1 &
set -uo pipefail
cd /workspace
export PYTHONUNBUFFERED=1
LANE=$1; shift
LOGS=/workspace/logs
QLOG=$LOGS/queue_$LANE.log
mkdir -p "$LOGS"
echo $$ > "$LOGS/queue_$LANE.pid"
say() { echo "$(date -Is) $*" >> "$QLOG"; }

for CFG in "$@"; do
  RUN=$(sed -n 's/^run_name:[[:space:]]*//p' "$CFG" | tr -d '"'"'"' \r')
  ZIP=$(sed -n 's/^zip_path:[[:space:]]*//p' "$CFG" | tr -d '"'"'"' \r')
  LAST=$(sed -n 's/^epochs:[[:space:]]*//p' "$CFG" | tr -d ' \r')
  DIR=checkpoints/$RUN
  if [ -f "$DIR/epoch$LAST.pt" ]; then say "$RUN deja fini (epoch$LAST.pt), saute"; continue; fi
  say "$RUN : attente des donnees $ZIP"
  if ! python -X utf8 runpod/oai/check_data.py "$ZIP" --wait >> "$QLOG" 2>&1; then
    say "$RUN : check_data ECHEC, voie arretee"; exit 1
  fi
  EXTRA=""
  if [ -f "$DIR/last.pt" ]; then
    EXTRA="--resume"
  else
    if ! python -X utf8 runpod/oai/dryrun.py "$CFG" >> "$LOGS/dryrun_$RUN.log" 2>&1; then
      say "$RUN : controle a blanc ECHEC (voir dryrun_$RUN.log), voie arretee"; exit 1
    fi
  fi
  say "$RUN : lancement $EXTRA"
  python -X utf8 -m src.train.train --config "$CFG" $EXTRA >> "$LOGS/train_$RUN.log" 2>&1 &
  PID=$!
  echo "$PID" > "$LOGS/train_$LANE.pid"
  wait "$PID"; RC=$?
  say "train exit $RC $RUN"
  rm -f "$LOGS/train_$LANE.pid"
  if [ "$RC" -ne 0 ]; then say "voie $LANE arretee sur $RUN"; exit "$RC"; fi
done
say "voie $LANE terminee"
rm -f "$LOGS/queue_$LANE.pid"
