#!/usr/bin/env bash
# A lancer PAR L'UTILISATEUR dans un terminal interactif du pod :
#   ssh -t -p PORT -i ~/.ssh/id_ed25519_runpod root@HOTE "bash /workspace/runpod/oai/nda_login.sh"
# downloadcmd demande l'identifiant puis le mot de passe NDA, les valide,
# les garde (settings.cfg + keyring fichier sous /root, disque conteneur),
# et telecharge 1 archive test. Un 2e appel sans terminal verifie que le
# pipeline n'aura plus rien a demander.
set -euo pipefail
cd /workspace
T=/root/oai_login_test
mkdir -p "$T"
python - <<'PY'
import pandas as pd
s = pd.read_csv("data/oai/derived/oai_series.csv").sort_values("file_size")
s = s[s.file_size > 1000]
open("/root/oai_login_test/links_login_1.txt", "w").write(s.s3_url.iloc[0] + "\n")
open("/root/oai_login_test/links_login_2.txt", "w").write(s.s3_url.iloc[1] + "\n")
PY
python -m NDATools.clientscripts.downloadcmd -dp 1249964 -t "$T/links_login_1.txt" -d "$T" -wt 1
echo "--- 2e appel sans terminal (doit passer sans rien demander)"
setsid -w python -m NDATools.clientscripts.downloadcmd -dp 1249964 -t "$T/links_login_2.txt" -d "$T" -wt 1 < /dev/null
n=$(find "$T" -name '*.tar.gz' | wc -l)
rm -rf "$T"
if [ "$n" -ge 2 ]; then echo "NDA OK ($n archives test)"; else echo "NDA ECHEC ($n archive)"; exit 1; fi
