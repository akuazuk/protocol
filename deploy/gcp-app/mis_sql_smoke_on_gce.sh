#!/usr/bin/env bash
# SQL smoke: Marina MariaDB from GCE via Secret Manager password (no password printed).
set -euo pipefail

PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
ENV_MIS_REMOTE="${ENV_MIS_REMOTE:-/opt/protocol/.env.mis}"
MIS_SM_SECRET="${MIS_SM_SECRET:-kravira-db-password}"

gcloud config set project "$PROJECT" --quiet >/dev/null

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
export ENV_MIS='${ENV_MIS_REMOTE}'
export GCP_PROJECT='${PROJECT}'
export MIS_SM_SECRET='${MIS_SM_SECRET}'
export MIS_PASSWORD_SOURCE=secretmanager
# Prefer installed night helper; fallback inline if not yet synced
if [[ -f /opt/protocol/deploy/gcp-app/load_mis_env.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/protocol/deploy/gcp-app/load_mis_env.sh
else
  export KRAVIRA_DB_HOST=178.163.240.131 KRAVIRA_DB_PORT=6330
  export KRAVIRA_DB_USER=kravira_mc_user KRAVIRA_DB_NAME=kravira_mc
  export KRAVIRA_DB_PASSWORD=\"\$(gcloud secrets versions access latest --secret=\$MIS_SM_SECRET --project=\$GCP_PROJECT)\"
  echo MIS_DSN_OK source=secretmanager-inline
fi
python3 -m venv /tmp/marina_smoke_venv >/tmp/marina_smoke_venv.log 2>&1
/tmp/marina_smoke_venv/bin/pip -q install pymysql >>/tmp/marina_smoke_venv.log 2>&1
/tmp/marina_smoke_venv/bin/python - <<'PY'
import os, pymysql
host=os.environ.get('KRAVIRA_DB_HOST','178.163.240.131')
port=int(os.environ.get('KRAVIRA_DB_PORT','6330'))
user=os.environ.get('KRAVIRA_DB_USER','kravira_mc_user')
db=os.environ.get('KRAVIRA_DB_NAME','kravira_mc')
pw=os.environ['KRAVIRA_DB_PASSWORD']
conn=pymysql.connect(host=host, port=port, user=user, password=pw, database=db,
                     connect_timeout=int(os.environ.get('MIS_DB_CONNECT_TIMEOUT','30')),
                     read_timeout=40, charset='utf8mb4')
with conn.cursor() as cur:
    cur.execute('SELECT DATABASE(), USER(), @@version, MAX(date) FROM mis_protocol')
    database, user, ver, mx = cur.fetchone()
    print('SQL_OK')
    print('host=', host)
    print('port=', port)
    print('database=', database)
    print('user=', user.split('@')[0] + '@***')
    print('server_version=', ver)
    print('mis_protocol_max_date=', mx)
conn.close()
PY
rm -rf /tmp/marina_smoke_venv
"
