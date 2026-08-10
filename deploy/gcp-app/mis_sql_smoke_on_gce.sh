#!/usr/bin/env bash
# SQL smoke: Marina MariaDB from GCE using /opt/protocol/.env.mis (no password printed).
set -euo pipefail

PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
ENV_MIS_REMOTE="${ENV_MIS_REMOTE:-/opt/protocol/.env.mis}"

gcloud config set project "$PROJECT" --quiet >/dev/null

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
ENV_MIS='${ENV_MIS_REMOTE}'
test -f \"\$ENV_MIS\"
python3 -m venv /tmp/marina_smoke_venv >/tmp/marina_smoke_venv.log 2>&1
/tmp/marina_smoke_venv/bin/pip -q install pymysql >>/tmp/marina_smoke_venv.log 2>&1
set -a
# shellcheck disable=SC1090
source \"\$ENV_MIS\"
set +a
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
