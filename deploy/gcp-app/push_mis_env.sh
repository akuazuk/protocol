#!/usr/bin/env bash
# Push Marina/MIS config to GCE (E2). Does not print secret values.
#
# - Password → GCP Secret Manager (kravira-db-password)
# - Non-secret host/port/user/timeouts → /opt/protocol/.env.mis (owner=GCE_OPS_USER)
#
# Sources for password/local defaults: repo .env and/or ~/CURSOR/sql_epam/.env
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
ENV_MIS_REMOTE="${ENV_MIS_REMOTE:-/opt/protocol/.env.mis}"
MIS_SM_SECRET="${MIS_SM_SECRET:-kravira-db-password}"
# Cron owner on protocol-app (must match /var/spool/cron/crontabs/<user>).
# Deploy SSH may land as pavelkuzauka; night cron runs as pavel.
GCE_OPS_USER="${GCE_OPS_USER:-pavel}"
RESTART_WEB="${RESTART_WEB:-0}"

usage() {
  cat <<'EOF'
Usage: deploy/gcp-app/push_mis_env.sh [--restart-web]

Updates Secret Manager secret kravira-db-password and writes non-secret
/opt/protocol/.env.mis on protocol-app (chmod 600, owner=GCE_OPS_USER=pavel).
Does not rebuild images.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --restart-web) RESTART_WEB=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

gcloud config set project "$PROJECT" --quiet >/dev/null

python3 - <<'PY'
from pathlib import Path
import sys

mis_want = {
    "KRAVIRA_DB_PASSWORD",
    "KRAVIRA_DB_HOST",
    "KRAVIRA_DB_PORT",
    "KRAVIRA_DB_USER",
    "KRAVIRA_DB_NAME",
    "MIS_DB_CONNECT_TIMEOUT",
    "MIS_DB_READ_TIMEOUT",
}
vals: dict[str, str] = {}


def _ingest(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        if k in mis_want and v.strip() and k not in vals:
            vals[k] = v.strip().strip('"').strip("'")


_ingest(Path(".env"))
_ingest(Path.home() / "CURSOR" / "sql_epam" / ".env")
vals.setdefault("KRAVIRA_DB_HOST", "178.163.240.131")
vals.setdefault("KRAVIRA_DB_PORT", "6330")
vals.setdefault("KRAVIRA_DB_USER", "kravira_mc_user")
vals.setdefault("KRAVIRA_DB_NAME", "kravira_mc")
vals.setdefault("MIS_DB_CONNECT_TIMEOUT", "30")
vals.setdefault("MIS_DB_READ_TIMEOUT", "600")
vals.setdefault("RUN_HOST", "gcp")
pw = vals.get("KRAVIRA_DB_PASSWORD")
if not pw:
    raise SystemExit("ERROR: KRAVIRA_DB_PASSWORD missing in .env / sql_epam/.env")

pw_path = Path("/tmp/protocol-sm-mis-pw")
pw_path.write_text(pw, encoding="utf-8")
pw_path.chmod(0o600)

# Non-secret file only (password stays in Secret Manager).
public = {k: v for k, v in vals.items() if k != "KRAVIRA_DB_PASSWORD"}
out = Path("/tmp/protocol-gcp-mis.env")
out.write_text("".join(f"{k}={v}\n" for k, v in sorted(public.items())), encoding="utf-8")
out.chmod(0o600)
print(f"local_mis_env public_keys={len(public)} password_for_sm=yes chars={len(pw)}")
PY

# Secret Manager: create or add version
if gcloud secrets describe "$MIS_SM_SECRET" --project="$PROJECT" >/dev/null 2>&1; then
  gcloud secrets versions add "$MIS_SM_SECRET" --data-file=/tmp/protocol-sm-mis-pw --project="$PROJECT" >/dev/null
  echo "SM_VERSION_ADDED secret=$MIS_SM_SECRET"
else
  gcloud secrets create "$MIS_SM_SECRET" --data-file=/tmp/protocol-sm-mis-pw \
    --project="$PROJECT" --replication-policy=automatic >/dev/null
  SA="$(gcloud compute instances describe "$VM" --zone="$ZONE" --project="$PROJECT" \
    --format='get(serviceAccounts[0].email)')"
  gcloud secrets add-iam-policy-binding "$MIS_SM_SECRET" \
    --project="$PROJECT" \
    --member="serviceAccount:${SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet >/dev/null
  echo "SM_CREATED secret=$MIS_SM_SECRET accessor=$SA"
fi
rm -f /tmp/protocol-sm-mis-pw

gcloud compute scp /tmp/protocol-gcp-mis.env "${VM}:~/protocol-gcp-mis.env" --zone="$ZONE" --quiet
rm -f /tmp/protocol-gcp-mis.env

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo mkdir -p /opt/protocol
sudo mv ~/protocol-gcp-mis.env '${ENV_MIS_REMOTE}'
OPS_USER='${GCE_OPS_USER}'
if ! getent passwd \"\$OPS_USER\" >/dev/null 2>&1; then
  OPS_USER=\"\$(whoami)\"
fi
sudo chown \"\$OPS_USER:\$OPS_USER\" '${ENV_MIS_REMOTE}'
sudo chmod 600 '${ENV_MIS_REMOTE}'
sudo -u \"\$OPS_USER\" test -r '${ENV_MIS_REMOTE}'
# must not contain password
if grep -qE '^KRAVIRA_DB_PASSWORD=' '${ENV_MIS_REMOTE}'; then
  echo 'ERROR: password must not be stored in .env.mis (use Secret Manager)' >&2
  exit 2
fi
awk -F= '{print \$1}' '${ENV_MIS_REMOTE}' | sed 's/^/MIS_KEY /'
echo MIS_ENV_OK path=${ENV_MIS_REMOTE} owner=\$OPS_USER password=secretmanager:${MIS_SM_SECRET}
"

if [[ "$RESTART_WEB" == "1" ]]; then
  echo "RESTART_WEB=1 requested but MIS stays out of protocol-web; skipping" >&2
fi

echo "Done. Smoke: bash deploy/gcp-app/mis_sql_smoke_on_gce.sh"
