#!/usr/bin/env bash
# Push Marina/MIS secrets to GCE (E2). Does not print secret values.
# Sources: local .env and/or ~/CURSOR/sql_epam/.env → remote /opt/protocol/.env.mis
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
ENV_MIS_REMOTE="${ENV_MIS_REMOTE:-/opt/protocol/.env.mis}"
RESTART_WEB="${RESTART_WEB:-0}"

usage() {
  cat <<'EOF'
Usage: deploy/gcp-app/push_mis_env.sh [--restart-web]

Writes /opt/protocol/.env.mis on protocol-app (chmod 600).
Does not rebuild images. Optional --restart-web only if INCLUDE_MIS_IN_WEB_ENV
was used during deploy (MIS keys normally stay out of protocol-web).
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
import os

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
if not vals.get("KRAVIRA_DB_PASSWORD"):
    raise SystemExit("ERROR: KRAVIRA_DB_PASSWORD missing in .env / sql_epam/.env")
out = Path("/tmp/protocol-gcp-mis.env")
out.write_text("".join(f"{k}={v}\n" for k, v in sorted(vals.items())), encoding="utf-8")
out.chmod(0o600)
print(f"local_mis_env keys={len(vals)} password=yes")
PY

gcloud compute scp /tmp/protocol-gcp-mis.env "${VM}:~/protocol-gcp-mis.env" --zone="$ZONE" --quiet
rm -f /tmp/protocol-gcp-mis.env

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo mkdir -p /opt/protocol
sudo mv ~/protocol-gcp-mis.env '${ENV_MIS_REMOTE}'
sudo chown \"\$(whoami):\$(whoami)\" '${ENV_MIS_REMOTE}'
sudo chmod 600 '${ENV_MIS_REMOTE}'
# key names only
awk -F= '{print \$1}' '${ENV_MIS_REMOTE}' | sed 's/^/MIS_KEY /'
test -n \"\$(grep -E '^KRAVIRA_DB_PASSWORD=.' '${ENV_MIS_REMOTE}' || true)\"
echo MIS_ENV_OK path=${ENV_MIS_REMOTE}
"

if [[ "$RESTART_WEB" == "1" ]]; then
  echo "RESTART_WEB=1 requested but MIS stays in .env.mis; skipping web recreate" >&2
fi

echo "Done. Smoke: bash deploy/gcp-app/mis_sql_smoke_on_gce.sh"
