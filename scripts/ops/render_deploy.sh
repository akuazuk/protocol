#!/usr/bin/env bash
# Manage the Render service via the public API (deploy, restart, status, logs).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVICE_ID="${RENDER_SERVICE_ID:-srv-d78he6h5pdvs73b1kufg}"
PROD_URL="${PROTOCOL_PROD_URL:-https://protocol-bimy.onrender.com}"
API_BASE="${RENDER_API_BASE:-https://api.render.com/v1}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-15}"

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/render_deploy.sh <command> [options]

Commands:
  status              service settings (autoDeploy/branch/suspended) + last deploys + prod version
  deploys             recent deploys with status and commit
  deploy              trigger a deploy of the latest commit on the service branch
  restart             restart the service without rebuilding (use after uploading data to /var/data)
  logs                recent service logs
  services            list services visible to the API key (to find RENDER_SERVICE_ID)
  suspend             stop the service (billing and auto-deploy stop, the disk is kept)
  resume              start a suspended service again
  has-key             exit 0 if an API key is configured, 1 otherwise (for scripting)

Options:
  --wait              wait until the triggered deploy reaches "live"
  --clear-cache       deploy with a cleared build cache
  --commit=SHA        deploy a specific commit instead of the branch tip
  --limit=N           number of deploys / log lines (default 5 / 100)
  --service-id=ID     override RENDER_SERVICE_ID
  --prod-url=URL      override PROTOCOL_PROD_URL
  --timeout-sec=N     wait timeout, default 900
  --interval-sec=N    poll interval, default 15

Auth:
  RENDER_API_KEY from the environment or from .env (never commit it).
  Create a key at https://dashboard.render.com/u/settings?add-api-key
EOF
}

CMD="${1:-}"
if [[ -z "$CMD" || "$CMD" == "-h" || "$CMD" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

WAIT=0
CLEAR_CACHE=0
COMMIT_SHA=""
LIMIT=""

for arg in "$@"; do
  case "$arg" in
    --wait) WAIT=1 ;;
    --clear-cache) CLEAR_CACHE=1 ;;
    --commit=*) COMMIT_SHA="${arg#*=}" ;;
    --limit=*) LIMIT="${arg#*=}" ;;
    --service-id=*) SERVICE_ID="${arg#*=}" ;;
    --prod-url=*) PROD_URL="${arg#*=}" ;;
    --timeout-sec=*) WAIT_TIMEOUT_SEC="${arg#*=}" ;;
    --interval-sec=*) WAIT_INTERVAL_SEC="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${RENDER_API_KEY:-}" && -f .env ]]; then
  RENDER_API_KEY="$(sed -n 's/^[[:space:]]*RENDER_API_KEY[[:space:]]*=[[:space:]]*//p' .env | head -n1 | tr -d '"'"'"'\r')"
fi

if [[ "$CMD" == "has-key" ]]; then
  if [[ -n "${RENDER_API_KEY:-}" ]]; then
    echo "RENDER_API_KEY is configured"
    exit 0
  fi
  exit 1
fi

if [[ -z "${RENDER_API_KEY:-}" ]]; then
  cat >&2 <<EOF
ERROR: RENDER_API_KEY is not set.

Create a key at https://dashboard.render.com/u/settings?add-api-key
and add it to .env (the file is gitignored):

  RENDER_API_KEY=rnd_xxxxxxxxxxxxxxxxxxxx
EOF
  exit 1
fi

api() {
  local method="$1" path="$2" body="${3:-}"
  local args=(-fsS --request "$method" --url "${API_BASE}${path}"
    --header "Authorization: Bearer ${RENDER_API_KEY}"
    --header 'Accept: application/json')
  if [[ -n "$body" ]]; then
    args+=(--header 'Content-Type: application/json' --data "$body")
  fi
  curl "${args[@]}"
}

# Render wraps list responses as [{"<key>": {...}, "cursor": "..."}]; unwrap to a plain list.
PY_UNWRAP='
import json, sys
key = sys.argv[1]
data = json.load(sys.stdin)
if isinstance(data, list):
    data = [item.get(key, item) if isinstance(item, dict) else item for item in data]
json.dump(data, sys.stdout)
'

PY_SERVICES='
import json, sys
for s in json.load(sys.stdin):
    sid = s.get("id", "")
    name = s.get("name", "")
    stype = s.get("type", "")
    branch = s.get("branch", "-")
    auto = s.get("autoDeploy", "-")
    print(f"{sid}  {name:<24} {stype}  branch={branch}  autoDeploy={auto}")
'

PY_SERVICE='
import json, sys
s = json.load(sys.stdin)
for label, key in (("id", "id"), ("name", "name"), ("repo", "repo"), ("branch", "branch"),
                   ("autoDeploy", "autoDeploy"), ("suspended", "suspended"),
                   ("dashboard", "dashboardUrl")):
    print("  {:<11} {}".format(label + ":", s.get(key)))
if s.get("autoDeploy") == "no":
    print("  WARNING: autoDeploy is off - a push to the branch will NOT deploy.")
'

PY_DEPLOYS='
import json, sys
limit = int(sys.argv[1])
for d in json.load(sys.stdin)[:limit]:
    commit = d.get("commit") or {}
    sha = (commit.get("id") or "")[:7] or "-"
    msg = (commit.get("message") or "").splitlines()
    status = d.get("status", "?")
    created = (d.get("createdAt") or "")[:19]
    trigger = d.get("trigger", "")
    print(f"  {status:<22} {sha:<8} {created}  {trigger}")
    if msg:
        print("      " + msg[0][:60])
'

PY_LOGS='
import json, sys
data = json.load(sys.stdin)
entries = data.get("logs", []) if isinstance(data, dict) else (data or [])
for e in reversed(entries):
    ts = (e.get("timestamp") or "")[:19]
    print(f"{ts}  " + (e.get("message") or "").rstrip())
'

prod_version() {
  curl -fsS --max-time 20 "${PROD_URL%/}/api/version" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null \
    || echo "unreachable"
}

local_build_version() {
  python3 - <<'PY'
import re
from pathlib import Path
t = Path("rag_server.py").read_text(encoding="utf-8")
m = re.search(r'^BUILD_VERSION\s*=\s*"([^"]+)"', t, re.M)
print(m.group(1) if m else "")
PY
}

wait_for_deploy() {
  local deploy_id="$1"
  local start_ts elapsed status
  start_ts="$(date +%s)"
  echo "Waiting for deploy $deploy_id (timeout ${WAIT_TIMEOUT_SEC}s)..."
  while true; do
    elapsed=$(( $(date +%s) - start_ts ))
    status="$(api GET "/services/${SERVICE_ID}/deploys/${deploy_id}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))')"
    echo "  elapsed=${elapsed}s status=${status}"
    case "$status" in
      live)
        echo "DEPLOY_OK status=live prod_version=$(prod_version)"
        return 0
        ;;
      build_failed|update_failed|canceled|pre_deploy_failed|deactivated)
        echo "DEPLOY_FAILED status=$status" >&2
        echo "Logs: scripts/ops/render_deploy.sh logs --limit=100" >&2
        return 1
        ;;
    esac
    if (( elapsed >= WAIT_TIMEOUT_SEC )); then
      echo "DEPLOY_TIMEOUT status=$status" >&2
      return 1
    fi
    sleep "$WAIT_INTERVAL_SEC"
  done
}

case "$CMD" in
  services)
    api GET "/services?limit=50" \
      | python3 -c "$PY_UNWRAP" service \
      | python3 -c "$PY_SERVICES"
    ;;

  status)
    echo "Service"
    api GET "/services/${SERVICE_ID}" | python3 -c "$PY_SERVICE"
    echo
    echo "Recent deploys:"
    api GET "/services/${SERVICE_ID}/deploys?limit=${LIMIT:-5}" \
      | python3 -c "$PY_UNWRAP" deploy \
      | python3 -c "$PY_DEPLOYS" "${LIMIT:-5}"
    echo
    build_ver="$(local_build_version)"
    prod_ver="$(prod_version)"
    echo "Version"
    echo "  local BUILD_VERSION: $build_ver"
    echo "  prod /api/version:   $prod_ver"
    if [[ "$build_ver" == "$prod_ver" ]]; then
      echo "  IN_SYNC"
    else
      echo "  OUT_OF_SYNC - prod does not run the local commit yet"
    fi
    ;;

  deploys)
    api GET "/services/${SERVICE_ID}/deploys?limit=${LIMIT:-5}" \
      | python3 -c "$PY_UNWRAP" deploy \
      | python3 -c "$PY_DEPLOYS" "${LIMIT:-5}"
    ;;

  deploy)
    cache_mode="do_not_clear"
    if [[ "$CLEAR_CACHE" == "1" ]]; then
      cache_mode="clear"
    fi
    body="$(python3 -c '
import json, sys
payload = {"clearCache": sys.argv[1]}
if sys.argv[2]:
    payload["commitId"] = sys.argv[2]
print(json.dumps(payload))
' "$cache_mode" "$COMMIT_SHA")"
    deploy_id="$(api POST "/services/${SERVICE_ID}/deploys" "$body" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')"
    echo "Triggered deploy: $deploy_id"
    if [[ "$WAIT" == "1" ]]; then
      wait_for_deploy "$deploy_id"
    else
      echo "Follow with: scripts/ops/render_deploy.sh deploys"
    fi
    ;;

  suspend|resume)
    api POST "/services/${SERVICE_ID}/${CMD}" '{}' >/dev/null
    echo "Requested ${CMD} for ${SERVICE_ID}"
    api GET "/services/${SERVICE_ID}" | python3 -c "$PY_SERVICE"
    ;;

  restart)
    api POST "/services/${SERVICE_ID}/restart" '{}' >/dev/null
    echo "Restart requested for ${SERVICE_ID}"
    if [[ "$WAIT" == "1" ]]; then
      echo "Waiting for /health/live..."
      for _ in $(seq 1 40); do
        if curl -fsS --max-time 10 "${PROD_URL%/}/health/live" >/dev/null 2>&1; then
          echo "RESTART_OK version=$(prod_version)"
          exit 0
        fi
        sleep "$WAIT_INTERVAL_SEC"
      done
      echo "RESTART_TIMEOUT: /health/live still failing" >&2
      exit 1
    fi
    ;;

  logs)
    owner_id="$(api GET "/services/${SERVICE_ID}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin).get("ownerId",""))')"
    if [[ -z "$owner_id" ]]; then
      echo "ERROR: cannot resolve ownerId for ${SERVICE_ID}" >&2
      exit 1
    fi
    api GET "/logs?ownerId=${owner_id}&resource=${SERVICE_ID}&limit=${LIMIT:-100}&direction=backward" \
      | python3 -c "$PY_LOGS"
    ;;

  *)
    echo "Unknown command: $CMD" >&2
    usage >&2
    exit 2
    ;;
esac
