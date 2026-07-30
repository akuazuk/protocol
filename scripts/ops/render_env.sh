#!/usr/bin/env bash
# Inspect and change environment variables of the live Render service.
#
# render.yaml does NOT configure prod (it belongs to the suspended protocol-rag service),
# so a variable added there never reaches the running app. This is the tool that does.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVICE_ID="${RENDER_SERVICE_ID:-srv-d78he6h5pdvs73b1kufg}"
API_BASE="${RENDER_API_BASE:-https://api.render.com/v1}"
SHOW_VALUES=0

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/render_env.sh list [--show-values]
  scripts/ops/render_env.sh diff
  scripts/ops/render_env.sh get KEY
  scripts/ops/render_env.sh set KEY=VALUE
  scripts/ops/render_env.sh unset KEY

Commands:
  list    variables of the live service (values masked unless --show-values)
  diff    what render.yaml declares vs what the service actually has
  get     one variable
  set     add or update one variable (Render redeploys the service afterwards)
  unset   delete one variable

Options:
  --service-id=ID   override RENDER_SERVICE_ID (default: prod)
  --show-values     print values instead of masking them

Auth:
  RENDER_API_KEY from the environment or from .env.
EOF
}

CMD="${1:-}"
if [[ -z "$CMD" || "$CMD" == "-h" || "$CMD" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

ARG=""
for arg in "$@"; do
  case "$arg" in
    --service-id=*) SERVICE_ID="${arg#*=}" ;;
    --show-values) SHOW_VALUES=1 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    *) ARG="$arg" ;;
  esac
done

if [[ -z "${RENDER_API_KEY:-}" && -f .env ]]; then
  RENDER_API_KEY="$(sed -n 's/^[[:space:]]*RENDER_API_KEY[[:space:]]*=[[:space:]]*//p' .env | head -n1 | tr -d '"'"'"'\r')"
fi
if [[ -z "${RENDER_API_KEY:-}" ]]; then
  echo "ERROR: RENDER_API_KEY is not set (see scripts/ops/render_deploy.sh --help)." >&2
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

fetch_env_vars() {
  local resp
  # limit is capped at 100 by the API; the service has far fewer variables than that.
  if ! resp="$(api GET "/services/${SERVICE_ID}/env-vars?limit=100")" || [[ -z "$resp" ]]; then
    echo "ERROR: cannot read env vars of ${SERVICE_ID} (check RENDER_API_KEY and the id)." >&2
    return 1
  fi
  printf '%s' "$resp"
}

PY_LIST='
import json, sys
show = sys.argv[1] == "1"
rows = [e.get("envVar", e) for e in json.load(sys.stdin)]
for r in sorted(rows, key=lambda x: x.get("key", "")):
    name = r.get("key", "")
    value = r.get("value", "")
    if not show:
        value = f"<{len(value)} chars>" if len(value) > 12 else value
    print(f"  {name:<40} {value}")
print(f"total: {len(rows)}")
'

PY_DIFF='
import json, re, sys
from pathlib import Path

live = {e.get("envVar", e)["key"]: e.get("envVar", e).get("value", "")
        for e in json.load(sys.stdin)}
text = Path("render.yaml").read_text(encoding="utf-8")
declared = {}
key = None
for line in text.splitlines():
    m = re.match(r"\s*-\s*key:\s*(\S+)", line)
    if m:
        key = m.group(1)
        continue
    m = re.match(r"\s*value:\s*(.*)", line)
    if m and key:
        declared[key] = m.group(1).strip().strip(chr(34)).strip(chr(39))
        key = None

missing = [k for k in declared if k not in live]
different = [k for k in declared if k in live and declared[k] != live[k]]
extra = [k for k in live if k not in declared]

print(f"render.yaml declares: {len(declared)}   service has: {len(live)}")
print()
print(f"missing on the service ({len(missing)}):")
for k in sorted(missing):
    print(f"  {k} = {declared[k]}")
print()
print(f"different value ({len(different)}):")
for k in sorted(different):
    print(f"  {k}: render.yaml={declared[k]!r} service=<{len(live[k])} chars>")
print()
print(f"only on the service ({len(extra)}):")
for k in sorted(extra):
    print(f"  {k}")
'

case "$CMD" in
  list)
    fetch_env_vars | python3 -c "$PY_LIST" "$SHOW_VALUES"
    ;;

  diff)
    fetch_env_vars | python3 -c "$PY_DIFF"
    ;;

  get)
    if [[ -z "$ARG" ]]; then
      echo "ERROR: KEY is required." >&2
      exit 2
    fi
    api GET "/services/${SERVICE_ID}/env-vars/${ARG}" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("key"), "=", d.get("value"))'
    ;;

  set)
    if [[ "$ARG" != *=* ]]; then
      echo "ERROR: expected KEY=VALUE." >&2
      exit 2
    fi
    env_key="${ARG%%=*}"
    env_value="${ARG#*=}"
    body="$(python3 -c 'import json,sys; print(json.dumps({"value": sys.argv[1]}))' "$env_value")"
    api PUT "/services/${SERVICE_ID}/env-vars/${env_key}" "$body" >/dev/null
    echo "Set ${env_key} on ${SERVICE_ID}"
    echo "Render redeploys the service to apply it: scripts/ops/render_deploy.sh deploys"
    ;;

  unset)
    if [[ -z "$ARG" ]]; then
      echo "ERROR: KEY is required." >&2
      exit 2
    fi
    api DELETE "/services/${SERVICE_ID}/env-vars/${ARG}" >/dev/null
    echo "Removed ${ARG} from ${SERVICE_ID}"
    ;;

  *)
    echo "Unknown command: $CMD" >&2
    usage >&2
    exit 2
    ;;
esac
