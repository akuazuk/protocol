#!/usr/bin/env bash
# Wait until Render /api/version matches expected BUILD_VERSION.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROD_URL="${PROTOCOL_PROD_URL:-https://protocol-bimy.onrender.com}"
EXPECTED_VERSION="${EXPECTED_VERSION:-}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
INTERVAL_SEC="${INTERVAL_SEC:-20}"

usage() {
  cat <<'EOF'
Usage:
  scripts/render_wait_version.sh [--expected VERSION] [--prod-url URL] [--timeout-sec N] [--interval-sec N]

Defaults:
  --expected      BUILD_VERSION from local rag_server.py
  --prod-url      https://protocol-bimy.onrender.com
  --timeout-sec   900
  --interval-sec  20
EOF
}

for arg in "$@"; do
  case "$arg" in
    --expected=*) EXPECTED_VERSION="${arg#*=}" ;;
    --prod-url=*) PROD_URL="${arg#*=}" ;;
    --timeout-sec=*) TIMEOUT_SEC="${arg#*=}" ;;
    --interval-sec=*) INTERVAL_SEC="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$EXPECTED_VERSION" ]]; then
  EXPECTED_VERSION="$(python3 - <<'PY'
import re
from pathlib import Path
t = Path("rag_server.py").read_text(encoding="utf-8")
m = re.search(r'^BUILD_VERSION\s*=\s*"([^"]+)"', t, re.M)
print(m.group(1) if m else "")
PY
)"
fi

if [[ -z "$EXPECTED_VERSION" ]]; then
  echo "ERROR: expected version is empty (no BUILD_VERSION found)." >&2
  exit 1
fi

echo "Waiting for Render version..."
echo "  URL:      ${PROD_URL%/}/api/version"
echo "  expected: $EXPECTED_VERSION"
echo "  timeout:  ${TIMEOUT_SEC}s"
echo "  interval: ${INTERVAL_SEC}s"

start_ts="$(date +%s)"
attempt=0

while true; do
  attempt=$((attempt + 1))
  now="$(date +%s)"
  elapsed=$((now - start_ts))

  ver="$(curl -fsS "${PROD_URL%/}/api/version" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null || echo "unreachable")"
  echo "check#$attempt elapsed=${elapsed}s version=$ver"

  if [[ "$ver" == "$EXPECTED_VERSION" ]]; then
    echo "DEPLOY_OK version=$ver"
    exit 0
  fi

  if [[ "$elapsed" -ge "$TIMEOUT_SEC" ]]; then
    echo "DEPLOY_TIMEOUT last_version=$ver target=$EXPECTED_VERSION"
    exit 1
  fi

  sleep "$INTERVAL_SEC"
done
