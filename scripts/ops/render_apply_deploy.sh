#!/usr/bin/env bash
# Make sure the pushed commit actually reaches prod.
#
# With an API key we follow the deploy of this exact commit: normally the one the push
# webhook already created, and otherwise one we trigger ourselves. Without a key there is
# nothing to watch, so we fall back to polling /api/version.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROD_URL="${PROTOCOL_PROD_URL:-https://protocol-bimy.onrender.com}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-20}"
COMMIT_SHA=""
TRIGGER=1

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/render_apply_deploy.sh [--prod-url URL] [--commit SHA] [--no-trigger]
                                     [--timeout-sec N] [--interval-sec N]

Steps:
  1) follow the deploy of this commit, triggering one if the push did not (needs RENDER_API_KEY)
  2) wait until the deploy is live and /api/version matches local BUILD_VERSION
EOF
}

for arg in "$@"; do
  case "$arg" in
    --prod-url=*) PROD_URL="${arg#*=}" ;;
    --commit=*) COMMIT_SHA="${arg#*=}" ;;
    --no-trigger) TRIGGER=0 ;;
    --timeout-sec=*) WAIT_TIMEOUT_SEC="${arg#*=}" ;;
    --interval-sec=*) WAIT_INTERVAL_SEC="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$COMMIT_SHA" ]]; then
  COMMIT_SHA="$(git rev-parse HEAD)"
fi

if [[ "$TRIGGER" == "1" ]] && scripts/ops/render_deploy.sh has-key >/dev/null 2>&1; then
  scripts/ops/render_deploy.sh ensure-deploy \
    --commit="$COMMIT_SHA" \
    --wait \
    --prod-url="$PROD_URL" \
    --timeout-sec="$WAIT_TIMEOUT_SEC" \
    --interval-sec="$WAIT_INTERVAL_SEC"
else
  echo "No RENDER_API_KEY: relying on the push webhook for this deploy."
fi

scripts/ops/render_wait_version.sh \
  --prod-url="$PROD_URL" \
  --timeout-sec="$WAIT_TIMEOUT_SEC" \
  --interval-sec="$WAIT_INTERVAL_SEC"
