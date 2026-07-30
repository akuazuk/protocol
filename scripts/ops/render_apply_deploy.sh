#!/usr/bin/env bash
# Make sure the pushed commit actually reaches prod.
#
# Auto-deploy on push is configured on the service but has not fired for this repo
# (see docs/deploy/multi-machine-git-deploy-runbook.md, section 5.2), so with an API
# key we trigger the deploy explicitly instead of waiting for a webhook that may
# never arrive. Without a key we keep the old behaviour and just poll /api/version.
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
  1) trigger a deploy through the Render API (skipped without RENDER_API_KEY)
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
  echo "Triggering Render deploy for ${COMMIT_SHA:0:7} via API..."
  scripts/ops/render_deploy.sh deploy \
    --commit="$COMMIT_SHA" \
    --wait \
    --prod-url="$PROD_URL" \
    --timeout-sec="$WAIT_TIMEOUT_SEC" \
    --interval-sec="$WAIT_INTERVAL_SEC"
else
  echo "No RENDER_API_KEY: relying on Render auto-deploy for this push."
  echo "Auto-deploy has not been firing for this service - if the version below never"
  echo "changes, add the key to .env and rerun, or deploy from the dashboard."
fi

scripts/ops/render_wait_version.sh \
  --prod-url="$PROD_URL" \
  --timeout-sec="$WAIT_TIMEOUT_SEC" \
  --interval-sec="$WAIT_INTERVAL_SEC"
