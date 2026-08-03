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
REMOTE_NAME="${REMOTE_NAME:-origin}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-20}"
COMMIT_SHA=""
EXPECTED_VERSION="${EXPECTED_VERSION:-}"
TRIGGER=1

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/render_apply_deploy.sh [--prod-url URL] [--commit SHA] [--no-trigger]
                                     [--timeout-sec N] [--interval-sec N]

Steps:
  1) require the commit to equal current origin/main
  2) follow the deploy of this commit, triggering one if the push did not (needs RENDER_API_KEY)
  3) wait until /api/version matches BUILD_VERSION from that commit
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
  echo "ERROR: --commit is required; local HEAD is never an implicit release source." >&2
  exit 2
fi

git fetch "$REMOTE_NAME" "$TARGET_BRANCH" -q
main_sha="$(git rev-parse "${REMOTE_NAME}/${TARGET_BRANCH}^{commit}")"
commit_sha="$(git rev-parse "${COMMIT_SHA}^{commit}" 2>/dev/null || true)"
if [[ -z "$commit_sha" || "$commit_sha" != "$main_sha" ]]; then
  echo "ERROR: deploy commit must equal current ${REMOTE_NAME}/${TARGET_BRANCH}." >&2
  echo "requested=${commit_sha:-invalid} ${REMOTE_NAME}/${TARGET_BRANCH}=$main_sha" >&2
  exit 1
fi
COMMIT_SHA="$commit_sha"

if [[ -z "$EXPECTED_VERSION" ]]; then
  EXPECTED_VERSION="$(
    git show "${COMMIT_SHA}:rag_server.py" \
      | python3 -c 'import re,sys; m=re.search(r"^BUILD_VERSION\s*=\s*\"([^\"]+)\"", sys.stdin.read(), re.M); print(m.group(1) if m else "")'
  )"
fi
if [[ -z "$EXPECTED_VERSION" ]]; then
  echo "ERROR: expected BUILD_VERSION is empty for $COMMIT_SHA." >&2
  exit 1
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
  --expected="$EXPECTED_VERSION" \
  --prod-url="$PROD_URL" \
  --timeout-sec="$WAIT_TIMEOUT_SEC" \
  --interval-sec="$WAIT_INTERVAL_SEC"
