#!/usr/bin/env bash
# Promote current HEAD to Render deploy branch (main) safely.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

REMOTE_NAME="${REMOTE_NAME:-origin}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
PROD_URL="${PROTOCOL_PROD_URL:-https://protocol-bimy.onrender.com}"
WAIT_RENDER_VERSION="${WAIT_RENDER_VERSION:-1}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-20}"

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/render_promote_main.sh [--remote origin] [--target-branch main] [--prod-url URL] [--no-wait-version]

What it does:
  1) verifies clean git state
  2) ensures origin/main is ancestor of current HEAD (fast-forward safe)
  3) pushes HEAD -> origin/main
  4) optionally waits until /api/version matches local BUILD_VERSION
EOF
}

for arg in "$@"; do
  case "$arg" in
    --remote=*) REMOTE_NAME="${arg#*=}" ;;
    --target-branch=*) TARGET_BRANCH="${arg#*=}" ;;
    --prod-url=*) PROD_URL="${arg#*=}" ;;
    --no-wait-version) WAIT_RENDER_VERSION=0 ;;
    --wait-version) WAIT_RENDER_VERSION=1 ;;
    --timeout-sec=*) WAIT_TIMEOUT_SEC="${arg#*=}" ;;
    --interval-sec=*) WAIT_INTERVAL_SEC="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; usage; exit 2 ;;
  esac
done

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" == "HEAD" ]]; then
  echo "ERROR: detached HEAD is not supported." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is dirty. Commit first." >&2
  git status --short >&2
  exit 1
fi

git fetch "$REMOTE_NAME" -q

if ! git show-ref --quiet "refs/remotes/${REMOTE_NAME}/${TARGET_BRANCH}"; then
  echo "ERROR: ${REMOTE_NAME}/${TARGET_BRANCH} not found." >&2
  exit 1
fi

if ! git merge-base --is-ancestor "${REMOTE_NAME}/${TARGET_BRANCH}" HEAD; then
  echo "ERROR: ${REMOTE_NAME}/${TARGET_BRANCH} is not ancestor of current HEAD." >&2
  echo "Fix: rebase/merge ${REMOTE_NAME}/${TARGET_BRANCH} into '$branch' before promote." >&2
  exit 1
fi

echo "Promote: HEAD ($branch) -> ${REMOTE_NAME}/${TARGET_BRANCH}"
git push "$REMOTE_NAME" "HEAD:${TARGET_BRANCH}"

echo "Verifying remote branch tip..."
remote_tip="$(git ls-remote --heads "$REMOTE_NAME" "$TARGET_BRANCH" | awk '{print $1}')"
local_tip="$(git rev-parse HEAD)"
if [[ -z "$remote_tip" || "$remote_tip" != "$local_tip" ]]; then
  echo "ERROR: remote ${TARGET_BRANCH} tip differs after push." >&2
  echo "local=$local_tip remote=${remote_tip:-missing}" >&2
  exit 1
fi
echo "OK: ${REMOTE_NAME}/${TARGET_BRANCH} at $local_tip"

if [[ "$WAIT_RENDER_VERSION" == "1" ]]; then
  scripts/ops/render_wait_version.sh \
    --prod-url="$PROD_URL" \
    --timeout-sec="$WAIT_TIMEOUT_SEC" \
    --interval-sec="$WAIT_INTERVAL_SEC"
else
  echo "Skip wait. Use:"
  echo "  scripts/ops/render_wait_version.sh --prod-url=$PROD_URL"
fi
