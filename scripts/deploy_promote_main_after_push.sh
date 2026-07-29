#!/usr/bin/env bash
# Push current branch, then safely promote HEAD to Render branch (main).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REMOTE_NAME="${REMOTE_NAME:-origin}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
PROD_URL="${PROTOCOL_PROD_URL:-https://protocol-bimy.onrender.com}"
WAIT_RENDER_VERSION="${WAIT_RENDER_VERSION:-1}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-20}"
SKIP_PUSH="${SKIP_PUSH:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_promote_main_after_push.sh [--remote=origin] [--target-branch=main] [--prod-url=URL] [--wait-version] [--no-wait-version] [--no-push]

What it does:
  1) verifies current branch and clean git state
  2) pushes current branch to remote (unless --no-push)
  3) promotes current HEAD to target Render branch (fast-forward only)
  4) waits for /api/version to match local BUILD_VERSION (default: on)
EOF
}

for arg in "$@"; do
  case "$arg" in
    --remote=*) REMOTE_NAME="${arg#*=}" ;;
    --target-branch=*) TARGET_BRANCH="${arg#*=}" ;;
    --prod-url=*) PROD_URL="${arg#*=}" ;;
    --wait-version) WAIT_RENDER_VERSION=1 ;;
    --no-wait-version) WAIT_RENDER_VERSION=0 ;;
    --timeout-sec=*) WAIT_TIMEOUT_SEC="${arg#*=}" ;;
    --interval-sec=*) WAIT_INTERVAL_SEC="${arg#*=}" ;;
    --no-push) SKIP_PUSH=1 ;;
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

if [[ "$SKIP_PUSH" != "1" ]]; then
  echo "Pushing current branch '$branch' to '$REMOTE_NAME'..."
  git push "$REMOTE_NAME" "$branch"
fi

echo "Promoting current HEAD to ${REMOTE_NAME}/${TARGET_BRANCH}..."
args=(
  "--remote=${REMOTE_NAME}"
  "--target-branch=${TARGET_BRANCH}"
  "--prod-url=${PROD_URL}"
  "--timeout-sec=${WAIT_TIMEOUT_SEC}"
  "--interval-sec=${WAIT_INTERVAL_SEC}"
)
if [[ "$WAIT_RENDER_VERSION" != "1" ]]; then
  args+=("--no-wait-version")
fi
scripts/ops/render_promote_main.sh "${args[@]}"
