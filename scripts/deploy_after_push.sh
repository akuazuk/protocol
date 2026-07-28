#!/usr/bin/env bash
# One-command flow for Render Git-connected deploy:
# push selected branch, then run strict deploy guard.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REMOTE_NAME="${REMOTE_NAME:-origin}"
RENDER_DEPLOY_BRANCH="${RENDER_DEPLOY_BRANCH:-main}"
PROD_URL="${PROTOCOL_PROD_URL:-https://protocol-bimy.onrender.com}"
PUSH_BRANCH="${PUSH_BRANCH:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_after_push.sh [--branch main] [--remote origin] [--prod-url URL]

Env:
  REMOTE_NAME=origin
  RENDER_DEPLOY_BRANCH=main
  PROTOCOL_PROD_URL=https://protocol-bimy.onrender.com

Example:
  scripts/deploy_after_push.sh --branch=main --prod-url=https://protocol-bimy.onrender.com
EOF
}

for arg in "$@"; do
  case "$arg" in
    --branch=*) PUSH_BRANCH="${arg#*=}" ;;
    --remote=*) REMOTE_NAME="${arg#*=}" ;;
    --prod-url=*) PROD_URL="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; usage; exit 2 ;;
  esac
done

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" == "HEAD" ]]; then
  echo "ERROR: detached HEAD not supported." >&2
  exit 1
fi

if [[ -z "$PUSH_BRANCH" ]]; then
  PUSH_BRANCH="$current_branch"
fi

if [[ "$PUSH_BRANCH" != "$RENDER_DEPLOY_BRANCH" ]]; then
  echo "ERROR: wrapper is for Render Git branch '$RENDER_DEPLOY_BRANCH', got '$PUSH_BRANCH'." >&2
  echo "Fix: switch/push '$RENDER_DEPLOY_BRANCH' or set RENDER_DEPLOY_BRANCH explicitly." >&2
  exit 1
fi

if [[ "$current_branch" != "$PUSH_BRANCH" ]]; then
  echo "ERROR: current branch '$current_branch' differs from push branch '$PUSH_BRANCH'." >&2
  echo "Fix: git switch '$PUSH_BRANCH'" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is dirty. Commit first." >&2
  git status --short >&2
  exit 1
fi

echo "Pushing '$PUSH_BRANCH' to '$REMOTE_NAME'..."
git push "$REMOTE_NAME" "$PUSH_BRANCH"

echo "Running Render Git deploy guard..."
scripts/git_deploy_guard.sh \
  --render-git \
  --render-branch="$RENDER_DEPLOY_BRANCH" \
  --prod-url="$PROD_URL"

echo
echo "OK: push + pre-deploy guard passed."
echo "Next: trigger/verify Render deploy for branch '$RENDER_DEPLOY_BRANCH'."
