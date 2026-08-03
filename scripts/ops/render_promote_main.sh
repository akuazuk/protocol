#!/usr/bin/env bash
# Deprecated command kept only to block direct task HEAD promotion.
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

This command is disabled. Use PR merge followed by:
  scripts/ops/render_release_main.sh --commit="$(git rev-parse origin/main)"
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

cat >&2 <<'EOF'
ERROR: direct promotion of task HEAD to main is permanently disabled.

Required workflow:
  task branch -> PR -> merge -> origin/main -> Render

After merge, run:
  git fetch origin
  scripts/ops/render_release_main.sh --commit="$(git rev-parse origin/main)"
EOF
exit 64
