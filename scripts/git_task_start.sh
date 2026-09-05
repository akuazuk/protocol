#!/usr/bin/env bash
# Start a new task branch safely for multi-machine workflow.
# Default behavior: create a clean worktree from origin/main.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "" ]]; then
  echo "Usage: scripts/git_task_start.sh <task-slug> [--pc=pc1|pc2] [--base=origin/main] [--branch=feature/<slug>-pcX] [--worktree=/private/tmp/protocol-task-<slug>]"
  exit 0
fi

task_slug="$1"
shift

PC_TAG="pc1"
BASE_REF="origin/main"
BRANCH_NAME=""
WORKTREE_PATH=""

for arg in "$@"; do
  case "$arg" in
    --pc=*) PC_TAG="${arg#*=}" ;;
    --base=*) BASE_REF="${arg#*=}" ;;
    --branch=*) BRANCH_NAME="${arg#*=}" ;;
    --worktree=*) WORKTREE_PATH="${arg#*=}" ;;
    -h|--help)
      echo "Usage: scripts/git_task_start.sh <task-slug> [--pc=pc1|pc2] [--base=origin/main] [--branch=feature/<slug>-pcX] [--worktree=/private/tmp/protocol-task-<slug>]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BRANCH_NAME" ]]; then
  BRANCH_NAME="feature/${task_slug}-${PC_TAG}"
fi
if [[ -z "$WORKTREE_PATH" ]]; then
  WORKTREE_PATH="/private/tmp/protocol-task-${task_slug}-${PC_TAG}"
fi

git fetch origin -q

if ! git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
  echo "ERROR: base ref '$BASE_REF' not found." >&2
  exit 1
fi

if git show-ref --quiet "refs/heads/${BRANCH_NAME}"; then
  echo "INFO: local branch already exists: $BRANCH_NAME"
fi

if [[ -e "$WORKTREE_PATH" ]]; then
  if [[ -d "$WORKTREE_PATH/.git" || -f "$WORKTREE_PATH/.git" ]]; then
    echo "INFO: worktree already exists: $WORKTREE_PATH"
    wt_branch="$(git -C "$WORKTREE_PATH" rev-parse --abbrev-ref HEAD)"
    wt_sha="$(git -C "$WORKTREE_PATH" rev-parse --short HEAD)"
    echo "Current: $wt_branch @ $wt_sha"
  else
    echo "ERROR: path exists but is not a git worktree: $WORKTREE_PATH" >&2
    exit 1
  fi
else
  if git show-ref --quiet "refs/heads/${BRANCH_NAME}"; then
    git worktree add "$WORKTREE_PATH" "$BRANCH_NAME"
  else
    git worktree add -b "$BRANCH_NAME" "$WORKTREE_PATH" "$BASE_REF"
  fi
  echo "OK: created worktree $WORKTREE_PATH on branch $BRANCH_NAME from $BASE_REF"
fi

echo
echo "Next commands:"
echo "  cd \"$WORKTREE_PATH\""
echo "  scripts/git_safe_pull.sh"
echo "  scripts/ops/check_pr_file_overlap.sh"
echo "  # work -> tests -> commit -> push"
echo "  git push -u origin \"$BRANCH_NAME\""
echo "  # after another tab merges: scripts/ops/rebase_task_onto_main.sh"
echo "  scripts/git_deploy_guard.sh --prod-url=https://protocol-bimy.onrender.com"
