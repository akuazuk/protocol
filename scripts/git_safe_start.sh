#!/usr/bin/env bash
# Multi-machine safe session start.
# - verifies local state
# - fetches origin
# - prints actionable branch sync hints
# - can auto-create clean sync worktree from origin/main
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

AUTO_WORKTREE="${AUTO_WORKTREE:-0}"
SYNC_WORKTREE_PATH="${SYNC_WORKTREE_PATH:-/private/tmp/protocol-main-sync}"
SYNC_BRANCH="${SYNC_BRANCH:-codex/main-sync}"
DEFAULT_BASE="${DEFAULT_BASE:-origin/main}"

usage() {
  cat <<'EOF'
Usage:
  scripts/git_safe_start.sh [--auto-worktree]

Options:
  --auto-worktree  create/update clean sync worktree from origin/main

Env:
  AUTO_WORKTREE=1
  SYNC_WORKTREE_PATH=/private/tmp/protocol-main-sync
  SYNC_BRANCH=codex/main-sync
EOF
}

for arg in "$@"; do
  case "$arg" in
    --auto-worktree) AUTO_WORKTREE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; usage; exit 2 ;;
  esac
done

branch="$(git rev-parse --abbrev-ref HEAD)"
short_sha="$(git rev-parse --short HEAD)"

echo "Repo:    $ROOT"
echo "Branch:  $branch"
echo "HEAD:    $short_sha"
echo

echo "[1/4] Local status"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "WARN: working tree is not clean."
  git status --short
else
  echo "OK: working tree is clean."
fi
echo

echo "[2/4] Fetch origin"
git fetch origin -q
echo "OK: origin fetched."
echo

echo "[3/4] Divergence vs origin/main"
if git show-ref --quiet refs/remotes/origin/main; then
  ahead="$(git rev-list --count origin/main..HEAD || echo 0)"
  behind="$(git rev-list --count HEAD..origin/main || echo 0)"
  echo "ahead=$ahead behind=$behind (vs origin/main)"
  if [[ "$ahead" -gt 0 && "$behind" -gt 0 ]]; then
    echo "WARN: diverged from origin/main."
  elif [[ "$ahead" -gt 0 ]]; then
    echo "INFO: local-only commits present."
  elif [[ "$behind" -gt 0 ]]; then
    echo "INFO: remote has newer commits."
  else
    echo "OK: HEAD aligned with origin/main."
  fi
else
  echo "WARN: origin/main not found."
fi
echo

echo "[4/4] Safe next step"
if [[ "$AUTO_WORKTREE" != "1" ]]; then
  cat <<EOF
Recommended command:
  AUTO_WORKTREE=1 scripts/git_safe_start.sh --auto-worktree
EOF
  exit 0
fi

if [[ -d "$SYNC_WORKTREE_PATH/.git" || -f "$SYNC_WORKTREE_PATH/.git" ]]; then
  echo "Worktree already exists: $SYNC_WORKTREE_PATH"
  wt_branch="$(git -C "$SYNC_WORKTREE_PATH" rev-parse --abbrev-ref HEAD)"
  wt_sha="$(git -C "$SYNC_WORKTREE_PATH" rev-parse --short HEAD)"
  echo "Current: $wt_branch @ $wt_sha"
  git -C "$SYNC_WORKTREE_PATH" fetch origin -q
  if [[ "$wt_branch" == "$SYNC_BRANCH" ]]; then
    git -C "$SYNC_WORKTREE_PATH" pull --ff-only origin main >/dev/null
    echo "Updated: $SYNC_WORKTREE_PATH -> origin/main"
  else
    echo "WARN: worktree branch is '$wt_branch' (expected '$SYNC_BRANCH'), not auto-updating."
  fi
else
  git worktree add -b "$SYNC_BRANCH" "$SYNC_WORKTREE_PATH" "$DEFAULT_BASE"
  echo "Created: $SYNC_WORKTREE_PATH ($SYNC_BRANCH from $DEFAULT_BASE)"
fi

echo
echo "Use this clean tree for active work:"
echo "  cd \"$SYNC_WORKTREE_PATH\""
