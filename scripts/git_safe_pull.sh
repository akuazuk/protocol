#!/usr/bin/env bash
# Safe pull wrapper for multi-machine work.
# Aborts on dirty tree or branch divergence and prints exact recovery commands.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STRICT="${STRICT:-1}"
DEFAULT_BASE_REMOTE="${DEFAULT_BASE_REMOTE:-origin}"

usage() {
  cat <<'EOF'
Usage:
  scripts/git_safe_pull.sh [--allow-dirty] [--remote origin]

Options:
  --allow-dirty   do not block on dirty tree (not recommended)
  --remote NAME   remote name (default: origin)
EOF
}

ALLOW_DIRTY=0
for arg in "$@"; do
  case "$arg" in
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --remote=*) DEFAULT_BASE_REMOTE="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; usage; exit 2 ;;
  esac
done

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" == "HEAD" ]]; then
  echo "ERROR: detached HEAD is not supported for safe pull." >&2
  exit 1
fi

if [[ "$ALLOW_DIRTY" != "1" ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is dirty. Safe pull blocked." >&2
  echo "Fix: commit/move changes or work in a clean worktree first." >&2
  git status --short >&2
  exit 1
fi

git fetch "$DEFAULT_BASE_REMOTE" -q

upstream_ref=""
if upstream_ref="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"; then
  :
elif git show-ref --quiet "refs/remotes/${DEFAULT_BASE_REMOTE}/${branch}"; then
  upstream_ref="${DEFAULT_BASE_REMOTE}/${branch}"
  git branch --set-upstream-to="$upstream_ref" "$branch" >/dev/null
else
  echo "ERROR: no upstream for branch '$branch' and ${DEFAULT_BASE_REMOTE}/${branch} missing." >&2
  exit 1
fi

ahead="$(git rev-list --count "${upstream_ref}..HEAD" || echo 0)"
behind="$(git rev-list --count "HEAD..${upstream_ref}" || echo 0)"

echo "Branch:   $branch"
echo "Upstream: $upstream_ref"
echo "ahead=$ahead behind=$behind"

if [[ "$ahead" -eq 0 && "$behind" -eq 0 ]]; then
  echo "OK: already up to date."
  exit 0
fi

if [[ "$ahead" -eq 0 && "$behind" -gt 0 ]]; then
  echo "Running: git pull --ff-only"
  git pull --ff-only "$DEFAULT_BASE_REMOTE" "$branch"
  echo "OK: fast-forward pull completed."
  exit 0
fi

if [[ "$ahead" -gt 0 && "$behind" -eq 0 ]]; then
  echo "INFO: local branch is ahead only - nothing to pull."
  echo "Next: git push -u $DEFAULT_BASE_REMOTE $branch"
  exit 0
fi

echo "ERROR: branch diverged (ahead and behind)." >&2
echo "Safe pull stopped. Choose one path:" >&2
echo "  1) Rebase local commits: git rebase ${upstream_ref}" >&2
echo "  2) Keep current tree and work from clean sync worktree:" >&2
echo "     AUTO_WORKTREE=1 scripts/git_safe_start.sh --auto-worktree" >&2
exit 1
