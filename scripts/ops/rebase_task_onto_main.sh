#!/usr/bin/env bash
# Rebase current task branch onto origin/main.
# If the only conflict is BUILD_VERSION in rag_server.py, resolve and re-stamp.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/ops/rebase_task_onto_main.sh

Fetch origin/main and rebase this task branch onto it.
BUILD_VERSION-only conflicts in rag_server.py are resolved automatically.
Any other conflict aborts the rebase and prints the files.
EOF
  exit 0
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: dirty worktree. Commit or move aside first." >&2
  exit 1
fi

git fetch --prune origin

if git merge-base --is-ancestor origin/main HEAD; then
  echo "OK: already contains origin/main"
  exit 0
fi

resolve_one() {
  local conflicts extra f
  conflicts="$(git diff --name-only --diff-filter=U || true)"
  if [[ -z "$conflicts" ]]; then
    return 0
  fi
  extra=0
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if [[ "$f" != "rag_server.py" ]]; then
      extra=1
    fi
  done <<< "$conflicts"
  if [[ "$extra" -ne 0 ]]; then
    echo "ERROR: conflicts beyond BUILD_VERSION:" >&2
    echo "$conflicts" >&2
    echo "Aborting rebase. Resolve with the other PR owner or wait for their merge." >&2
    git rebase --abort
    return 2
  fi
  local tmp
  tmp="$(mktemp -d)"
  git show :1:rag_server.py >"$tmp/base"
  git show :2:rag_server.py >"$tmp/ours"
  git show :3:rag_server.py >"$tmp/theirs"
  if ! python3 "$ROOT/scripts/ops/pr_isolation.py" resolve-rag-server \
    --base "$tmp/base" --ours "$tmp/ours" --theirs "$tmp/theirs" \
    --out rag_server.py --slug-out "$tmp/slug"; then
    echo "ERROR: rag_server.py has a real conflict, not only BUILD_VERSION." >&2
    git rebase --abort
    rm -rf "$tmp"
    return 2
  fi
  local slug
  slug="$(tr -d '[:space:]' <"$tmp/slug")"
  rm -rf "$tmp"
  if [[ -n "$slug" ]]; then
    "$ROOT/scripts/ops/bump_build_version.sh" "$slug"
  else
    "$ROOT/scripts/ops/bump_build_version.sh"
  fi
  git add rag_server.py
  set +e
  GIT_EDITOR=true git rebase --continue
  cont_rc=$?
  set -e
  if [[ "$cont_rc" -eq 0 ]]; then
    return 0
  fi
  if [[ -d "$(git rev-parse --git-path rebase-merge)" ]]; then
    return 0
  fi
  echo "ERROR: git rebase --continue failed." >&2
  return "$cont_rc"
}

set +e
git rebase origin/main
rebase_rc=$?
set -e
if [[ "$rebase_rc" -eq 0 ]]; then
  echo "OK: rebased onto origin/main"
  exit 0
fi

while git rev-parse --git-path rebase-merge >/dev/null 2>&1 \
  && [[ -d "$(git rev-parse --git-path rebase-merge)" ]]; do
  resolve_one || exit $?
done

if git merge-base --is-ancestor origin/main HEAD; then
  echo "OK: rebased onto origin/main (BUILD_VERSION auto-resolved)"
  exit 0
fi

echo "ERROR: rebase did not finish. Check git status." >&2
exit 1
