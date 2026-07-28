#!/usr/bin/env bash
# Deploy preflight guard for multi-machine workflow.
# Validates branch policy, git sync status and BUILD_VERSION format.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ALLOWED_BRANCHES_DEFAULT="main release/* hotfix/* codex/main-sync"
ALLOWED_BRANCHES="${ALLOWED_BRANCHES:-$ALLOWED_BRANCHES_DEFAULT}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
PROD_URL="${PROTOCOL_PROD_URL:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/git_deploy_guard.sh [--prod-url URL]

Env:
  ALLOWED_BRANCHES="main release/* hotfix/* codex/main-sync"
  REMOTE_NAME=origin
  PROTOCOL_PROD_URL=https://protocol-bimy.onrender.com
EOF
}

for arg in "$@"; do
  case "$arg" in
    --prod-url=*) PROD_URL="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; usage; exit 2 ;;
  esac
done

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" == "HEAD" ]]; then
  echo "ERROR: detached HEAD deploy is forbidden." >&2
  exit 1
fi

allowed=0
for pattern in $ALLOWED_BRANCHES; do
  if [[ "$branch" == $pattern ]]; then
    allowed=1
    break
  fi
done
if [[ "$allowed" -ne 1 ]]; then
  echo "ERROR: branch '$branch' is not in deploy allowlist: $ALLOWED_BRANCHES" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is dirty. Deploy blocked." >&2
  git status --short >&2
  exit 1
fi

git fetch "$REMOTE_NAME" -q

if ! upstream_ref="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"; then
  if git show-ref --quiet "refs/remotes/${REMOTE_NAME}/${branch}"; then
    upstream_ref="${REMOTE_NAME}/${branch}"
  else
    echo "ERROR: no upstream configured and ${REMOTE_NAME}/${branch} not found." >&2
    exit 1
  fi
fi

ahead="$(git rev-list --count "${upstream_ref}..HEAD" || echo 0)"
behind="$(git rev-list --count "HEAD..${upstream_ref}" || echo 0)"

if [[ "$ahead" -gt 0 ]]; then
  echo "ERROR: local branch has unpushed commits (ahead=$ahead). Push first." >&2
  exit 1
fi
if [[ "$behind" -gt 0 ]]; then
  echo "ERROR: local branch is behind upstream (behind=$behind). Sync first." >&2
  exit 1
fi

build_version="$(python3 - <<'PY'
import re
from pathlib import Path
t = Path("rag_server.py").read_text(encoding="utf-8")
m = re.search(r'^BUILD_VERSION\s*=\s*"([^"]+)"', t, re.M)
print(m.group(1) if m else "")
PY
)"
if [[ -z "$build_version" ]]; then
  echo "ERROR: BUILD_VERSION not found in rag_server.py" >&2
  exit 1
fi
if [[ ! "$build_version" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}-r[0-9]+-[a-z0-9-]+$ ]]; then
  echo "ERROR: BUILD_VERSION has invalid format: $build_version" >&2
  exit 1
fi

echo "OK: deploy guard passed."
echo "Branch:        $branch"
echo "Upstream:      $upstream_ref"
echo "BUILD_VERSION: $build_version"

if [[ -n "$PROD_URL" ]]; then
  prod_ver="$(curl -fsS "${PROD_URL%/}/api/version" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null || true)"
  if [[ -n "$prod_ver" ]]; then
    if [[ "$prod_ver" == "$build_version" ]]; then
      echo "INFO: prod already on same version: $prod_ver"
    else
      echo "INFO: prod version differs: prod=$prod_ver local=$build_version"
    fi
  else
    echo "WARN: could not read ${PROD_URL%/}/api/version"
  fi
fi
