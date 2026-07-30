#!/usr/bin/env bash
# Pick the next free BUILD_VERSION and write it into rag_server.py.
#
# Two machines working in parallel kept choosing the same rN and hitting a merge conflict
# in rag_server.py, so the number is derived from every branch on the remote instead of
# from the local file alone.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

REMOTE_NAME="${REMOTE_NAME:-origin}"
DRY_RUN=0
ALLOW_STALE="${ALLOW_STALE:-0}"
SLUG=""

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/bump_build_version.sh <slug> [--dry-run] [--allow-stale] [--remote=origin]

Example:
  scripts/ops/bump_build_version.sh render-env-tool
  -> BUILD_VERSION = "2026-07-30-r15-render-env-tool"

The slug is 2-4 latin words in kebab-case describing the commit.
The number is one above the highest rN used today on any remote branch or locally.
If the remote cannot be fetched the command fails, because a stale view of the
branches is how two machines end up on the same number; --allow-stale overrides.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --allow-stale) ALLOW_STALE=1 ;;
    --remote=*) REMOTE_NAME="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    *) SLUG="$arg" ;;
  esac
done

if [[ -z "$SLUG" ]]; then
  echo "ERROR: slug is required." >&2
  usage >&2
  exit 2
fi

if [[ ! "$SLUG" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  echo "ERROR: slug must be latin kebab-case, got '$SLUG'." >&2
  exit 2
fi

if ! git fetch "$REMOTE_NAME" -q; then
  # Stale refs are exactly how a colliding number gets picked, so this is fatal by default.
  if [[ "$ALLOW_STALE" != "1" ]]; then
    cat >&2 <<EOF
ERROR: cannot fetch '$REMOTE_NAME', so branches from other machines are invisible
and the number picked here may already be taken.

Retry when the network is back, or accept the risk explicitly:
  scripts/ops/bump_build_version.sh $SLUG --allow-stale
EOF
    exit 1
  fi
  echo "WARNING: fetch failed, using possibly stale refs (--allow-stale)" >&2
fi

today="$(date +%Y-%m-%d)"
versions_file="$(mktemp)"
trap 'rm -f "$versions_file"' EXIT

extract_version() {
  grep -m1 -E '^BUILD_VERSION[[:space:]]*=' 2>/dev/null | sed -E 's/.*"([^"]+)".*/\1/' || true
}

# Local working copy, plus every branch on the remote.
extract_version < rag_server.py >> "$versions_file"
while read -r ref; do
  git show "${ref}:rag_server.py" 2>/dev/null | extract_version >> "$versions_file" || true
done < <(git for-each-ref --format='%(refname:short)' "refs/remotes/${REMOTE_NAME}" | grep -v "^${REMOTE_NAME}$")

next_num="$(python3 - "$today" "$versions_file" <<'PY'
import re
import sys

today, path = sys.argv[1], sys.argv[2]
best = 0
with open(path, encoding="utf-8") as fh:
    for line in fh:
        m = re.match(rf"{re.escape(today)}-r(\d+)", line.strip())
        if m:
            best = max(best, int(m.group(1)))
print(best + 1)
PY
)"

new_version="${today}-r${next_num}-${SLUG}"
current="$(extract_version < rag_server.py)"

echo "current: $current"
echo "next:    $new_version"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run, file not changed)"
  exit 0
fi

python3 - "$new_version" <<'PY'
import re
import sys
from pathlib import Path

new_version = sys.argv[1]
path = Path("rag_server.py")
text = path.read_text(encoding="utf-8")
text, count = re.subn(
    r'^BUILD_VERSION\s*=\s*"[^"]+"',
    f'BUILD_VERSION = "{new_version}"',
    text,
    count=1,
    flags=re.M,
)
if count != 1:
    raise SystemExit("ERROR: BUILD_VERSION not found in rag_server.py")
path.write_text(text, encoding="utf-8")
PY

echo "updated rag_server.py"
