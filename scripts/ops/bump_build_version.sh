#!/usr/bin/env bash
# Write BUILD_VERSION into rag_server.py as UTC timestamp to the second.
#
# Old rN counters collided when two machines/agents picked the same day number.
# Timestamp UTC (YYYY-MM-DD-HHMMSSZ) makes parallel bumps unique without fetching remotes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DRY_RUN=0
SLUG=""

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/bump_build_version.sh [slug] [--dry-run]

Examples:
  scripts/ops/bump_build_version.sh
  -> BUILD_VERSION = "2026-08-06-033512Z"

  scripts/ops/bump_build_version.sh recompute-csv
  -> BUILD_VERSION = "2026-08-06-033512Z-recompute-csv"

Slug (optional): latin kebab-case, 2-4 words describing the commit.
Time is always UTC to the second.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    *) SLUG="$arg" ;;
  esac
done

if [[ -n "$SLUG" && ! "$SLUG" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  echo "ERROR: slug must be latin kebab-case, got '$SLUG'." >&2
  exit 2
fi

stamp="$(date -u +%Y-%m-%d-%H%M%SZ)"
if [[ -n "$SLUG" ]]; then
  new_version="${stamp}-${SLUG}"
else
  new_version="${stamp}"
fi

extract_version() {
  grep -m1 -E '^BUILD_VERSION[[:space:]]*=' 2>/dev/null | sed -E 's/.*"([^"]+)".*/\1/' || true
}

current="$(extract_version < rag_server.py)"

# Same-second collision (two agents in parallel): wait 1s and rebuild stamp.
if [[ "$current" == "$new_version" || "$current" == "${stamp}-"* ]]; then
  sleep 1
  stamp="$(date -u +%Y-%m-%d-%H%M%SZ)"
  if [[ -n "$SLUG" ]]; then
    new_version="${stamp}-${SLUG}"
  else
    new_version="${stamp}"
  fi
fi

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
