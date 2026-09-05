#!/usr/bin/env bash
# Compare files on this branch vs other open PRs. Does not fail the required CI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/ops/check_pr_file_overlap.sh

Prints open PRs that touch the same files as origin/main...HEAD.
Exit 0 if no hard overlap, 1 if another PR shares a non-version file.
Missing gh is a warning, not a failure.
EOF
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "WARN: gh not found; skip overlap check." >&2
  exit 0
fi

git fetch --prune origin >/dev/null

ours_csv=""
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  if [[ -n "$ours_csv" ]]; then
    ours_csv+=",$f"
  else
    ours_csv="$f"
  fi
done < <(git diff --name-only origin/main...HEAD)

if [[ -z "$ours_csv" ]]; then
  echo "OK: no files vs origin/main yet"
  exit 0
fi

rag_only=0
if git diff --name-only origin/main...HEAD | grep -qx "rag_server.py"; then
  if python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, r'''$ROOT/scripts/ops''')
from pr_isolation import is_build_version_only_diff
diff = sys.stdin.read()
raise SystemExit(0 if is_build_version_only_diff(diff) else 1)
" < <(git diff origin/main...HEAD -- rag_server.py); then
    rag_only=1
  fi
fi

others_json="$(gh pr list --repo akuazuk/protocol --state open --json number,title,url,headRefName,files)"
current_branch="$(git rev-parse --abbrev-ref HEAD)"

python3 - "$ROOT" "$ours_csv" "$others_json" "$current_branch" "$rag_only" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "scripts" / "ops"))
from pr_isolation import classify_overlap

ours = [p for p in sys.argv[2].split(",") if p]
payload = json.loads(sys.argv[3])
branch = sys.argv[4]
our_rag_only = sys.argv[5] == "1"
hard_any = False
print("This branch files:", ", ".join(ours))
print()
for pr in payload:
    if pr.get("headRefName") == branch:
        continue
    their = [f.get("path") for f in (pr.get("files") or []) if f.get("path")]
    kind = classify_overlap(ours, their, our_rag_only_version=our_rag_only)
    if not kind["all"]:
        continue
    label = "HARD" if kind["hard"] else "soft"
    if kind["hard"]:
        hard_any = True
    print(
        f"{label} #{pr['number']} {pr.get('title')}: "
        f"hard={kind['hard'] or '-'} soft={kind['soft'] or '-'}"
    )
    print(pr.get("url") or "")
if not hard_any:
    print("OK: no hard file overlap with other open PRs")
    sys.exit(0)
print("STOP: wait for the other PR or split files. Do not merge both blindly.")
sys.exit(1)
PY
