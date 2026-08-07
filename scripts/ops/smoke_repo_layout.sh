#!/usr/bin/env bash
# Fast smoke-check for canonical script layout and wrappers.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "[1/7] bash syntax check for canonical wrappers"
for f in scripts/ops/*.sh scripts/deploy/*.sh scripts/data/*.sh scripts/dev/*.sh; do
  bash -n "$f"
done
echo "OK: wrapper syntax"

echo "[2/7] python wrapper compile check"
python3 -m py_compile scripts/data/py/*.py scripts/dev/py/*.py \
  scripts/mo/py/*.py scripts/mis/py/*.py scripts/corpus/py/*.py \
  services/mis_bridge/*.py services/llm_worker/*.py
echo "OK: python wrapper compile"

echo "[2b/7] domain README indexes exist"
for f in scripts/README.md scripts/mo/README.md scripts/mis/README.md scripts/corpus/README.md \
  docs/architecture/README.md docs/product/README.md \
  services/README.md services/api/README.md services/mo_pipeline/README.md \
  services/mis_bridge/README.md services/llm_worker/README.md \
  deploy/mac-bridge/extract-contract.md deploy/gcp-llm/job-contract.md \
  deploy/gcp-app/Dockerfile deploy/gcp-llm/Dockerfile deploy/mac-bridge/Dockerfile \
  deploy/by-home/Dockerfile requirements-llm-worker.txt requirements-mis-bridge.txt; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done
echo "OK: domain indexes"

echo "[3/7] ops commands expose help/usage"
scripts/ops/git_safe_start.sh --help >/dev/null
scripts/ops/git_safe_pull.sh --help >/dev/null
scripts/ops/git_deploy_guard.sh --help >/dev/null
scripts/ops/git_task_start.sh --help >/dev/null
scripts/ops/deploy_after_push.sh --help >/dev/null
scripts/ops/deploy_promote_main_after_push.sh --help >/dev/null
scripts/ops/render_wait_version.sh --help >/dev/null
scripts/ops/render_promote_main.sh --help >/dev/null
scripts/ops/render_deploy.sh --help >/dev/null
scripts/ops/render_apply_deploy.sh --help >/dev/null
scripts/ops/render_release_main.sh --help >/dev/null
scripts/ops/render_env.sh --help >/dev/null
scripts/ops/bump_build_version.sh --help >/dev/null
echo "OK: ops help"

echo "[4/7] deploy/data/dev wrappers forward safely"
if scripts/ops/render_promote_main.sh >/dev/null 2>&1; then
  echo "ERROR: deprecated render_promote_main.sh must fail closed" >&2
  exit 1
fi
if scripts/ops/deploy_promote_main_after_push.sh >/dev/null 2>&1; then
  echo "ERROR: deprecated deploy_promote_main_after_push.sh must fail closed" >&2
  exit 1
fi
scripts/ops/render_release_main.sh --commit="$(git rev-parse origin/main)" --dry-run >/dev/null
scripts/deploy/render_mis_protocol_data.sh >/dev/null 2>&1 || true
scripts/data/pull_methodist_feedback.sh >/dev/null 2>&1 || true
scripts/dev/run_mo_daily_launchd.sh unknown >/dev/null 2>&1 || true
echo "OK: wrapper forward smoke"

echo "[5/7] hygiene command runs"
scripts/ops/check_repo_hygiene.sh >/tmp/protocol_hygiene_smoke.out 2>&1 || true
echo "OK: hygiene smoke"

echo "[6/7] service contour CLI dry-run"
PYTHONPATH=. python3 -m services.mis_bridge.extract_day --day 2026-08-06 --dry-run >/dev/null
tmpdir="$(mktemp -d)"
mkdir -p "$tmpdir/llm_outbox/smoke-run"
cp tests/fixtures/llm_job/manifest.json tests/fixtures/llm_job/cases.jsonl "$tmpdir/llm_outbox/smoke-run/"
PYTHONPATH=. python3 -m services.llm_worker.grade_day \
  --day 2026-08-06 --run-id smoke-run --data-root "$tmpdir" --dry-run >/dev/null
rm -rf "$tmpdir"
echo "OK: service contour dry-run"

echo "SMOKE_OK repo layout wrappers"
