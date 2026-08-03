#!/usr/bin/env bash
# Fast smoke-check for canonical script layout and wrappers.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "[1/5] bash syntax check for canonical wrappers"
for f in scripts/ops/*.sh scripts/deploy/*.sh scripts/data/*.sh scripts/dev/*.sh; do
  bash -n "$f"
done
echo "OK: wrapper syntax"

echo "[2/5] python wrapper compile check"
python3 -m py_compile scripts/data/py/*.py scripts/dev/py/*.py
echo "OK: python wrapper compile"

echo "[3/5] ops commands expose help/usage"
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

echo "[4/5] deploy/data/dev wrappers forward safely"
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

echo "[5/5] hygiene command runs"
scripts/ops/check_repo_hygiene.sh >/tmp/protocol_hygiene_smoke.out 2>&1 || true
echo "OK: hygiene smoke"

echo "SMOKE_OK repo layout wrappers"
