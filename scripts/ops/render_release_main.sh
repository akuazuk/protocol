#!/usr/bin/env bash
# LEGACY. Render больше не прод: сервис protocol приостановлен (503).
# Прод - GCE, https://protocol.kravira.by. См. docs/deploy/gce-production-runbook.md
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ "${ALLOW_LEGACY_RENDER_RELEASE:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
ОТКАЗ: Render не является продом Protocol.

Сервис protocol на Render приостановлен и отдаёт 503. Прод развёрнут на GCE:
  домен  https://protocol.kravira.by
  VM     protocol-app, зона europe-central2-a

Релиз выполняется так:
  bash deploy/gcp-app/deploy_to_gce.sh          # с HEAD ровно на origin/main
  либо GitHub Action "Production GCE release"

Если восстанавливаете именно старый Render-контур осознанно:
  ALLOW_LEGACY_RENDER_RELEASE=1 scripts/ops/render_release_main.sh ...
EOF
  exit 2
fi

REMOTE_NAME="${REMOTE_NAME:-origin}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
PROD_URL="${PROTOCOL_PROD_URL:-https://protocol-bimy.onrender.com}"
COMMIT_SHA=""
DRY_RUN=0
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-20}"

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/render_release_main.sh --commit=MERGE_SHA [--prod-url=URL] [--dry-run]

The commit is mandatory and must equal the current origin/main HEAD. The script never
pushes or promotes a task branch. It deploys/watches that exact merge commit and verifies
both BUILD_VERSION and RENDER_GIT_COMMIT in production.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --commit=*) COMMIT_SHA="${arg#*=}" ;;
    --remote=*) REMOTE_NAME="${arg#*=}" ;;
    --target-branch=*) TARGET_BRANCH="${arg#*=}" ;;
    --prod-url=*) PROD_URL="${arg#*=}" ;;
    --timeout-sec=*) WAIT_TIMEOUT_SEC="${arg#*=}" ;;
    --interval-sec=*) WAIT_INTERVAL_SEC="${arg#*=}" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$COMMIT_SHA" ]]; then
  echo "ERROR: --commit=MERGE_SHA is required." >&2
  exit 2
fi

git fetch "$REMOTE_NAME" "$TARGET_BRANCH" -q
main_ref="${REMOTE_NAME}/${TARGET_BRANCH}"
main_sha="$(git rev-parse "${main_ref}^{commit}")"
if ! requested_sha="$(git rev-parse "${COMMIT_SHA}^{commit}" 2>/dev/null)"; then
  echo "ERROR: commit cannot be resolved: $COMMIT_SHA" >&2
  exit 2
fi

if [[ "$requested_sha" != "$main_sha" ]]; then
  echo "ERROR: requested commit is not current ${main_ref}." >&2
  echo "requested=$requested_sha" >&2
  echo "${main_ref}=$main_sha" >&2
  echo "Deploy blocked. Merge the PR first, then use the new ${main_ref} SHA." >&2
  exit 1
fi

expected_version="$(
  git show "${main_sha}:rag_server.py" \
    | python3 -c 'import re,sys; m=re.search(r"^BUILD_VERSION\s*=\s*\"([^\"]+)\"", sys.stdin.read(), re.M); print(m.group(1) if m else "")'
)"
if [[ -z "$expected_version" ]]; then
  echo "ERROR: BUILD_VERSION is missing in $main_sha." >&2
  exit 1
fi

echo "Release verified:"
echo "  source:  $main_ref"
echo "  commit:  $main_sha"
echo "  version: $expected_version"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN_OK"
  exit 0
fi

EXPECTED_VERSION="$expected_version" scripts/ops/render_apply_deploy.sh \
  --commit="$main_sha" \
  --prod-url="$PROD_URL" \
  --timeout-sec="$WAIT_TIMEOUT_SEC" \
  --interval-sec="$WAIT_INTERVAL_SEC"

prod_json="$(curl -fsS "${PROD_URL%/}/api/version")"
prod_commit="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("git_commit",""))' <<<"$prod_json")"
if [[ "$prod_commit" != "$main_sha" ]]; then
  echo "ERROR: production git_commit mismatch." >&2
  echo "expected=$main_sha actual=${prod_commit:-missing}" >&2
  exit 1
fi

curl -fsS "${PROD_URL%/}/health/live" >/dev/null
echo "RELEASE_OK commit=$main_sha version=$expected_version"
