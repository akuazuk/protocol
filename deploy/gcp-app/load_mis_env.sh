#!/usr/bin/env bash
# Load Marina/MIS DSN for GCE night/smoke.
# Password: Secret Manager (default). Non-secret host/port/user: .env.mis or defaults.
#
# Source only (do not exec):
#   source /opt/protocol/deploy/gcp-app/load_mis_env.sh
#
# Env overrides:
#   ENV_MIS / ENV_MIS_REMOTE, MIS_SM_SECRET, GCP_PROJECT,
#   MIS_PASSWORD_SOURCE=secretmanager|envfile
#
# Safe to source from cron: does not change caller's set -e/-u options permanently.

ENV_MIS="${ENV_MIS:-${ENV_MIS_REMOTE:-/opt/protocol/.env.mis}}"
MIS_SM_SECRET="${MIS_SM_SECRET:-kravira-db-password}"
GCP_PROJECT="${GCP_PROJECT:-protocol-home-e1}"
MIS_PASSWORD_SOURCE="${MIS_PASSWORD_SOURCE:-secretmanager}"

if [[ -f "$ENV_MIS" ]]; then
  if [[ ! -r "$ENV_MIS" ]]; then
    echo "ERROR: cannot read $ENV_MIS as $(whoami) (owner/mode mismatch for cron user)" >&2
    return 2 2>/dev/null || exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  # shellcheck disable=SC1091
  source "$ENV_MIS"
  set +a
fi

export KRAVIRA_DB_HOST="${KRAVIRA_DB_HOST:-178.163.240.131}"
export KRAVIRA_DB_PORT="${KRAVIRA_DB_PORT:-6330}"
export KRAVIRA_DB_USER="${KRAVIRA_DB_USER:-kravira_mc_user}"
export KRAVIRA_DB_NAME="${KRAVIRA_DB_NAME:-kravira_mc}"
export MIS_DB_CONNECT_TIMEOUT="${MIS_DB_CONNECT_TIMEOUT:-30}"
export MIS_DB_READ_TIMEOUT="${MIS_DB_READ_TIMEOUT:-600}"
export RUN_HOST="${RUN_HOST:-gcp}"

# SM is source of truth for password unless emergency envfile mode.
unset KRAVIRA_DB_PASSWORD || true

case "$MIS_PASSWORD_SOURCE" in
  secretmanager|sm|gsm)
    if ! command -v gcloud >/dev/null 2>&1; then
      echo "ERROR: gcloud not in PATH; cannot read Secret Manager ($MIS_SM_SECRET)" >&2
      return 2 2>/dev/null || exit 2
    fi
    _pw="$(gcloud secrets versions access latest \
      --secret="$MIS_SM_SECRET" \
      --project="$GCP_PROJECT" 2>/dev/null || true)"
    if [[ -z "${_pw}" ]]; then
      echo "ERROR: empty/failed Secret Manager access for $MIS_SM_SECRET (project=$GCP_PROJECT)" >&2
      return 2 2>/dev/null || exit 2
    fi
    export KRAVIRA_DB_PASSWORD="$_pw"
    unset _pw
    echo "MIS_DSN_OK source=secretmanager secret=$MIS_SM_SECRET host=$KRAVIRA_DB_HOST"
    ;;
  envfile|file)
    if [[ -f "$ENV_MIS" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "$ENV_MIS"
      set +a
    fi
    if [[ -z "${KRAVIRA_DB_PASSWORD:-}" ]]; then
      echo "ERROR: MIS_PASSWORD_SOURCE=envfile but KRAVIRA_DB_PASSWORD missing in $ENV_MIS" >&2
      return 2 2>/dev/null || exit 2
    fi
    echo "MIS_DSN_OK source=envfile host=$KRAVIRA_DB_HOST"
    ;;
  *)
    echo "ERROR: unknown MIS_PASSWORD_SOURCE=$MIS_PASSWORD_SOURCE" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
