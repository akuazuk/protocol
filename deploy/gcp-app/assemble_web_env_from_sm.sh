#!/usr/bin/env bash
# Assemble /opt/protocol/.env.gcp-staging for protocol-web from:
#   - non-secret /opt/protocol/.env.gcp-public (or legacy .env.gcp-staging public keys)
#   - Secret Manager (google-api-key, methodist-token, …)
#
# On VM (cron-safe PATH has /usr/bin/gcloud):
#   bash /opt/protocol/deploy/gcp-app/assemble_web_env_from_sm.sh
set -euo pipefail

PROJECT="${GCP_PROJECT:-protocol-home-e1}"
OPS_USER="${GCE_OPS_USER:-pavel}"
PUBLIC_ENV="${ENV_WEB_PUBLIC:-/opt/protocol/.env.gcp-public}"
OUT_ENV="${ENV_WEB_REMOTE:-/opt/protocol/.env.gcp-staging}"
LEGACY_ENV="/opt/protocol/.env.gcp-staging"

SECRET_KEYS=(
  GOOGLE_API_KEY
  GOOGLE_API_KEY_2
  GEMINI_API_KEY
  GEMINI_API_KEY_2
  GENERATIVE_LANGUAGE_API_KEY
  METHODIST_TOKEN
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
  RENDER_API_KEY
  KRAVIRA_DB_PASSWORD
)

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud not in PATH" >&2
  exit 2
fi

tmp="$(mktemp)"
chmod 600 "$tmp"
trap 'rm -f "$tmp"' EXIT

# 1) Public / non-secret keys
src_public=""
if [[ -f "$PUBLIC_ENV" ]]; then
  src_public="$PUBLIC_ENV"
elif [[ -f "$LEGACY_ENV" ]]; then
  src_public="$LEGACY_ENV"
fi

if [[ -n "$src_public" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    s="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$s" || "$s" == \#* || "$s" != *=* ]] && continue
    key="${s%%=*}"
    skip=0
    for sk in "${SECRET_KEYS[@]}"; do
      if [[ "$key" == "$sk" ]]; then
        skip=1
        break
      fi
    done
    [[ "$skip" == "1" ]] && continue
    printf '%s\n' "$s" >>"$tmp"
  done <"$src_public"
fi

# 2) Secrets from SM (skip missing optional secrets)
fetch_sm() {
  local env_key="$1"
  local secret_id="$2"
  local val=""
  if ! gcloud secrets describe "$secret_id" --project="$PROJECT" >/dev/null 2>&1; then
    return 0
  fi
  val="$(gcloud secrets versions access latest --secret="$secret_id" --project="$PROJECT" 2>/dev/null || true)"
  if [[ -z "$val" ]]; then
    echo "WARN: empty secret $secret_id" >&2
    return 0
  fi
  # Avoid newline pollution in env values
  val="${val%$'\n'}"
  printf '%s=%s\n' "$env_key" "$val" >>"$tmp"
  echo "WEB_SM_OK key=$env_key secret=$secret_id"
}

fetch_sm GOOGLE_API_KEY google-api-key
fetch_sm GOOGLE_API_KEY_2 google-api-key-2
fetch_sm GEMINI_API_KEY gemini-api-key
fetch_sm GEMINI_API_KEY_2 gemini-api-key-2
fetch_sm GENERATIVE_LANGUAGE_API_KEY generative-language-api-key
fetch_sm METHODIST_TOKEN methodist-token
fetch_sm TELEGRAM_BOT_TOKEN telegram-bot-token
fetch_sm TELEGRAM_CHAT_ID telegram-chat-id
fetch_sm RENDER_API_KEY render-api-key

# Require at least one LLM key
if ! grep -qE '^(GOOGLE_API_KEY|GEMINI_API_KEY)=' "$tmp"; then
  echo "ERROR: need google-api-key or gemini-api-key in Secret Manager" >&2
  exit 2
fi

sudo mkdir -p "$(dirname "$OUT_ENV")"
sudo cp "$tmp" "$OUT_ENV"
if getent passwd "$OPS_USER" >/dev/null 2>&1; then
  sudo chown "$OPS_USER:$OPS_USER" "$OUT_ENV"
else
  sudo chown "$(whoami):$(whoami)" "$OUT_ENV"
fi
sudo chmod 600 "$OUT_ENV"
# Keep public + assembled env owned by cron user even if SSH login differs.
for f in "$PUBLIC_ENV" "$OUT_ENV" /opt/protocol/.env.mis; do
  if [[ -f "$f" ]] && getent passwd "$OPS_USER" >/dev/null 2>&1; then
    sudo chown "$OPS_USER:$OPS_USER" "$f" 2>/dev/null || true
    sudo chmod 600 "$f" 2>/dev/null || true
  fi
done
echo "WEB_ENV_ASSEMBLED path=$OUT_ENV keys=$(grep -cE '^[A-Z]' "$OUT_ENV" || true)"
