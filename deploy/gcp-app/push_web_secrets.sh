#!/usr/bin/env bash
# Push web secrets (Gemini / methodist / telegram / render) to Secret Manager.
# Writes allowlisted non-secret /opt/protocol/.env.gcp-public and assembles staging.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
GCE_OPS_USER="${GCE_OPS_USER:-pavel}"
RESTART_WEB="${RESTART_WEB:-0}"

usage() {
  cat <<'EOF'
Usage: deploy/gcp-app/push_web_secrets.sh [--restart-web]

Uploads API/auth secrets to Secret Manager and refreshes protocol-web env on the VM.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --restart-web) RESTART_WEB=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

gcloud config set project "$PROJECT" --quiet >/dev/null

python3 - <<'PY'
from pathlib import Path
import shutil

sm_map = {
    "GOOGLE_API_KEY": "google-api-key",
    "GOOGLE_API_KEY_2": "google-api-key-2",
    "GEMINI_API_KEY": "gemini-api-key",
    "GEMINI_API_KEY_2": "gemini-api-key-2",
    "GENERATIVE_LANGUAGE_API_KEY": "generative-language-api-key",
    "METHODIST_TOKEN": "methodist-token",
    "TELEGRAM_BOT_TOKEN": "telegram-bot-token",
    "TELEGRAM_CHAT_ID": "telegram-chat-id",
    "RENDER_API_KEY": "render-api-key",
}
# Strict allowlist — never dump entire .env into public file.
public_allow = {
    "PORT",
    "MO_DATA_ROOT",
    "RAG_STARTUP_MODE",
    "RAG_LAZY_CHUNK_STORE",
    "RAG_GEMINI_EMBED_RERANK",
    "RAG_MANIFEST_PATH",
    "RAG_FORBID_FULL_CORPUS_RETRIEVE",
    "RAG_CHUNKS_JSONL",
    "ALLOWED_ORIGINS",
    "PYTHONUNBUFFERED",
    "MO_ICD_NAME_IN_PRIMARY",
    "MO_ICD_DIR_IN_PRIMARY",
    "MO_ICD_PIPELINE_IN_PRIMARY",
    "MO_ICD_LLM_REVIEW",
    "MO_ICD_LLM_CLEAR_WEAK",
    "GEMINI_MODEL",
    "GEMINI_METHODIST_MODEL",
    "GEMINI_GRADER_BULK_MODEL",
    "METHODIST_REVIEWER",
    "METHODIST_UI_AUTO_LOGIN",
    "ML_FEEDBACK_DIR",
    "RENDER_URL",
    "TELEGRAM_NOTIFY_ENABLED",
    "TELEGRAM_NOTIFY_GIT",
    "TELEGRAM_NOTIFY_RENDER",
    "TELEGRAM_ALERTS",
    "TELEGRAM_INSECURE_SSL",
}

vals: dict[str, str] = {}


def _ingest(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if v and k not in vals:
            vals[k] = v


_ingest(Path(".env"))
if not vals.get("GOOGLE_API_KEY") and not vals.get("GEMINI_API_KEY"):
    raise SystemExit("ERROR: need GOOGLE_API_KEY or GEMINI_API_KEY in .env")

sm_dir = Path("/tmp/protocol-web-sm")
if sm_dir.exists():
    shutil.rmtree(sm_dir)
sm_dir.mkdir(parents=True)
uploaded = 0
for env_key, secret_id in sm_map.items():
    val = vals.get(env_key)
    if not val:
        continue
    path = sm_dir / secret_id
    path.write_text(val, encoding="utf-8")
    path.chmod(0o600)
    uploaded += 1

public = {k: vals[k] for k in public_allow if k in vals}
public.setdefault("PORT", "8000")
public.setdefault("MO_DATA_ROOT", "/var/data/medical_exams")
public.setdefault("RAG_STARTUP_MODE", "manifest")
public.setdefault("RAG_LAZY_CHUNK_STORE", "1")
public.setdefault("RAG_GEMINI_EMBED_RERANK", "0")
public.setdefault("RAG_MANIFEST_PATH", "data/catalog/corpus_path_manifest.jsonl")
public.setdefault("RAG_FORBID_FULL_CORPUS_RETRIEVE", "1")
public.setdefault("ALLOWED_ORIGINS", "*")
public.setdefault("PYTHONUNBUFFERED", "1")
public.setdefault("MO_ICD_NAME_IN_PRIMARY", "1")
public.setdefault("MO_ICD_DIR_IN_PRIMARY", "0")
public.setdefault("MO_ICD_PIPELINE_IN_PRIMARY", "0")
public.setdefault("MO_ICD_LLM_REVIEW", "0")
public.setdefault("MO_ICD_LLM_CLEAR_WEAK", "0")

out = Path("/tmp/protocol-gcp-public.env")
out.write_text("".join(f"{k}={v}\n" for k, v in sorted(public.items())), encoding="utf-8")
out.chmod(0o600)
print(f"local_web_public keys={len(public)} sm_files={uploaded}")
PY

SA="$(gcloud compute instances describe "$VM" --zone="$ZONE" --project="$PROJECT" \
  --format='get(serviceAccounts[0].email)')"

upsert_secret() {
  local secret_id="$1"
  local file="$2"
  if gcloud secrets describe "$secret_id" --project="$PROJECT" >/dev/null 2>&1; then
    gcloud secrets versions add "$secret_id" --data-file="$file" --project="$PROJECT" >/dev/null
    echo "SM_VERSION_ADDED secret=$secret_id"
  else
    gcloud secrets create "$secret_id" --data-file="$file" \
      --project="$PROJECT" --replication-policy=automatic >/dev/null
    gcloud secrets add-iam-policy-binding "$secret_id" \
      --project="$PROJECT" \
      --member="serviceAccount:${SA}" \
      --role="roles/secretmanager.secretAccessor" \
      --quiet >/dev/null
    echo "SM_CREATED secret=$secret_id"
  fi
}

shopt -s nullglob
for f in /tmp/protocol-web-sm/*; do
  upsert_secret "$(basename "$f")" "$f"
done
rm -rf /tmp/protocol-web-sm

gcloud compute scp /tmp/protocol-gcp-public.env "${VM}:~/protocol-gcp-public.env" --zone="$ZONE" --quiet
rm -f /tmp/protocol-gcp-public.env

gcloud compute scp "$ROOT/deploy/gcp-app/assemble_web_env_from_sm.sh" \
  "${VM}:~/assemble_web_env_from_sm.sh" --zone="$ZONE" --quiet

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo mkdir -p /opt/protocol/deploy/gcp-app
sudo mv ~/protocol-gcp-public.env /opt/protocol/.env.gcp-public
sudo mv ~/assemble_web_env_from_sm.sh /opt/protocol/deploy/gcp-app/assemble_web_env_from_sm.sh
sudo chmod +x /opt/protocol/deploy/gcp-app/assemble_web_env_from_sm.sh
OPS='${GCE_OPS_USER}'
getent passwd \"\$OPS\" >/dev/null || OPS=\$(whoami)
sudo chown \"\$OPS:\$OPS\" /opt/protocol/.env.gcp-public
sudo chmod 600 /opt/protocol/.env.gcp-public
# refuse secrets in public
if grep -qE '^(GOOGLE_API_KEY|GEMINI_API_KEY|METHODIST_TOKEN|KRAVIRA_DB_PASSWORD|RENDER_API_KEY|TELEGRAM_BOT_TOKEN)=' /opt/protocol/.env.gcp-public; then
  echo 'ERROR: secrets must not be in .env.gcp-public' >&2
  exit 2
fi
bash /opt/protocol/deploy/gcp-app/assemble_web_env_from_sm.sh
awk -F= '{print \$1}' /opt/protocol/.env.gcp-staging | sed 's/^/WEB_KEY /'
"

if [[ "$RESTART_WEB" == "1" ]]; then
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo docker rm -f protocol-web >/dev/null 2>&1 || true
IMG=\$(sudo docker images protocol-gcp-app:staging --format '{{.Repository}}:{{.Tag}}' | head -1)
if [[ -z \"\$IMG\" ]]; then
  echo 'ERROR: no protocol-gcp-app:staging image; run deploy_to_gce.sh' >&2
  exit 2
fi
CORPUS_MOUNTS=''
if [[ -d /var/data/protocol_corpus/minzdrav_protocols && -d /var/data/protocol_corpus/protocol_summaries/json && -f /var/data/protocol_corpus/protocol_catalog.jsonl ]]; then
  CORPUS_MOUNTS='-v /var/data/protocol_corpus/minzdrav_protocols:/app/minzdrav_protocols:ro -v /var/data/protocol_corpus/protocol_summaries:/app/data/protocol_summaries:ro -v /var/data/protocol_corpus/protocol_catalog.jsonl:/app/data/protocol_catalog.jsonl:ro'
fi
# shellcheck disable=SC2086
sudo docker run -d --name protocol-web --restart unless-stopped \
  -p 8000:8000 \
  --env-file /opt/protocol/.env.gcp-staging \
  -v /var/data:/var/data \
  -v /var/data/drug_safety:/app/data/drug_safety:ro \
  \$CORPUS_MOUNTS \
  -e MO_DATA_ROOT=/var/data/medical_exams \
  -e PORT=8000 \
  \"\$IMG\"
sleep 3
sudo docker ps --filter name=protocol-web --format 'table {{.Names}}\t{{.Status}}'
"
fi

echo "Done. Optional: --restart-web or full deploy_to_gce.sh"
