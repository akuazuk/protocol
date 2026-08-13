#!/usr/bin/env bash
# Deploy protocol-gcp-app to GCE staging VM (temporary hostname / IP).
# Requires: gcloud auth, local .env with GOOGLE_API_KEY, Docker on VM.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
REMOTE_DIR="${REMOTE_DIR:-/opt/protocol}"
IMAGE_TAG="${IMAGE_TAG:-protocol-gcp-app:staging}"
CONTAINER="${CONTAINER:-protocol-web}"
ENV_REMOTE="${ENV_REMOTE:-/opt/protocol/.env.gcp-staging}"
ENV_WEB_PUBLIC="${ENV_WEB_PUBLIC:-/opt/protocol/.env.gcp-public}"
ENV_MIS_REMOTE="${ENV_MIS_REMOTE:-/opt/protocol/.env.mis}"
MIS_SM_SECRET="${MIS_SM_SECRET:-kravira-db-password}"
# Cron owner on protocol-app; deploy SSH login may differ (pavel vs pavelkuzauka).
GCE_OPS_USER="${GCE_OPS_USER:-pavel}"

usage() {
  cat <<'EOF'
Usage: deploy/gcp-app/deploy_to_gce.sh [--dry-run]

Syncs allowlisted sources to GCE, builds Docker image on the VM, runs container
on :8000 with /var/data mounted. Does not touch Render DNS.
EOF
}

DRY=0
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

gcloud config set project "$PROJECT" --quiet >/dev/null
STATUS="$(gcloud compute instances describe "$VM" --zone="$ZONE" --format='get(status)')"
if [[ "$STATUS" != "RUNNING" ]]; then
  echo "Starting VM $VM ..."
  gcloud compute instances start "$VM" --zone="$ZONE" --quiet
  sleep 15
fi

ssh_cmd() {
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="$1"
}

CORPUS_REMOTE="${CORPUS_REMOTE:-/var/data/protocol_corpus}"
# 1=sync corpus when local PDFs/summaries exist; 0=reuse whatever is already on the VM
SYNC_PROTOCOL_CORPUS="${SYNC_PROTOCOL_CORPUS:-1}"

echo "[1/5] prepare remote dir"
ssh_cmd "sudo mkdir -p '$REMOTE_DIR' /var/data/medical_exams '$CORPUS_REMOTE' && sudo chown -R \"\$(whoami):\$(whoami)\" '$REMOTE_DIR' '$CORPUS_REMOTE'"

echo "[2/5] build staging env + Secret Manager payloads (no values printed)"
python3 - <<'PY'
from pathlib import Path
import shutil

src = Path(".env")
public_dst = Path("/tmp/protocol-gcp-public.env")
mis_dst = Path("/tmp/protocol-gcp-mis.env")
sm_dir = Path("/tmp/protocol-web-sm")
if sm_dir.exists():
    shutil.rmtree(sm_dir)
sm_dir.mkdir(parents=True)

secret_keys = {
    "GOOGLE_API_KEY",
    "GOOGLE_API_KEY_2",
    "GEMINI_API_KEY",
    "GEMINI_API_KEY_2",
    "GENERATIVE_LANGUAGE_API_KEY",
    "METHODIST_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "RENDER_API_KEY",
}
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
want = secret_keys | {
    "GEMINI_MODEL",
    "GEMINI_METHODIST_MODEL",
    "GEMINI_GRADER_BULK_MODEL",
    "METHODIST_REVIEWER",
    "METHODIST_UI_AUTO_LOGIN",
    "ML_FEEDBACK_DIR",
    "RAG_CHUNKS_JSONL",
    "RENDER_URL",
    "TELEGRAM_NOTIFY_ENABLED",
    "TELEGRAM_NOTIFY_GIT",
    "TELEGRAM_NOTIFY_RENDER",
    "TELEGRAM_ALERTS",
    "TELEGRAM_INSECURE_SSL",
}
# E2: Marina / MIS from GCE (not Mac bridge). Password usually lives in sql_epam.
mis_want = {
    "KRAVIRA_DB_PASSWORD",
    "KRAVIRA_DB_HOST",
    "KRAVIRA_DB_PORT",
    "KRAVIRA_DB_USER",
    "KRAVIRA_DB_NAME",
    "MIS_DB_CONNECT_TIMEOUT",
    "MIS_DB_READ_TIMEOUT",
}
vals: dict[str, str] = {}
mis_vals: dict[str, str] = {}


def _ingest(path: Path, keys: set[str], into: dict) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        if k in keys and v.strip() and k not in into:
            into[k] = v.strip().strip('"').strip("'")


_ingest(src, want, vals)
_ingest(src, mis_want, mis_vals)
# Prefer sql_epam for MIS password (Mac secrets home); do not override if .env set.
_ingest(Path.home() / "CURSOR" / "sql_epam" / ".env", mis_want, mis_vals)
if not vals.get("GOOGLE_API_KEY") and not vals.get("GEMINI_API_KEY"):
    raise SystemExit("ERROR: need GOOGLE_API_KEY or GEMINI_API_KEY in .env")
# staging defaults (non-secret)
vals.setdefault("PORT", "8000")
vals.setdefault("MO_DATA_ROOT", "/var/data/medical_exams")
vals.setdefault("RAG_STARTUP_MODE", "manifest")
vals.setdefault("RAG_LAZY_CHUNK_STORE", "1")
vals.setdefault("RAG_GEMINI_EMBED_RERANK", "0")
vals.setdefault("RAG_MANIFEST_PATH", "data/catalog/corpus_path_manifest.jsonl")
vals.setdefault("RAG_FORBID_FULL_CORPUS_RETRIEVE", "1")
vals.setdefault("ALLOWED_ORIGINS", "*")
vals.setdefault("PYTHONUNBUFFERED", "1")
vals.setdefault("MO_ICD_NAME_IN_PRIMARY", "1")
vals.setdefault("MO_ICD_DIR_IN_PRIMARY", "0")
vals.setdefault("MO_ICD_PIPELINE_IN_PRIMARY", "0")
vals.setdefault("MO_ICD_LLM_REVIEW", "0")
vals.setdefault("MO_ICD_LLM_CLEAR_WEAK", "0")
# MIS DSN defaults (password required separately)
mis_vals.setdefault("KRAVIRA_DB_HOST", "178.163.240.131")
mis_vals.setdefault("KRAVIRA_DB_PORT", "6330")
mis_vals.setdefault("KRAVIRA_DB_USER", "kravira_mc_user")
mis_vals.setdefault("KRAVIRA_DB_NAME", "kravira_mc")
mis_vals.setdefault("MIS_DB_CONNECT_TIMEOUT", "30")
mis_vals.setdefault("MIS_DB_READ_TIMEOUT", "600")
mis_vals.setdefault("RUN_HOST", "gcp")
have_mis_pw = bool(mis_vals.get("KRAVIRA_DB_PASSWORD"))
import os as _os

# Optional: include MIS DSN (still without preferring password in web) for debug.
if have_mis_pw and _os.environ.get("INCLUDE_MIS_IN_WEB_ENV", "").strip().lower() in (
    "1",
    "true",
    "yes",
):
    for k, v in mis_vals.items():
        if k != "KRAVIRA_DB_PASSWORD":
            vals[k] = v

# Web secrets → SM files; public → .env.gcp-public
for env_key, secret_id in sm_map.items():
    val = vals.get(env_key)
    if not val:
        continue
    p = sm_dir / secret_id
    p.write_text(val, encoding="utf-8")
    p.chmod(0o600)

public_vals = {k: v for k, v in vals.items() if k not in secret_keys}
public_dst.write_text(
    "".join(f"{k}={v}\n" for k, v in sorted(public_vals.items())), encoding="utf-8"
)
public_dst.chmod(0o600)

public_mis = {k: v for k, v in mis_vals.items() if k != "KRAVIRA_DB_PASSWORD"}
if have_mis_pw:
    pw_path = Path("/tmp/protocol-sm-mis-pw")
    pw_path.write_text(mis_vals["KRAVIRA_DB_PASSWORD"], encoding="utf-8")
    pw_path.chmod(0o600)
    mis_dst.write_text(
        "".join(f"{k}={v}\n" for k, v in sorted(public_mis.items())), encoding="utf-8"
    )
    mis_dst.chmod(0o600)
    print(f"wrote {mis_dst} mis_public_keys={len(public_mis)} (password→Secret Manager)")
else:
    mis_dst.write_text("", encoding="utf-8")
    print(
        "WARN: no KRAVIRA_DB_PASSWORD in .env/sql_epam; skip .env.mis/SM upload "
        "(keep remote). Use deploy/gcp-app/push_mis_env.sh"
    )
print(
    f"wrote {public_dst} public_keys={len(public_vals)} "
    f"web_sm_files={len(list(sm_dir.iterdir()))}"
)
PY

SA="$(gcloud compute instances describe "$VM" --zone="$ZONE" --project="$PROJECT" \
  --format='get(serviceAccounts[0].email)')"

upsert_sm() {
  local secret_id="$1"
  local file="$2"
  if gcloud secrets describe "$secret_id" --project="$PROJECT" >/dev/null 2>&1; then
    gcloud secrets versions add "$secret_id" --data-file="$file" --project="$PROJECT" >/dev/null
    echo "[2b] SM version added secret=$secret_id"
  else
    gcloud secrets create "$secret_id" --data-file="$file" \
      --project="$PROJECT" --replication-policy=automatic >/dev/null
    gcloud secrets add-iam-policy-binding "$secret_id" \
      --project="$PROJECT" \
      --member="serviceAccount:${SA}" \
      --role="roles/secretmanager.secretAccessor" \
      --quiet >/dev/null
    echo "[2b] SM created secret=$secret_id"
  fi
}

if [[ -s /tmp/protocol-sm-mis-pw ]]; then
  upsert_sm "$MIS_SM_SECRET" /tmp/protocol-sm-mis-pw
  rm -f /tmp/protocol-sm-mis-pw
fi

shopt -s nullglob
for f in /tmp/protocol-web-sm/*; do
  upsert_sm "$(basename "$f")" "$f"
done
rm -rf /tmp/protocol-web-sm

if [[ "$DRY" == "1" ]]; then
  echo "dry-run: skip sync/build/run"
  exit 0
fi

echo "[3/5] sync sources to VM"
# shellcheck disable=SC2086
tar czf - \
  rag_server.py env_load.py icd_mkb.py retrieval_bm25.py gemini_verify.py consult_review_pipeline.py \
  download_minzdrav_protocols.py \
  requirements.txt requirements-rag.txt \
  backend frontend clinical_knowledge corpus_pipeline config scripts data/catalog \
  data/drug_safety/high_alert.json data/drug_safety/stopp_start_beers.json \
  data/icd_reference/icd10_ru_mkb10su.json \
  data/icd_reference/icd10_ru_mkb10su.meta.json \
  data/icd_reference/dx_aliases_ru.json \
  data/icd_reference/icd10_who_2016_terminal_codes.json \
  data/regulations \
  output/registry/protocol_cards.jsonl \
  services \
  deploy/gcp-app/*.sh deploy/gcp-app/Dockerfile .dockerignore \
  | gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="mkdir -p '$REMOTE_DIR' && tar xzf - -C '$REMOTE_DIR'"

gcloud compute scp /tmp/protocol-gcp-public.env "${VM}:~/protocol-gcp-public.env" --zone="$ZONE" --quiet
if [[ -s /tmp/protocol-gcp-mis.env ]]; then
  gcloud compute scp /tmp/protocol-gcp-mis.env "${VM}:~/protocol-gcp-mis.env" --zone="$ZONE" --quiet
  ssh_cmd "sudo mv ~/protocol-gcp-mis.env '$ENV_MIS_REMOTE' && OPS='${GCE_OPS_USER}'; getent passwd \"\$OPS\" >/dev/null || OPS=\$(whoami); sudo chown \"\$OPS:\$OPS\" '$ENV_MIS_REMOTE' && sudo chmod 600 '$ENV_MIS_REMOTE' && if grep -qE '^KRAVIRA_DB_PASSWORD=' '$ENV_MIS_REMOTE'; then echo 'ERROR: password must not be in .env.mis' >&2; exit 2; fi"
fi
ssh_cmd "sudo mv ~/protocol-gcp-public.env '$ENV_WEB_PUBLIC' && OPS='${GCE_OPS_USER}'; getent passwd \"\$OPS\" >/dev/null || OPS=\$(whoami); sudo chown \"\$OPS:\$OPS\" '$ENV_WEB_PUBLIC' && sudo chmod 600 '$ENV_WEB_PUBLIC' && bash '$REMOTE_DIR'/deploy/gcp-app/assemble_web_env_from_sm.sh"
rm -f /tmp/protocol-gcp-public.env /tmp/protocol-gcp-mis.env

if [[ "$SYNC_PROTOCOL_CORPUS" == "1" ]] \
  && [[ -d minzdrav_protocols ]] \
  && [[ -d data/protocol_summaries/json ]] \
  && [[ -f data/protocol_catalog.jsonl ]]; then
  echo "[3b/5] sync protocol corpus (PDF + summaries) → ${CORPUS_REMOTE}"
  bash deploy/gcp-app/sync_protocol_corpus.sh
else
  echo "[3b/5] skip protocol corpus sync (SYNC_PROTOCOL_CORPUS=${SYNC_PROTOCOL_CORPUS})"
fi

echo "[4/5] docker build on VM (may take several minutes)"
ssh_cmd "cd '$REMOTE_DIR' && sudo docker build -f deploy/gcp-app/Dockerfile -t '$IMAGE_TAG' ."

HAS_CORPUS="$(ssh_cmd "if [[ -d '$CORPUS_REMOTE/minzdrav_protocols' && -d '$CORPUS_REMOTE/protocol_summaries/json' && -f '$CORPUS_REMOTE/protocol_catalog.jsonl' ]]; then echo yes; else echo no; fi")"
CORPUS_MOUNTS=""
if [[ "$HAS_CORPUS" == "yes" ]]; then
  echo "[5/5] run container with protocol corpus mounts"
  ssh_cmd "sudo mkdir -p '$CORPUS_REMOTE'/output/registry '$CORPUS_REMOTE'/_sync; if [[ ! -f '$CORPUS_REMOTE'/protocol_icd_profiles.jsonl && -f '$REMOTE_DIR'/data/catalog/protocol_icd_profiles.jsonl ]]; then sudo cp '$REMOTE_DIR'/data/catalog/protocol_icd_profiles.jsonl '$CORPUS_REMOTE'/protocol_icd_profiles.jsonl; fi; if [[ ! -f '$CORPUS_REMOTE'/output/registry/protocol_cards.jsonl && -f '$REMOTE_DIR'/output/registry/protocol_cards.jsonl ]]; then sudo cp '$REMOTE_DIR'/output/registry/protocol_cards.jsonl '$CORPUS_REMOTE'/output/registry/protocol_cards.jsonl; fi"
  CORPUS_MOUNTS="-v $CORPUS_REMOTE/minzdrav_protocols:/app/minzdrav_protocols:ro -v $CORPUS_REMOTE/protocol_summaries:/app/data/protocol_summaries:ro -v $CORPUS_REMOTE/protocol_catalog.jsonl:/app/data/protocol_catalog.jsonl:ro"
  CORPUS_MOUNTS="$CORPUS_MOUNTS -v $CORPUS_REMOTE/protocol_icd_profiles.jsonl:/app/data/catalog/protocol_icd_profiles.jsonl:ro"
  CORPUS_MOUNTS="$CORPUS_MOUNTS -v $CORPUS_REMOTE/output/registry/protocol_cards.jsonl:/app/output/registry/protocol_cards.jsonl:ro"
else
  echo "[5/5] run container WITHOUT protocol corpus (navigator will be empty; sync with deploy/gcp-app/sync_protocol_corpus.sh)" >&2
fi

# DDInter pairs are gitignored; keep them on the data disk and mount into the image.
ssh_cmd "sudo mkdir -p /var/data/drug_safety && sudo cp -n '$REMOTE_DIR'/data/drug_safety/high_alert.json /var/data/drug_safety/ 2>/dev/null || true; sudo cp -n '$REMOTE_DIR'/data/drug_safety/stopp_start_beers.json /var/data/drug_safety/ 2>/dev/null || true; if [[ -f '$REMOTE_DIR'/data/drug_safety/ddinter_pairs.json ]]; then sudo cp '$REMOTE_DIR'/data/drug_safety/ddinter_pairs.json /var/data/drug_safety/; fi; ls -la /var/data/drug_safety || true"
DRUG_SAFETY_MOUNT="-v /var/data/drug_safety:/app/data/drug_safety:ro"

ssh_cmd "sudo docker rm -f '$CONTAINER' >/dev/null 2>&1 || true
sudo docker run -d --name '$CONTAINER' --restart unless-stopped \
  -p 8000:8000 \
  --env-file '$ENV_REMOTE' \
  -v /var/data:/var/data \
  $CORPUS_MOUNTS \
  $DRUG_SAFETY_MOUNT \
  -e MO_DATA_ROOT=/var/data/medical_exams \
  -e PROTOCOL_CORPUS_ROOT=/var/data/protocol_corpus \
  -e PROTOCOL_ICD_PROFILE_INDEX=/app/data/catalog/protocol_icd_profiles.jsonl \
  -e PROTOCOL_CARDS_PATH=/app/output/registry/protocol_cards.jsonl \
  -e PORT=8000 \
  '$IMAGE_TAG'
sleep 3
sudo docker ps --filter name='$CONTAINER' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"

IP="$(gcloud compute addresses describe protocol-app-ip --region=europe-central2 --format='get(address)')"
echo "Waiting for /health/live on ${IP}:8000 ..."
for i in $(seq 1 36); do
  if curl -fsS --max-time 5 "http://${IP}:8000/health/live" >/dev/null 2>&1; then
    echo "HEALTH_OK"
    curl -fsS "http://${IP}:8000/api/version"
    echo
    exit 0
  fi
  sleep 5
done
echo "HEALTH timeout; recent logs:" >&2
ssh_cmd "sudo docker logs --tail 80 '$CONTAINER'" || true
exit 1
