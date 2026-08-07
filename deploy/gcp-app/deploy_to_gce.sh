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

echo "[1/5] prepare remote dir"
ssh_cmd "sudo mkdir -p '$REMOTE_DIR' /var/data/medical_exams && sudo chown -R \"\$(whoami):\$(whoami)\" '$REMOTE_DIR'"

echo "[2/5] build staging env (no values printed)"
python3 - <<'PY'
from pathlib import Path
src = Path(".env")
dst = Path("/tmp/protocol-gcp-staging.env")
want = {
    "GOOGLE_API_KEY",
    "GOOGLE_API_KEY_2",
    "GEMINI_API_KEY",
    "GEMINI_API_KEY_2",
    "GENERATIVE_LANGUAGE_API_KEY",
    "METHODIST_TOKEN",
    "GEMINI_MODEL",
    "GEMINI_METHODIST_MODEL",
    "GEMINI_GRADER_BULK_MODEL",
}
vals = {}
for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s:
        continue
    k, v = s.split("=", 1)
    k = k.strip()
    if k in want and v.strip():
        vals[k] = v.strip().strip('"').strip("'")
if not vals.get("GOOGLE_API_KEY") and not vals.get("GEMINI_API_KEY"):
    raise SystemExit("ERROR: need GOOGLE_API_KEY or GEMINI_API_KEY in .env")
# staging defaults
vals.setdefault("PORT", "8000")
vals.setdefault("MO_DATA_ROOT", "/var/data/medical_exams")
vals.setdefault("RAG_STARTUP_MODE", "manifest")
vals.setdefault("RAG_LAZY_CHUNK_STORE", "1")
vals.setdefault("RAG_GEMINI_EMBED_RERANK", "0")
vals.setdefault("RAG_MANIFEST_PATH", "data/catalog/corpus_path_manifest.jsonl")
vals.setdefault("RAG_FORBID_FULL_CORPUS_RETRIEVE", "1")
vals.setdefault("ALLOWED_ORIGINS", "*")
vals.setdefault("PYTHONUNBUFFERED", "1")
dst.write_text("".join(f"{k}={v}\n" for k, v in sorted(vals.items())), encoding="utf-8")
dst.chmod(0o600)
print(f"wrote {dst} keys={len(vals)}")
PY

if [[ "$DRY" == "1" ]]; then
  echo "dry-run: skip sync/build/run"
  exit 0
fi

echo "[3/5] sync sources to VM"
# shellcheck disable=SC2086
tar czf - \
  rag_server.py env_load.py icd_mkb.py retrieval_bm25.py gemini_verify.py consult_review_pipeline.py \
  requirements.txt requirements-rag.txt \
  backend frontend clinical_knowledge corpus_pipeline config scripts data/catalog services \
  deploy/gcp-app/Dockerfile .dockerignore \
  | gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="mkdir -p '$REMOTE_DIR' && tar xzf - -C '$REMOTE_DIR'"

gcloud compute scp /tmp/protocol-gcp-staging.env "${VM}:${ENV_REMOTE}" --zone="$ZONE" --quiet
rm -f /tmp/protocol-gcp-staging.env
ssh_cmd "chmod 600 '$ENV_REMOTE'"

echo "[4/5] docker build on VM (may take several minutes)"
ssh_cmd "cd '$REMOTE_DIR' && sudo docker build -f deploy/gcp-app/Dockerfile -t '$IMAGE_TAG' ."

echo "[5/5] run container"
ssh_cmd "sudo docker rm -f '$CONTAINER' >/dev/null 2>&1 || true
sudo docker run -d --name '$CONTAINER' --restart unless-stopped \
  -p 8000:8000 \
  --env-file '$ENV_REMOTE' \
  -v /var/data:/var/data \
  -e MO_DATA_ROOT=/var/data/medical_exams \
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
