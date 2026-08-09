#!/usr/bin/env bash
# Sync KP PDFs + Protocol Summaries to GCE /var/data/protocol_corpus for navigator.
# Safe to re-run. Does not rebuild Docker image.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
CORPUS_REMOTE="${CORPUS_REMOTE:-/var/data/protocol_corpus}"

need=(
  minzdrav_protocols
  data/protocol_catalog.jsonl
  data/protocol_summaries/json
  data/protocol_summaries/source_text
)
for p in "${need[@]}"; do
  if [[ ! -e "$p" ]]; then
    echo "ERROR: missing local $p" >&2
    exit 1
  fi
done

gcloud config set project "$PROJECT" --quiet >/dev/null
gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="sudo mkdir -p '$CORPUS_REMOTE' && sudo chown -R \"\$(whoami):\$(whoami)\" '$CORPUS_REMOTE'"

echo "Packing protocol corpus from $ROOT ..."
export COPYFILE_DISABLE=1
# Prefer GNU/BSD tar without Apple xattrs when possible.
TAR=(tar)
if tar --help 2>&1 | grep -q -- '--disable-copyfile'; then
  TAR=(tar --disable-copyfile)
fi

"${TAR[@]}" -czf - \
  minzdrav_protocols \
  data/protocol_catalog.jsonl \
  data/protocol_summaries/json \
  data/protocol_summaries/yaml \
  data/protocol_summaries/source_text \
  data/protocol_summaries/drafts \
| gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -e
TMP=\$(mktemp -d)
# GNU tar may warn on Apple xattrs; do not fail the sync for that.
tar xzf - -C \"\$TMP\" || true
rm -rf '$CORPUS_REMOTE/minzdrav_protocols' '$CORPUS_REMOTE/protocol_summaries'
mkdir -p '$CORPUS_REMOTE/protocol_summaries'
mv \"\$TMP/minzdrav_protocols\" '$CORPUS_REMOTE/'
mv \"\$TMP/data/protocol_catalog.jsonl\" '$CORPUS_REMOTE/protocol_catalog.jsonl'
mv \"\$TMP/data/protocol_summaries/json\" '$CORPUS_REMOTE/protocol_summaries/'
test -d \"\$TMP/data/protocol_summaries/yaml\" && mv \"\$TMP/data/protocol_summaries/yaml\" '$CORPUS_REMOTE/protocol_summaries/' || true
mv \"\$TMP/data/protocol_summaries/source_text\" '$CORPUS_REMOTE/protocol_summaries/'
test -d \"\$TMP/data/protocol_summaries/drafts\" && mv \"\$TMP/data/protocol_summaries/drafts\" '$CORPUS_REMOTE/protocol_summaries/' || true
rm -rf \"\$TMP\"
du -sh '$CORPUS_REMOTE' '$CORPUS_REMOTE/minzdrav_protocols' '$CORPUS_REMOTE/protocol_summaries'
find '$CORPUS_REMOTE/minzdrav_protocols' -name '*.pdf' | wc -l
find '$CORPUS_REMOTE/protocol_summaries/json' -type f | wc -l
"
echo "OK: corpus at ${VM}:${CORPUS_REMOTE}"
echo "Restart container with mounts via: bash deploy/gcp-app/deploy_to_gce.sh"
