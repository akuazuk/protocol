#!/usr/bin/env bash
# Install rceth watchdog (+ optional weekly full) into GCE crontab of ops user.
# Does NOT replace night MIS/KP cron - merges markers.
#
# From Mac:
#   bash deploy/gcp-app/install_rceth_cron.sh --remote --enable-watchdog
# Weekly full (Sun 04:00 UTC) + watchdog:
#   bash deploy/gcp-app/install_rceth_cron.sh --remote --enable-watchdog --enable-weekly
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE=0
ENABLE_WATCHDOG=0
ENABLE_WEEKLY=0
for arg in "$@"; do
  case "$arg" in
    --remote) REMOTE=1 ;;
    --enable-watchdog) ENABLE_WATCHDOG=1 ;;
    --enable-weekly) ENABLE_WEEKLY=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: deploy/gcp-app/install_rceth_cron.sh [--remote] [--enable-watchdog] [--enable-weekly]
EOF
      exit 0
      ;;
    *) echo "Unknown: $arg" >&2; exit 2 ;;
  esac
done

PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"

install_local() {
  local proto_root="${PROTOCOL_ROOT:-/opt/protocol}"
  local data="${RCETH_DATA_ROOT:-/var/data/rceth}"
  local mo="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}"
  sudo mkdir -p "${data}/_sync" "${data}/pdfs/instr" "${data}/labels" "${mo}/logs"
  sudo chown -R "$(whoami):$(whoami)" "${data}" || true
  chmod +x "${proto_root}/deploy/gcp-app/rceth_sync_job.sh" \
    "${proto_root}/deploy/gcp-app/rceth_sync_watchdog.sh" 2>/dev/null || true

  local tmp
  tmp="$(mktemp)"
  crontab -l 2>/dev/null | grep -v 'RCETH_SYNC_MARKER' | grep -v 'rceth_sync_watchdog' | grep -v 'rceth_sync_job.sh' >"$tmp" || true
  {
    echo "# RCETH_SYNC_MARKER begin"
    if [[ "$ENABLE_WATCHDOG" == "1" ]]; then
      echo "# every 10 min: resume if job died (max 6 restarts/day)"
      echo "*/10 * * * * RCETH_DATA_ROOT=${data} GCE_MO_DATA_ROOT=${mo} PROTOCOL_ROOT=${proto_root} ${proto_root}/deploy/gcp-app/rceth_sync_watchdog.sh"
    else
      echo "# watchdog disabled (pass --enable-watchdog)"
    fi
    if [[ "$ENABLE_WEEKLY" == "1" ]]; then
      echo "# Sun 04:00 UTC: full refresh (LIMIT=0); download resume-safe"
      echo "0 4 * * 0 RCETH_MODE=weekly RCETH_LIMIT=0 RCETH_PARSE=1 RCETH_DATA_ROOT=${data} GCE_MO_DATA_ROOT=${mo} PROTOCOL_ROOT=${proto_root} RCETH_INSECURE_SSL=1 ${proto_root}/deploy/gcp-app/rceth_sync_job.sh"
    else
      echo "# weekly full disabled (pass --enable-weekly)"
    fi
    echo "# RCETH_SYNC_MARKER end"
  } >>"$tmp"
  crontab "$tmp"
  rm -f "$tmp"
  echo "RCETH_CRON_INSTALLED watchdog=${ENABLE_WATCHDOG} weekly=${ENABLE_WEEKLY}"
  crontab -l | grep -A20 'RCETH_SYNC_MARKER' || crontab -l
}

if [[ "$REMOTE" == "1" ]]; then
  gcloud config set project "$PROJECT" --quiet >/dev/null
  tar czf - \
    -C "$ROOT" \
    deploy/gcp-app/rceth_sync_job.sh \
    deploy/gcp-app/rceth_sync_watchdog.sh \
    deploy/gcp-app/install_rceth_cron.sh \
    deploy/gcp-app/run_rceth_full_on_gce.sh \
    deploy/gcp-app/watch_rceth_sync.sh \
    | gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
mkdir -p ~/protocol-rceth-sync
tar xzf - -C ~/protocol-rceth-sync
sudo mkdir -p /opt/protocol/deploy/gcp-app
sudo cp -f ~/protocol-rceth-sync/deploy/gcp-app/*.sh /opt/protocol/deploy/gcp-app/
sudo chmod +x /opt/protocol/deploy/gcp-app/*.sh
OPS_USER=\"\${GCE_OPS_USER:-pavel}\"
if ! getent passwd \"\$OPS_USER\" >/dev/null 2>&1; then
  OPS_USER=\"\$(whoami)\"
fi
sudo chown -R \"\$OPS_USER:\$OPS_USER\" /opt/protocol/deploy/gcp-app
ARGS=()
if [[ '${ENABLE_WATCHDOG}' == '1' ]]; then ARGS+=(--enable-watchdog); fi
if [[ '${ENABLE_WEEKLY}' == '1' ]]; then ARGS+=(--enable-weekly); fi
bash /opt/protocol/deploy/gcp-app/install_rceth_cron.sh \"\${ARGS[@]}\"
"
else
  install_local
fi
