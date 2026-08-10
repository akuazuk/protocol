#!/usr/bin/env bash
# Install GCE night cron: 02:00 UTC main, 03:00 UTC retry (+1h).
# Server timezone is Etc/UTC (protocol-app).
#
# From Mac:
#   bash deploy/gcp-app/install_night_cron.sh --remote
# On VM:
#   bash deploy/gcp-app/install_night_cron.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE=0
for arg in "$@"; do
  case "$arg" in
    --remote) REMOTE=1 ;;
    -h|--help)
      echo "Usage: deploy/gcp-app/install_night_cron.sh [--remote]"
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
  local data="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}"
  sudo mkdir -p "${data}/logs" "${data}/state" "${data}/inbound/extract" "${data}/staging"
  sudo chown -R "$(whoami):$(whoami)" "${data}/logs" "${data}/state" "${data}/inbound" "${data}/staging"
  if [[ ! -x /opt/protocol/venv-mis/bin/python ]]; then
    bash "${proto_root}/deploy/gcp-app/setup_mis_venv.sh"
  fi
  local cron_file="/tmp/protocol-night.cron"
  cat >"$cron_file" <<EOF
# Protocol E2: Marina extract only from GCE (server time = UTC)
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PROTOCOL_ROOT=${proto_root}
GCE_MO_DATA_ROOT=${data}
# 02:00 server/UTC - main extract+score for yesterday (Europe/Minsk)
0 2 * * * ${proto_root}/deploy/gcp-app/night_mis_pipeline.sh main
# 03:00 server/UTC - retry +1h if main failed
0 3 * * * ${proto_root}/deploy/gcp-app/night_mis_pipeline.sh retry
EOF
  crontab "$cron_file"
  rm -f "$cron_file"
  echo "CRON_INSTALLED"
  crontab -l
  timedatectl | head -5 || date -u
}

if [[ "$REMOTE" == "1" ]]; then
  gcloud config set project "$PROJECT" --quiet >/dev/null
  # Sync scripts needed on VM (home then sudo)
  tar czf - \
    -C "$ROOT" \
    deploy/gcp-app/night_mis_pipeline.sh \
    deploy/gcp-app/setup_mis_venv.sh \
    deploy/gcp-app/install_night_cron.sh \
    deploy/gcp-app/score_inbound_day.sh \
    requirements-mis-bridge.txt \
    scripts/export_mis_protocol_month.py \
    clinical_knowledge/mis_protocol_parse.py \
    | gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command='
set -euo pipefail
mkdir -p ~/protocol-night-sync
tar xzf - -C ~/protocol-night-sync
sudo mkdir -p /opt/protocol/deploy/gcp-app /opt/protocol/scripts /opt/protocol/clinical_knowledge
sudo cp -f ~/protocol-night-sync/deploy/gcp-app/*.sh /opt/protocol/deploy/gcp-app/
sudo cp -f ~/protocol-night-sync/requirements-mis-bridge.txt /opt/protocol/
sudo cp -f ~/protocol-night-sync/scripts/export_mis_protocol_month.py /opt/protocol/scripts/
sudo cp -f ~/protocol-night-sync/clinical_knowledge/mis_protocol_parse.py /opt/protocol/clinical_knowledge/
sudo chmod +x /opt/protocol/deploy/gcp-app/*.sh
sudo chown -R "$(whoami):$(whoami)" /opt/protocol/deploy/gcp-app
# venv may be root-owned from prior attempt
if [[ -d /opt/protocol/venv-mis ]]; then
  sudo chown -R "$(whoami):$(whoami)" /opt/protocol/venv-mis || true
fi
bash /opt/protocol/deploy/gcp-app/install_night_cron.sh
'
else
  install_local
fi
