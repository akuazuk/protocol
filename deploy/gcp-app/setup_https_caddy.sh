#!/usr/bin/env bash
# Install/configure Caddy TLS for protocol.kravira.by on GCE protocol-app.
#
# From Mac:
#   bash deploy/gcp-app/setup_https_caddy.sh --remote
# On VM:
#   sudo bash deploy/gcp-app/setup_https_caddy.sh
#
# Pre-req: DNS A protocol.kravira.by → this VM's static IP (see INVENTORY.md).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOMAIN="${PROTOCOL_DOMAIN:-protocol.kravira.by}"
PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
EXPECTED_IP="${PROTOCOL_EXPECTED_IP:-34.118.21.47}"

install_on_host() {
  export DEBIAN_FRONTEND=noninteractive
  if ! command -v caddy >/dev/null 2>&1; then
    apt-get update -y
    apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
      | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
      | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    apt-get update -y
    apt-get install -y caddy
  fi

  mkdir -p /etc/caddy
  cp /tmp/protocol-Caddyfile /etc/caddy/Caddyfile

  # Quick local health of upstream
  if ! curl -fsS --max-time 5 http://127.0.0.1:8000/health/live >/dev/null; then
    echo "ERROR: upstream http://127.0.0.1:8000/health/live failed (is protocol-web up?)" >&2
    exit 1
  fi

  RESOLVED="$(getent ahostsv4 "$DOMAIN" 2>/dev/null | awk '{print $1; exit}' || true)"
  echo "DNS $DOMAIN -> ${RESOLVED:-unresolved} (want $EXPECTED_IP)"
  DNS_OK=0
  if [[ "$RESOLVED" == "$EXPECTED_IP" ]]; then
    DNS_OK=1
  else
    echo "WARN: DNS does not point to this VM yet. Let's Encrypt will fail until A-record is set."
    echo "Set: $DOMAIN  A  $EXPECTED_IP   (remove CNAME to Render)"
  fi

  caddy validate --config /etc/caddy/Caddyfile
  systemctl enable caddy
  systemctl restart caddy
  sleep 3
  if ! systemctl is-active --quiet caddy; then
    echo "ERROR: caddy failed to start" >&2
    journalctl -u caddy -n 40 --no-pager || true
    exit 1
  fi
  systemctl --no-pager --full status caddy | head -20 || true

  # Hit THIS VM only (--resolve), never public DNS (may still be Render).
  echo "Trying HTTPS via VM IP ($EXPECTED_IP)..."
  if curl -fsS --max-time 45 \
      --resolve "${DOMAIN}:443:${EXPECTED_IP}" \
      "https://${DOMAIN}/health/live"; then
    echo
    echo "HTTPS_OK (via $EXPECTED_IP) https://${DOMAIN}/health/live"
    if [[ "$DNS_OK" -ne 1 ]]; then
      echo "NOTE: public DNS still not on GCE; flip A-record for end users."
      exit 2
    fi
  else
    echo
    echo "HTTPS_PENDING: Caddy up; LE needs DNS A $DOMAIN → $EXPECTED_IP"
    echo "  dig +short $DOMAIN A"
    echo "  sudo systemctl restart caddy"
    journalctl -u caddy -n 30 --no-pager || true
    exit 2
  fi
}

if [[ "${1:-}" == "--remote" ]]; then
  gcloud config set project "$PROJECT" --quiet >/dev/null
  gcloud compute scp "$ROOT/deploy/gcp-app/Caddyfile" \
    "${VM}:/tmp/protocol-Caddyfile" --zone="$ZONE" --quiet
  gcloud compute scp "$ROOT/deploy/gcp-app/setup_https_caddy.sh" \
    "${VM}:/tmp/setup_https_caddy.sh" --zone="$ZONE" --quiet
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
sudo cp /tmp/setup_https_caddy.sh /opt/protocol/deploy/gcp-app/setup_https_caddy.sh 2>/dev/null || {
  sudo mkdir -p /opt/protocol/deploy/gcp-app
  sudo cp /tmp/setup_https_caddy.sh /opt/protocol/deploy/gcp-app/setup_https_caddy.sh
}
sudo chmod +x /opt/protocol/deploy/gcp-app/setup_https_caddy.sh /tmp/setup_https_caddy.sh
sudo PROTOCOL_DOMAIN='$DOMAIN' PROTOCOL_EXPECTED_IP='$EXPECTED_IP' bash /tmp/setup_https_caddy.sh
"
  exit $?
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root on the VM, or use --remote from Mac." >&2
  exit 2
fi
install_on_host
