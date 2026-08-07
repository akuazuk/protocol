#!/bin/bash
# First-boot GCE setup for protocol-app (Debian 12).
set -euo pipefail

DATA_DEV="/dev/disk/by-id/google-protocol-data"
MOUNT="/var/data"
MARKER="/var/lib/protocol-startup.done"

if [[ -f "$MARKER" ]]; then
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl gnupg jq python3 python3-venv python3-pip

# Docker
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable --now docker
fi

# Data disk
mkdir -p "$MOUNT"
if [[ -b "$DATA_DEV" ]]; then
  if ! blkid "$DATA_DEV" >/dev/null 2>&1; then
    mkfs.ext4 -F -L protocol-data "$DATA_DEV"
  fi
  UUID="$(blkid -s UUID -o value "$DATA_DEV")"
  if ! grep -q "$UUID" /etc/fstab; then
    echo "UUID=${UUID} ${MOUNT} ext4 defaults,nofail 0 2" >> /etc/fstab
  fi
  mount -a || mount "$DATA_DEV" "$MOUNT"
fi

mkdir -p \
  "${MOUNT}/medical_exams/inbound/extract" \
  "${MOUNT}/medical_exams/warehouse" \
  "${MOUNT}/medical_exams/secure_cases" \
  "${MOUNT}/medical_exams/llm_inbox" \
  "${MOUNT}/medical_exams/llm_outbox" \
  "${MOUNT}/medical_exams/reports" \
  "${MOUNT}/medical_exams/gold_review"
chmod 755 "$MOUNT" "${MOUNT}/medical_exams"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MARKER"
echo "protocol-app startup complete" | tee /var/log/protocol-startup.log
