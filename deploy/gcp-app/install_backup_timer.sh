#!/usr/bin/env bash
# Ставит на VM ночной логический бэкап клинических данных.
#
# Идемпотентно: повторный запуск обновляет скрипт и unit-файлы.
# Запускать с Mac: bash deploy/gcp-app/install_backup_timer.sh
set -euo pipefail

ZONE="${GCE_ZONE:-europe-central2-a}"
VM="${GCE_VM:-protocol-app}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "== заливаю скрипты на $VM =="
gcloud compute scp \
  "$REPO_ROOT/deploy/gcp-app/backup_medical_exams.sh" \
  "$REPO_ROOT/deploy/gcp-app/restore_medical_exams.sh" \
  "$VM:/tmp/" --zone="$ZONE" --quiet

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command='
set -euo pipefail

# zstd и rsync нужны скрипту; на минимальном образе их может не быть.
missing=""
for tool in zstd rsync; do command -v "$tool" >/dev/null || missing="$missing $tool"; done
if [ -n "$missing" ]; then
  echo "== доустанавливаю:$missing =="
  sudo apt-get update -qq
  sudo apt-get install -y -qq $missing
fi

sudo install -m 0755 /tmp/backup_medical_exams.sh /usr/local/bin/protocol-backup
sudo install -m 0755 /tmp/restore_medical_exams.sh /usr/local/bin/protocol-restore
rm -f /tmp/backup_medical_exams.sh /tmp/restore_medical_exams.sh

sudo tee /etc/systemd/system/protocol-backup.service >/dev/null <<"UNIT"
[Unit]
Description=Логический бэкап клинических данных МО в GCS
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/protocol-backup
# Бэкап не должен мешать врачам: отдаём ему низкий приоритет CPU и диска.
Nice=10
IOSchedulingClass=idle
TimeoutStartSec=3600
UNIT

sudo tee /etc/systemd/system/protocol-backup.timer >/dev/null <<"UNIT"
[Unit]
Description=Ночной бэкап клинических данных МО

[Timer]
# 02:30 UTC - после снапшота диска (01:00) и до утреннего приёма.
OnCalendar=*-*-* 02:30:00 UTC
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
UNIT

# Учение по восстановлению. Бэкап, который ни разу не разворачивали, - это
# предположение, а не бэкап. Раз в месяц проверяем на самом свежем архиве.
sudo tee /etc/systemd/system/protocol-restore-drill.service >/dev/null <<"UNIT"
[Unit]
Description=Учение по восстановлению клинических данных из бэкапа
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/protocol-restore --latest --drill
Nice=15
IOSchedulingClass=idle
TimeoutStartSec=3600
UNIT

sudo tee /etc/systemd/system/protocol-restore-drill.timer >/dev/null <<"UNIT"
[Unit]
Description=Ежемесячное учение по восстановлению

[Timer]
OnCalendar=Sun *-*-01..07 04:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now protocol-backup.timer
sudo systemctl enable --now protocol-restore-drill.timer
echo "== таймеры =="
systemctl list-timers "protocol-*" --no-pager | head -4
'

echo
echo "Готово. Проверить вручную:"
echo "  gcloud compute ssh $VM --zone=$ZONE --command='sudo systemctl start protocol-backup && sudo journalctl -u protocol-backup -n 30 --no-pager'"
