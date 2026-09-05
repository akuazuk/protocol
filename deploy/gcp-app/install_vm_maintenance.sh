#!/usr/bin/env bash
# Гигиена загрузочного диска VM.
#
# Зачем: образ приложения собирается на самой VM, поэтому кэш сборки Docker
# растёт с каждым деплоем и ничем не ограничен. 2026-09-05 он занял 8,4 ГБ из
# 20 ГБ загрузочного диска - свободными оставались 3,4 ГБ. Логи Docker при
# драйвере json-file без max-size тоже не ротируются: цикл ошибок способен
# добить остаток и уронить приложение по нехватке места.
#
# Ставит:
#   - ротацию логов Docker (100 МБ x 3 файла на контейнер);
#   - недельную очистку кэша сборки и оборванных образов;
#   - алерт на нехватку места ставится отдельно, в install_backup_timer.sh.
set -euo pipefail

ZONE="${GCE_ZONE:-europe-central2-a}"
VM="${GCE_VM:-protocol-app}"

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command='
set -euo pipefail

echo "== ротация логов Docker =="
# Слияние с существующим daemon.json, если он появится: перезатирать чужие
# настройки демона нельзя.
sudo python3 - <<"PY"
import json
import pathlib

path = pathlib.Path("/etc/docker/daemon.json")
cfg = {}
if path.exists():
    try:
        cfg = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        raise SystemExit("daemon.json есть, но не парсится - разберись вручную")

cfg["log-driver"] = "json-file"
cfg["log-opts"] = {"max-size": "100m", "max-file": "3"}

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(cfg, indent=2) + "\n")
print("  " + json.dumps(cfg))
PY

# reload, а не restart: перезапуск демона уронил бы работающий контейнер.
# Действующий контейнер сохранит старую настройку логов до следующего деплоя,
# новые - подхватят сразу.
sudo systemctl reload docker
echo "  демон перечитал конфиг (работающий контейнер не тронут)"

echo "== недельная очистка =="
sudo tee /usr/local/bin/protocol-vm-cleanup >/dev/null <<"SCRIPT"
#!/usr/bin/env bash
# Освобождает место на загрузочном диске. Запускается таймером раз в неделю.
set -euo pipefail
free_before=$(df --output=avail / | tail -1)

# Кэш сборки старше недели: нужен только для ускорения повторной сборки,
# на восстановление данных не влияет.
docker builder prune -af --filter 'until=168h' >/dev/null 2>&1 || true
docker image prune -af --filter 'until=168h' >/dev/null 2>&1 || true
docker container prune -f --filter 'until=168h' >/dev/null 2>&1 || true
journalctl --vacuum-size=200M >/dev/null 2>&1 || true

free_after=$(df --output=avail / | tail -1)
freed_mb=$(( (free_after - free_before) / 1024 ))
avail_gb=$(( free_after / 1048576 ))
echo "освобождено ${freed_mb} МБ, свободно ${avail_gb} ГБ"

# Место кончается - сообщаем в Cloud Logging, на этом висит алерт.
if [ "$avail_gb" -lt 4 ]; then
  gcloud logging write protocol-vm-health \
    "{\"event\":\"disk_low\",\"status\":\"failed\",\"detail\":\"свободно ${avail_gb} ГБ на загрузочном диске\"}" \
    --payload-type=json --severity=ERROR >/dev/null 2>&1 || true
fi
SCRIPT
sudo chmod 0755 /usr/local/bin/protocol-vm-cleanup

sudo tee /etc/systemd/system/protocol-vm-cleanup.service >/dev/null <<"UNIT"
[Unit]
Description=Очистка загрузочного диска VM (кэш сборки Docker, журналы)

[Service]
Type=oneshot
ExecStart=/usr/local/bin/protocol-vm-cleanup
Nice=15
IOSchedulingClass=idle
UNIT

sudo tee /etc/systemd/system/protocol-vm-cleanup.timer >/dev/null <<"UNIT"
[Unit]
Description=Недельная очистка загрузочного диска VM

[Timer]
OnCalendar=Mon *-*-* 03:30:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now protocol-vm-cleanup.timer

echo "== проверка: прогон очистки =="
sudo /usr/local/bin/protocol-vm-cleanup

echo "== таймеры =="
systemctl list-timers "protocol-*" --no-pager | head -5
'
