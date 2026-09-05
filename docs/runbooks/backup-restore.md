# Бэкапы, восстановление и наблюдаемость

Что делать, когда данные потеряны, приложение упало или пришёл алерт.
Проверено на проде 2026-09-05.

## Что защищено и чем

| Данные | Объём | Как защищено | Восстановимо иначе |
|---|---|---|---|
| `/var/data/medical_exams` - оценки, разборы, обратная связь врачей | 2,7 ГБ | снапшот диска + логический бэкап в GCS | **нет** |
| `/var/data/rceth` - реестр лекарств | 12 ГБ | снапшот диска | да, из внешнего реестра |
| `/var/data/protocol_corpus` - индекс корпуса | 780 МБ | снапшот диска | да, из PDF в репозитории |
| Код | - | `origin/main` + зеркало | да |

Логический бэкап делается только для незаменимой части. Гонять 12 ГБ реестра
лекарств в GCS каждую ночь смысла нет: он пересобирается.

## Два контура бэкапа

Они независимы намеренно: снапшот диска не спасает от ошибки в самом проекте
GCP, а архив в GCS не спасает от порчи всей файловой системы.

**1. Снапшоты диска.** Resource policy `protocol-daily-snapshots`, ежедневно в
01:00 UTC, хранение 14 дней, привязана к `protocol-app` (загрузочный) и
`protocol-data`. Восстанавливают диск целиком и только внутрь того же проекта.

```bash
gcloud compute snapshots list --format="table(name,creationTimestamp,diskSizeGb)"
```

**2. Логический бэкап в GCS.** systemd-таймер `protocol-backup.timer`,
ежедневно в 02:30 UTC, в `gs://protocol-home-e1-backups` (europe-central2,
NEARLINE, автоудаление через 30 дней). Переносимый архив ~246 МБ.

```bash
gcloud storage ls -l "gs://protocol-home-e1-backups/medical_exams/**/*.tar.zst" | tail -5
```

Базы SQLite копируются через backup API, а не `cp`. На живой базе в режиме WAL
копия файла может застать транзакцию посередине; backup API даёт согласованный
снимок без остановки приложения. Каждая копия проверяется `PRAGMA
integrity_check` сразу при снятии, архив после выгрузки скачивается обратно и
пробуется на распаковку - иначе скрипт рапортовал бы успех и при обрезанном
файле в бакете.

## Учения

Бэкап, который ни разу не разворачивали, - предположение, а не бэкап.
`protocol-restore-drill.timer` раз в месяц (первое воскресенье, 04:00 UTC)
разворачивает свежий архив в `/var/tmp`, сверяет sha256 всех файлов с
манифестом, проверяет целостность баз и печатает число записей. Прод не трогает.

Прогнать вручную:

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a \
  --command='sudo /usr/local/bin/protocol-restore --latest --drill'
```

Ожидаемый вид (замер 2026-09-05):

```text
mis_zoho.sqlite: таблиц 3, записей 4727935
mo_analytics.sqlite: таблиц 28, записей 783875
mo_lab.sqlite: таблиц 2, записей 449228
mo_warehouse.sqlite: таблиц 0, записей 0
jsonl файлов восстановлено: 200
```

`mo_warehouse.sqlite` пустая штатно - см. план
`docs/plans/2026-09-05-mo-warehouse-storage-v1.md`.

Если число записей заметно упало по сравнению с прошлым учением - это не повод
восстанавливаться, но повод разобраться, что удалило данные.

## Восстановление данных

Приложение обязательно остановить: подменять базы под работающим сервисом
нельзя, скрипт это проверяет и откажется работать.

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a
sudo docker stop protocol-web
sudo /usr/local/bin/protocol-restore --latest --target /var/data/medical_exams
sudo docker start protocol-web
curl -sS https://protocol.kravira.by/health/live
```

Текущие данные не удаляются, а отодвигаются в
`/var/data/medical_exams.before-restore.<метка>`. Не удаляй этот каталог, пока
не убедился, что восстановленные данные верны; после проверки удали - он
занимает те же 2,7 ГБ.

Конкретный архив вместо самого свежего: `--archive gs://.../medical_exams-<метка>.tar.zst`.

## Восстановление диска из снапшота

Нужно, если повреждена файловая система или потерян весь диск данных.

```bash
# 1. Остановить VM
gcloud compute instances stop protocol-app --zone=europe-central2-a

# 2. Создать диск из снапшота
gcloud compute disks create protocol-data-restored \
  --source-snapshot=<имя-снапшота> --zone=europe-central2-a

# 3. Отключить старый, подключить восстановленный
gcloud compute instances detach-disk protocol-app --disk=protocol-data --zone=europe-central2-a
gcloud compute instances attach-disk protocol-app --disk=protocol-data-restored \
  --device-name=protocol-data --zone=europe-central2-a

# 4. Запустить и проверить
gcloud compute instances start protocol-app --zone=europe-central2-a
curl -sS https://protocol.kravira.by/health/live
```

Старый диск не удаляй до полной проверки.

## Алерты

Все уходят на email владельца проекта (канал «Protocol: владелец проекта»).

| Алерт | Когда | Первое действие |
|---|---|---|
| Protocol недоступен | `/health/live` не проходит из 2+ регионов 5 минут | `sudo docker ps`, `sudo docker logs --tail=100 protocol-web` |
| Сбой бэкапа или учения | запись `status=failed` в Cloud Logging | `sudo journalctl -u protocol-backup -n 50` |
| Бэкапа не было более суток | нет успеха 23,5 часа | `systemctl list-timers 'protocol-*'`, `df -h` |
| Заканчивается место | диск занят >85% 10 минут | `sudo /usr/local/bin/protocol-vm-cleanup` |
| Всплеск ошибок | >10 ERROR за 5 минут | `gcloud logging read 'logName:"protocol_web" AND severity>=ERROR' --freshness=30m` |

Алерт на **отсутствие** бэкапа существует отдельно от алерта на сбой не для
симметрии: сбой ловится только если скрипт запустился. Если таймер отключён,
VM выключена или кончилось место, ошибки не будет вообще - и без второго алерта
об этом узнали бы при первой же аварии.

Проверить, что алерты живы:

```bash
gcloud alpha monitoring policies list --format="table(displayName,enabled)"
gcloud alpha monitoring channels list --format="table(displayName,labels.email_address,enabled)"
```

## Место на диске

Загрузочный диск - 20 ГБ, и образ приложения собирается на самой VM, поэтому
кэш сборки Docker растёт с каждым деплоем. 2026-09-05 он занял 8,4 ГБ,
свободными оставалось 3,4 ГБ (82%). Разово вычищено 9,6 ГБ, дальше:

- логи Docker ротируются (100 МБ x 3 на контейнер) - до этого не ротировались
  вообще, и цикл ошибок мог добить остаток места;
- `protocol-vm-cleanup.timer` раз в неделю чистит кэш сборки и образы старше
  недели;
- алерт на 85%.

Правильное решение - собирать образ в CI и тянуть на VM готовый из Artifact
Registry, тогда кэша сборки на VM не будет вовсе. Пока не сделано.

## Наблюдаемость

Ops Agent (`google-cloud-ops-agent`) шлёт в Cloud Logging логи контейнера
приложения и метрики хоста. Собираем узко: только контейнер приложения, без
системных логов - шума много, пользы для эксплуатации нет.

Sentry намеренно **не** подключён. Он был бы ещё одним зарубежным обработчиком
персональных данных для белорусской клиники, а Cloud Logging уже в контуре
GCP, который описан в реестре обработки ПДн. Задачу «узнать, что приложение
сломалось» это решает так же.

Клинического текста в логах нет по устройству: телеметрия пациентов пишется по
списку разрешённых полей (`clinical_knowledge/patient_analytics.py`),
shadow-лог роутера пишет только метаданные классификации, трассировки Python не
содержат значений локальных переменных. Единственный GET-эндпоинт с текстом в
строке запроса - `/api/methodist/protocol-search?q=` - принимает название
протокола, до 120 символов, не данные пациента.

При добавлении логирования держи этот инвариант: свободный клинический текст в
логи не попадает. Иначе ПДн окажутся в Cloud Logging, а это уже вопрос к
реестру `docs/compliance/2026-09-05-personal-data-register-v1.md`.

## Установка с нуля

Если VM пересоздана:

```bash
bash deploy/gcp-app/install_backup_timer.sh     # бэкап + учения
bash deploy/gcp-app/install_vm_maintenance.sh   # ротация логов + очистка
```

Снапшоты диска, алерты и uptime-проверка живут в проекте GCP и пересоздания VM
не требуют. Привязать политику снапшотов к новым дискам:

```bash
gcloud compute disks add-resource-policies protocol-data \
  --zone=europe-central2-a --resource-policies=protocol-daily-snapshots
```
