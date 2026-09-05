# Прод Protocol на GCE: runbook

Дата: 2026-09-05. Заменяет Render-контур как источник истины по production.

## 1. Что где находится

| Что | Значение |
|---|---|
| Домен | `https://protocol.kravira.by` |
| Внешний IP | `34.118.21.47` (`protocol-app-ip`, статический) |
| Проект / зона | `protocol-home-e1` / `europe-central2-a` |
| VM | `protocol-app`, `e2-standard-2` |
| Диски | `protocol-app` (boot, 20 ГБ), `protocol-data` (50 ГБ, `/var/data`) |
| Контейнер | `protocol-web`, `restart=unless-stopped` |
| Образ | `protocol-gcp-app:<sha12>`, плюс подвижный тег `:staging` |
| TLS | Caddy на VM, `:80`/`:443` -> `127.0.0.1:8000` |
| Данные МО | `/var/data/medical_exams` |
| Корпус | `/var/data/protocol_corpus` |

**Render не прод.** Сервис `protocol` (`srv-d78he6h5pdvs73b1kufg`,
`protocol-bimy.onrender.com`) приостановлен и отдаёт `503`. Скрипты
`scripts/ops/render_*` оставлены как legacy и блокируют деплой; снять блок
можно только явным `ALLOW_LEGACY_RENDER=1`.

## 2. Сеть

Firewall (после 2026-09-05):

| Правило | Порты | Источник |
|---|---|---|
| `protocol-allow-web` | `tcp:80`, `tcp:443` | `0.0.0.0/0` |
| `default-allow-ssh` | `tcp:22` | `0.0.0.0/0` |
| `default-allow-internal` | все | `10.128.0.0/9` |
| `default-allow-icmp` | icmp | `0.0.0.0/0` |

Порт `8000` **закрыт** и контейнер публикуется только на `127.0.0.1:8000`:
до 2026-09-05 приложение полностью отдавалось по plaintext HTTP в обход TLS.
Не открывать его снова; для отладки использовать
`gcloud compute ssh protocol-app --zone=europe-central2-a --command='curl -s http://127.0.0.1:8000/health/live'`.

`default-allow-ssh` открыт всему интернету. Это осознанно оставленный
компромисс: у операторов динамические адреса. Улучшение - IAP-туннель и
`--tunnel-through-iap` вместо публичного `:22`.

## 3. Релиз

Предварительные условия:

- локальный `HEAD` **ровно** равен `origin/main` (скрипт это проверяет и иначе отказывает);
- `BUILD_VERSION` в `rag_server.py` поднят в этом же коммите;
- зелёный CI на merge-коммите.

```bash
git fetch origin main
git checkout main && git reset --hard origin/main   # только в чистом чекауте
bash deploy/gcp-app/deploy_to_gce.sh
```

Что делает скрипт:

1. отказывается работать, если `HEAD != origin/main`;
2. собирает env и заливает секреты в Secret Manager (значения не печатаются);
3. **`git archive` ровно релизного SHA** - на VM уезжает только закоммиченное.
   Раньше здесь был `tar` рабочего дерева, из-за чего в `/opt/protocol`
   попадали локальные правки и мусор macOS (`._*`);
4. собирает образ и тегирует его **по SHA** (`protocol-gcp-app:<sha12>`),
   храня последние 5 версий - это и есть материал для откатa;
5. запускает контейнер на `127.0.0.1:8000`;
6. ждёт `/health/live` внутри VM, затем проверяет домен;
7. сверяет `/api/version` с `BUILD_VERSION` **из релизного коммита** и
   `git_commit` с релизным SHA;
8. при любой неудаче **автоматически откатывается** на предыдущий образ.

Альтернатива - GitHub Action `Production GCE release` (ручной запуск с SHA).
Он требует настройки доступа GitHub -> GCP, см. раздел 6.

## 4. Проверка после релиза

```bash
curl -fsS https://protocol.kravira.by/health/live
curl -fsS https://protocol.kravira.by/api/version   # version + git_commit
curl -fsS -X POST https://protocol.kravira.by/api/search/run \
  -H 'Content-Type: application/json' \
  -d '{"query":"острый бронхит у взрослых лечение"}'
```

Ориентиры здорового прода (замер 2026-09-05):

| Показатель | Значение |
|---|---|
| `/health/live` | 200 |
| `rag_ready` | `true` (после прогрева; сразу после рестарта короткое время `false`) |
| `protocol_cards` | 6610 |
| Тёплый `search_timing.total_ms` | ~330 мс |
| Первый запрос после рестарта | до ~35 с, это ленивая загрузка корпуса |

Заголовки безопасности обязаны присутствовать:
`Content-Security-Policy`, `Strict-Transport-Security`, `X-Frame-Options`,
`X-Content-Type-Options`, `Referrer-Policy`. Проверка, что CORS закрыт:

```bash
curl -s -D - -o /dev/null -H 'Origin: https://evil.example' \
  https://protocol.kravira.by/api/version | grep -i access-control-allow-origin
# ожидается: пусто
```

## 5. Откат

Автоматический откат встроен в деплой. Вручную:

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a
sudo docker images protocol-gcp-app          # выбрать предыдущий <sha12>
sudo docker rm -f protocol-web
# запустить с тем же набором -v/-e, что в deploy/gcp-app/deploy_to_gce.sh,
# указав нужный тег образа
```

Данные:

```bash
gcloud compute snapshots list
gcloud compute disks create protocol-data-restored \
  --source-snapshot=<snapshot> --zone=europe-central2-a
```

## 6. Чего пока нет

- **Автоматического деплоя из GitHub.** Workflow
  `.github/workflows/gce-production-deploy.yml` написан, но не включён: нужны
  Workload Identity Federation и deploy-сервисный аккаунт, а значит - решение
  владельца о выдаче GitHub доступа в GCP. Требуемые переменные репозитория
  перечислены в заголовке workflow. До этого канонический путь - запуск
  `deploy_to_gce.sh` релиз-координатором.
- **Регулярных снапшотов.** Первые снапшоты в истории проекта созданы
  2026-09-05 вручную. Нужна политика расписания (Фаза 9).
- **Отдельного staging.** Тег образа называется `:staging`, но контур один -
  это и есть прод. Имя историческое, менять вместе с CI/CD.
