# Снимок прод-состояния перед программой production readiness

Дата: 2026-09-05
Ветка задачи: `cursor/production-readiness-agent1-pc1`
Base SHA: `5ba53536` (`origin/main` HEAD)

Документ фиксирует состояние **до** любых изменений, чтобы был точный откат.
Ссылка на программу работ: `docs/plans/2026-09-05-production-readiness-v1.md`.

## 1. Где реально живёт прод

| Контур | URL / адрес | Статус |
|---|---|---|
| **GCE (действующий прод)** | `https://protocol.kravira.by`, origin `34.118.21.47:8000` | **200 OK**, работает |
| Render `protocol` (`srv-d78he6h5pdvs73b1kufg`) | `https://protocol-bimy.onrender.com` | **503 Service Suspended** |

Задокументированный в `AGENTS.md` и `.cursor/rules/*` контур - Render - **не работает**.
Единственный deploy-workflow (`.github/workflows/render-production-deploy.yml`) целится
в приостановленный сервис. Решение владельца: **GCE - единственный прод**.

## 2. Версия и код на проде

| Параметр | Значение |
|---|---|
| `/api/version` -> `version` | `2026-09-05-054540Z-mo-families-gce-sync` |
| `/api/version` -> `git_commit` | `5ba5353664d65a6b850b995ea6afbf90d40c80d8` |
| `GIT_COMMIT_SHA` в контейнере | `5ba5353664d65a6b850b995ea6afbf90d40c80d8` |
| `BUILD_VERSION` в `rag_server.py:8488` | `2026-09-05-054540Z-mo-families-gce-sync` |
| `rag_ready` | `true` |
| `clinical_knowledge.protocol_cards` | 6610 |

Развёрнутый код **совпадает** с `origin/main` HEAD. Проверка функциональности:
`POST /api/search/run` с запросом «острый бронхит у взрослых лечение» -> 200,
`finish_reason: ICD_LOOKUP`, `search_timing.path: icd_fast_lookup`, `total_ms: 329`,
корректно предложен `J20.9`. Поисковая воронка работоспособна.

## 3. Инфраструктура GCE

| Ресурс | Значение |
|---|---|
| Проект / зона | `protocol-home-e1` / `europe-central2-a` |
| VM | `protocol-app`, `e2-standard-2`, RUNNING |
| Внешний IP | `34.118.21.47` |
| Диски | `protocol-app` (boot, 20 ГБ), `protocol-data` (50 ГБ) |
| `/var/data` | 49 ГБ, занято 15 ГБ (32%) |
| Контейнер | `protocol-web`, образ `protocol-gcp-app:staging` (928 МБ), `restart=unless-stopped` |
| Запущен | 2026-09-05T06:00:08Z |
| TLS | Caddy (active), `:80`/`:443`, HSTS preload + `X-Content-Type-Options` + `Referrer-Policy` |

Лишний образ на VM: `hire-app:latest` (211 МБ) - к проекту не относится.

## 4. Зафиксированные P0-проблемы прода

Каждая подтверждена на живом хосте.

1. **Порт 8000 открыт в интернет.** Firewall-правило `protocol-allow-web` (priority 1000)
   разрешает `tcp:80,tcp:443,tcp:8000` с `0.0.0.0/0`; docker-proxy слушает `0.0.0.0:8000`.
   `curl http://34.118.21.47:8000/health/live` -> 200. TLS, HSTS и Caddy обходятся полностью.
2. **CORS открыт всем.** В контейнере `ALLOWED_ORIGINS=*` (проставляется самим
   `deploy/gcp-app/deploy_to_gce.sh:85`), что в `rag_server.py:7710-7727` разворачивается в
   `allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]`.
3. **CSP отключён.** `ENABLE_DEFAULT_CSP` в контейнере пуст; в ответах прода нет ни
   `Content-Security-Policy`, ни `X-Frame-Options`.
4. **Прод работает на staging-образе** `protocol-gcp-app:staging` - без версионирования по SHA,
   без реестра образов, без возможности откатиться на предыдущую сборку.
5. **Лишние секреты в прод-контейнере**: `RENDER_API_KEY` (Render приостановлен),
   `TELEGRAM_BOT_TOKEN`. Ключ Render в контейнере веб-приложения не нужен.
6. **SSH и RDP открыты миру**: `default-allow-ssh` (`tcp:22`) и `default-allow-rdp`
   (`tcp:3389`) с `0.0.0.0/0`. RDP на Linux-VM не нужен вообще.
7. **Бэкапов не было.** До этой сессии в проекте не существовало ни одного снапшота диска.

## 5. Созданная сеть безопасности

| Артефакт | Расположение |
|---|---|
| Полное зеркало репозитория (4.2 ГБ) | `~/protocol-backup-2026-09-05.git` |
| Снапшот прод-данных | `protocol-data-20260905` (50 ГБ, READY, storage `eu`) |
| Снапшот boot-диска | `protocol-app-boot-20260905` |
| Локальные несохранённые правки | `/tmp/protocol-local-edits/` |

## 6. Состояние git на момент старта

| Параметр | Значение |
|---|---|
| `.git` | 4.2 ГБ (pack 2.82 ГиБ + 1.32 ГиБ мусорных `tmp_pack_*`) |
| Отслеживаемых файлов | 4369 |
| Локальных ветвей | 143 (38 влиты в `origin/main`, 105 - нет) |
| Открытых PR | 9 (старейший от 2026-08-08) |
| Локальный `main` | отставал от `origin/main` на 128 коммитов, рабочее дерево грязное |

Крупнейшие блобы в истории: `corpus_vector_index/vectors.npy` (980 МБ),
8 x `output/rich_chunks/rich_chunks.*.jsonl` (~340 МБ каждый),
`corpus_chunks_parts/*.jsonl` (7 x 45 МБ, во многих ревизиях).

## 7. Процедура откатa

```bash
# Код
git clone ~/protocol-backup-2026-09-05.git protocol-restored

# Данные прода
gcloud compute disks create protocol-data-restored \
  --source-snapshot=protocol-data-20260905 --zone=europe-central2-a

# Версия прода, к которой возвращаемся
# BUILD_VERSION=2026-09-05-054540Z-mo-families-gce-sync
# git_commit=5ba5353664d65a6b850b995ea6afbf90d40c80d8
```
