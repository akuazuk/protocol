# Ежедневная сверка КП Минздрава + вкладка в МО Аналитика (v1)

Дата: 2026-08-13  
Статус: **archived** (преемник: `2026-08-13-minzdrav-kp-daily-sync-v2.md`)  
Связанные: `2026-08-09-mo-protocol-nav-reader-v2.md`,
`2026-08-08-mo-icd-first-kp-suggest-v1.md`,
`2026-08-08-mo-analytics-ui-target-v2.md`,
`2026-08-08-mo-analytics-mz-sheet-layers-v2.md`,
`2026-08-07-by-home-gcp-llm-split-v1.md`,
`2026-08-10-mo-night-speed-skip-alerts-v1.md`.

---

## 1. Контекст

Сверка 2026-08-13: на сайте МЗ **396** уникальных PDF, в `index.csv` проекта **345**.
Совпало 335. На сайте нет у нас **61** файл, из них **~20 постановлений апреля-июня 2026**
(ОКС, АГ, ТЭЛА, ЗНО, родоразрешение, бронхиты и др.) плюс пакет ревматологии КП1-КП32.

Сейчас обновление ручное: `download_minzdrav_protocols.py` → `build_index.py` →
corpus pipeline → summaries/catalog → копирование на GCE
`/var/data/protocol_corpus`. Ночь MIS (02:00 UTC) корпус не трогает.
Канон меню МО - 6 пунктов; отдельной статистики по КП нет.

Цель v1: **каждый день** сверять сайт МЗ с корпусом на GCE, подтягивать новые и
изменённые PDF, прогонять производные артефакты по всем потребителям, показывать
статистику на отдельной вкладке МО Аналитика. Исторические PDF не удалять
(ссылки из старых разборов должны открываться).

---

## 2. Что изменится в проде

| Было | Станет |
|--|--|
| Каталог КП застыл (~24.06.2026) | Ежедневный crawl МЗ на GCE, incremental download |
| Корпус копируется руками при деплое | SSOT корпуса: `/var/data/protocol_corpus` + reload без полного redeploy |
| Нет экрана «что нового в КП» | Вкладка **Протоколы МЗ** в МО Аналитика |
| Suggest / план / RAG / viewer на старых PDF | Те же блоки читают обновлённый catalog + cards + summaries |
| Новые рубрики МЗ ломают `ALLOWED_SPECIALTY_SLUGS` | Рубрики подтягиваются из crawl; код + `verify_minzdrav_rubrics.py` |

Не меняем в v1: полный recompute всех исторических МО при каждом новом КП
(это отдельный хвост, см. шаг G). Night extract MIS не смешиваем с crawl КП.

---

## 3. Метрики

| Метрика | Было (2026-08-13) | Цель |
|--|--|--|
| Покрытие сайт МЗ ↔ корпус (unique basename / post) | 335/396 ≈ 85% | ≥99% файлов с сайта есть в корпусе (или явно `superseded`/`blocked`) |
| Лаг нового постановления | недели | ≤26 ч после появления ссылки на minzdrav.gov.by |
| Вкладка МО: дата последней сверки | нет | видна, `status=success` за вчера |
| Алерт при fail crawl | нет | Telegram ≤15 мин после job (как night MIS) |
| Потребители после apply | старый catalog | suggest, RAG, viewer, ICD-index, summaries, plan-zone читают новый snapshot |
| Ложные удаления PDF | риск при naive sync | 0: файлы с диска не удаляем автоматически |

---

## 4. Архитектура

```text
01:00 UTC GCE cron (до night MIS 02:00)
  1. crawl minzdrav.gov.by  (24 рубрики → PDF/DOC URL + sha256)
  2. diff vs /var/data/protocol_corpus/_sync/manifest.jsonl
  3. download new + changed (по sha); never unlink old
  4. mark superseded: local file not on site → status=superseded, keep file
  5. incremental rebuild только для changed paths:
       index.csv → chunks/cards → protocol_catalog.jsonl
       → protocol_icd_profiles → summaries (новые PDF в очередь)
  6. atomic swap snapshot + write kp_sync_YYYY-MM-DD.json
  7. SIGHUP / cache-bust rag_server (catalog, summaries, cards)
  8. Telegram если added>0 or fail

МО Аналитика → GET /api/mo/kp-sync  → вкладка «Протоколы МЗ»
```

Источник истины PDF: **диск GCE**, не git. В репозитории - скрипты, схема status JSON,
тесты. `index.csv` в git обновляется периодическим PR (не каждый день).

Расписание **до** night score, чтобы вчерашние МО уже оценивались по свежим КП.

---

## 5. Потребители (все блоки, которые обязаны увидеть обновление)

| Блок | Артефакт | Что сделать при apply |
|--|--|--|
| Подбор КП в разборе случая | `protocol_cards.jsonl`, catalog, ICD profiles | reload registry; новые path в top-3 |
| Зона «План по протоколу» | matched KP + summaries/checklists | новые КП участвуют в alignment; superseded не выбирать как primary |
| RAG / поиск врача | chunks, lex shards, vector sidecar | incremental reindex changed paths |
| Viewer / «Навигация по КП» | PDF mount + nav chunks | новые файлы открываются; старые path живы |
| Каталог методиста / search | `protocol_catalog.jsonl`, `index.csv` | новые titles/годы/посты |
| LLM grade / kz_checklist | `protocol_summaries/json` | очередь summaries только для added/updated |
| Рубрики UI / specialty filter | `ALLOWED_SPECIALTY_SLUGS` | если на сайте новая `.php` рубрика - PR + smoke |
| Деплой Docker | volume `/var/data/protocol_corpus` | не копировать PDF из git; mount как сейчас |
| Исторические разборы | старый `source_path` | не 404: файл остаётся; бейдж «заменён постановлением …» |

---

## 6. Шаги

### A. Разовый catch-up (закрыть дыру 2026)

- [ ] A1. На GCE (не Mac): crawl + download missing 61 + `--refresh` для файлов с тем же именем и другим sha.
- [ ] A2. Не удалять 10 «только локальных» (бронхит 2012, неонатология 2022, …) - пометить `superseded_by`.
- [ ] A3. Полный rebuild catalog/cards/summaries для **новых** PDF (ревматология КП1-32 и посты 04-06.2026).
- [ ] A4. Выкладка в `/var/data/protocol_corpus`, restart app, smoke: `/api/corpus-stats` + открыть 2 новых PDF в viewer.
- [ ] A5. Не делать полный recompute июля, пока не готов шаг G.

### B. Daily crawler (код)

- [ ] B1. Вынести crawl из `download_minzdrav_protocols.py` в `scripts/kp_sync/crawl_minzdrav.py`:
      рубрики, URL, basename, HTTP size/etag если есть, sha256 после download.
- [ ] B2. `scripts/kp_sync/diff_catalog.py`: added / updated / unchanged / superseded / errors.
      Матч: sha256 → basename → пост `ДД.ММ.ГГГГ №N` (как сверка 13.08).
- [ ] B3. `scripts/kp_sync/apply_download.py`: качать только added/updated; atomic write `.part`.
- [ ] B4. Status JSON: `/var/data/protocol_corpus/_sync/kp_sync_YYYY-MM-DD.json`
      (`status`, `added[]`, `updated[]`, `superseded[]`, `errors[]`, `site_count`, `local_count`).
- [ ] B5. Тесты на HTML-фикстуре (не живой сайт): 1 new, 1 rename same post, 1 sha change, 1 gone.

### C. Incremental corpus apply

- [ ] C1. `build_index.py` умеет `--only-paths` / merge в существующий `index.csv`.
- [ ] C2. `corpus_pipeline.run_pipeline` - режим `--changed-only` (сейчас пишет chunks.jsonl целиком - нужен merge, иначе ночь раздует диск и время).
- [ ] C3. `build_protocol_catalog.py` + `build_protocol_icd_index.py` от changed paths.
- [ ] C4. Summaries: очередь `protocol_summaries/_queue.jsonl`; LLM **не** в том же cron, что crawl
      (отдельный job или следующий слот; Gemini только GCE, не Mac).
- [ ] C5. Cache-bust: `clear_protocol_summary_cache` + reload cards/catalog в процессе
      (endpoint `/api/internal/reload-corpus` под METHODIST / ops token, или restart container).
- [ ] C6. Если появилась новая рубрика `.php` - job `status=needs_code` + алерт; не молча игнорить.

### D. Cron на GCE (отдельно от MIS)

- [ ] D1. Скрипт `deploy/gcp-app/night_kp_sync.sh` (owner `pavel`, self-heal как у MIS env).
- [ ] D2. Cron **01:00 UTC** daily; retry **01:40**; check **01:50** → Telegram.
- [ ] D3. Не класть MIS DSN в этот job. Сеть: исходящий HTTPS на minzdrav.gov.by (проверить SSL с GCE).
- [ ] D4. Лимиты: polite delay, max N downloads/night (остаток на retry), timeout.
- [ ] D5. Установить через `install_night_cron.sh` (добавить строки, не ломая 02:00 MIS).

### E. Вкладка «Протоколы МЗ» в МО Аналитика

Канон меню был 6 пунктов. Владелец просит **отдельную вкладку** - расширяем меню до 7.
Не прятать в «Справка»: это операционный экран свежести нормы, не легенда.

- [ ] E1. Пункт меню после «Отчёты»: **Протоколы МЗ** (`data-page="kp-sync"`).
      Обновить `ui-target-v2` / `dashboards-zones-first-v2` (7 пунктов).
- [ ] E2. API `GET /api/mo/kp-sync?days=30` (без ПДн): last status, counts, списки added/updated/superseded.
- [ ] E3. Экран:
      - строка фактов: сверка, сайт N, корпус M, лагов нет/есть;
      - таблица «новые / изменённые за 30 дней» (дата поста, №, рубрика, файл, статус);
      - таблица «заменены на сайте» (наш файл → новый, если известен);
      - ошибки crawl красным, клик не ведёт в очередь случаев.
- [ ] E4. Ссылка «открыть КП» → существующий proto-viewer.
- [ ] E5. Стиль как остальные вкладки МО (RU, без em dash, те же таблицы/chips).

### F. Учёт во всех блоках (контракт apply)

После успешного apply один snapshot id (`corpus_generation` / sha манифеста) должен
быть виден:

- [ ] F1. `/api/corpus-stats` и `/api/version` (или поле `kp_sync`) - дата сверки + counts.
- [ ] F2. Suggest: не возвращать `superseded=true` как top-1, если есть successor path.
- [ ] F3. Plan-zone: alignment только по non-superseded clinical KP (как сейчас «нет КП» ≠ штраф).
- [ ] F4. Viewer: PDF с диска; для superseded - баннер «Есть редакция от ДД.ММ.ГГГГ №N».
- [ ] F5. Doctor RAG: chunks новых path в индексе; старые path не выкидывать из индекса в v1
      (иначе исторические цитаты пустые) - понижать rank superseded.
- [ ] F6. Тесты: fixture PDF added → catalog + suggest path; superseded flag → не primary.

### G. Хвост (не блокирует v1)

- [ ] G1. Опциональный re-score последних 7/30 дней, если changed KP пересекается с ICD
      визитов (иначе план-зона молча устаревает). Отдельный флаг, не каждый night.
- [ ] G2. PR в git: периодический `index.csv` + список постов (без бинарных PDF).
- [ ] G3. Ревматология КП1-32: завести display_title из заголовка PDF, не из короткого имени файла.

---

## 7. Риски

| Риск | Митигация |
|--|--|
| Сайт МЗ меняет вёрстку / SSL | фикстуры + алерт `errors>0`; не падать night MIS |
| Полный rebuild chunks за ночь | только `--changed-only`; cap downloads |
| Удалили старый PDF → 404 в разборе | запрет unlink; статус superseded |
| Новая рубрика не в ALLOWED_SLUGS | `needs_code` + Telegram, не silent drop |
| LLM summaries отстают от PDF | catalog/viewer работают без summary; очередь видна на вкладке |
| Меню МО расползается | ровно **один** новый пункт, без подгрупп |
| Mac crawl (geo/SSL) | job только GCE, как Gemini |
| Owner env ≠ pavel | тот же self-heal, что у `.env.mis` |

---

## 8. Порядок внедрения (PR)

1. **PR1** - crawler + diff + status JSON + тесты (без UI, без cron).
2. **PR2** - incremental apply + reload corpus + GCE script/cron.
3. **PR3** - вкладка МО + API + канон меню 7 пунктов.
4. **Catch-up** на GCE после merge PR2 (координатор, не task-агент).
5. Primary deploy: `deploy_to_gce.sh` + smoke вкладки на `protocol.kravira.by`.

Одна безопасная следующая команда после согласования плана:

```bash
scripts/ops/git_task_start.sh minzdrav-kp-daily-sync --pc=pc1 \
  --branch=cursor/minzdrav-kp-daily-sync-v1-pc1
```
