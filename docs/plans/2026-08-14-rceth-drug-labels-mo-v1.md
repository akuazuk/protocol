# Rceth ЛС: скачивание, обновление, разметка для МО Аналитики (v1)

Дата: 2026-08-14
Статус: **active**
Автор: агент + владелец (разбор Refbank / ОХЛП vs текущий drug/DDI слой)
Связанные:

- `2026-08-10-mo-eval-quality-followups-v2.md` - drug-norm, DDI, findings;
- `2026-08-11-mo-ddi-topical-demote-v1.md` - форма ЛС (гель vs системно);
- `2026-08-13-minzdrav-kp-daily-sync-v2.md` - пайплайн сверки КП на GCE;
- `2026-08-14-minzdrav-kp-sync-stats-v1.md` - **канон UI/аналитики сверки** (KPI, журнал, графики);
- `2026-08-05-mo-eval-smirnova-concordance-v1.md` - Dx/concordance **не** заменяется этим планом.

Источник: [Гос. реестр ЛС РБ (rceth Refbank)](https://www.rceth.by/Refbank/).

---

## 1. Контекст и цель

### Зачем

В МО Аналитике уже есть:

- нормализация бренда → INN (`drug_normalizer`);
- safety: DDInter, high-alert, STOPP, дубль НПВП;
- клинические протоколы МЗ (suggest / plan-score).

Нет локального **label-check** по официальной инструкции РБ:

- назначение вне показаний к выставленному Dx/ICD;
- противопоказания по возрасту / состоянию;
- возрастные рамки формы (особенно педиатрия).

Инструкции специалиста (`_s.pdf`) = ОХЛП / инструкция по Решению ЕЭК № 88.
Они усиливают **оценку назначений**, не заменяют проверку **правильности диагноза** (это КП + concordance).

### Цель v1

1. Скачать и версионировать корпус **действующих** ЛС: карточка + `_s.pdf`.
2. Разметить структурированно (identity + разделы 4.1 / 4.2 / 4.3).
3. Подготовить API/контекст для findings в МО (сначала shadow).
4. Настроить обновление без полного перекачивания каждый раз.

### Вне scope v1

- Реестр ИМН / медтехники (`minstr`) - ROI около нуля для ЛС/Dx.
- Массовая закачка `_p.pdf` (пациент) - опционально позже.
- Недействующие / прекращённые / приостановленные - только если позже понадобится исторический label.
- Автоштраф «любой off-label = Важно» без калибровки.
- Замена DDInter или КП МЗ.

---

## 2. Объём данных

| Что | Оценка | Комментарий |
|--|--|--|
| Действующие записи ЛС | ~7 300 | без unterm/annul/pause |
| Из них с инструкцией | ~50% → **~3.5-4 тыс.** | покрытие PDF в выдаче |
| PDF к загрузке v1 | **~4 000 × `_s.pdf`** | не 27k |
| HTML-карточки | ~4-7.3 тыс. | лёгкие; паспорт + журнал НД |
| Диск | ~8-15 ГБ | медиана PDF ~2.4 МБ |
| Время первого прогона | **1-3 ч** (+ запас 3-5 ч) | 1-2 req/s; NDfiles бывает 503 |
| Пилот | **50-100 `_s`** | частые INN из МО |

Почему не 27k: ~20k - ИМН; ещё ~4k - дубли `_p`. Для оценки назначений ЛС достаточно действующих `_s`.

Дыра, которую 27k не закрывает: ~половина действующих записей **без** PDF в реестре - для них остаются карточка (INN/ATC/форма) + DDInter + КП.

---

## 3. Где качать: предпочтительно GCE

**Канон: весь download / sync / parse job - на GCE (`protocol-app`). Mac не является контуром bulk.**

| | GCE (предпочтительно) | Mac (только отладка) |
|--|--|--|
| Bulk ~4k PDF, 1-5 ч | **да** - cron / systemd / screen | нет (сон, VPN, сеть) |
| Данные | `/var/data/rceth/` рядом с МО UI | локальные fixtures только |
| Live-статус в МО | job пишет `status.json` на том же volume | нет общего writer |
| Параллельные агенты | один GCE job-writer | не качать bulk с Mac параллельно |
| Доступ rceth `/NDfiles/` | обязательный smoke 5× `_s` перед bulk | unit-тесты на сохранённых PDF fixtures |

Правила:

1. **Пилот 50-100 и полный корпус** - оба на GCE (один pipeline, разные `--limit`).
2. Mac: разработка кода + pytest на fixtures из `tests/fixtures/rceth/` (3 карточки), без ночного bulk.
3. Если GCE не достучится до NDfiles - **не молча уходить на Mac bulk**. Handoff + retry/backoff; fallback (rsync с Mac / egress) только явным решением владельца.
4. Код в git; PDF только на GCE data volume.

Связь: `2026-08-07-by-home-gcp-llm-split-v1.md`, `2026-08-13-minzdrav-kp-daily-sync-v2.md`, `2026-08-14-minzdrav-kp-sync-stats-v1.md`.

---

## 4. Аналитика и онлайн-видимость процесса (как «Протоколы МЗ»)

Паттерн UI брать с вкладки **Протоколы МЗ** (`/api/methodist/mo/kp-sync` + `loadKpSync`) и live-job с полингом как у пересчёта жёсткости (`recompute_job.progress` каждые ~2 с).

### 4.1. Вкладка / блок в МО Аналитике

Рабочее имя: **«Инструкции ЛС (rceth)»** (отдельная page или подсекция рядом с «Протоколы МЗ»; не смешивать KPI КП и rceth).

| Блок | Содержание (зеркало kp-sync) |
|--|--|
| KPI | сайт Refbank reachable; записей в манифесте; с `_s`; скачано; parse_ok; inactive |
| Свежесть | дата/время последней сверки; статус job: `idle` / `running` / `done` / `error` |
| Live-баннер | если `running`: фаза, `done/total`, скорость, ETA, last_error (poll 2 с) |
| Журнал прогонов | таблица history: день, added/updated/failed/no_pdf/parse_fail |
| Диаграммы | поступления по прогонам (stack); доля parse_ok; топ ATC / fails |
| Таблицы | last-run changed; failed URLs (без ПДн); needs_human parse |

### 4.2. API и файлы статуса (GCE)

```text
/var/data/rceth/
  manifest.jsonl
  pdfs/...
  labels/...
  _sync/
    rceth_sync_YYYY-MM-DD.json     # итог прогона (как kp_sync_*.json)
    status.json                    # live: phase, progress, started_at, pid
    history/                       # опц. архив
```

- `GET /api/methodist/mo/rceth-sync` → `public_rceth_sync_payload()` (аналог `public_kp_sync_payload`).
- В `/api/corpus-stats` или hero - краткий `rceth_sync` snapshot (ok/running/stale).
- Job на GCE **атомарно** обновляет `status.json` на каждом N файлов (не реже чем раз в 10-15 с).
- UI: пока `status === running|queued` - poll; иначе показать last sync + history.

### 4.3. Что видно «онлайн», пока идёт скачивание

Минимум в баннере:

- фаза: `crawl` → `download` → `parse` → `index`;
- `downloaded / to_download`, `parsed / to_parse`;
- bytes или avg MB/s (опционально);
- текущий `reg_id` (не клинический текст);
- число retry/503;
- кнопка «обновить» + автоpoll.

Не показывать содержимое инструкций и не логировать trade-level PII сверх reg_id/INN.

---

## 5. Что менять в проде (целевое состояние)

Пока **данные + UI сверки + shadow findings**, без ломки SSOT зон/№55.

| Слой | Артефакт | Куда |
|--|--|--|
| Raw | PDF + HTML + manifest | `/var/data/rceth/` на GCE (**не** в git) |
| Sync UI | `_sync/status.json` + `rceth_sync_*.json` | API + вкладка МО |
| Identity | `brand → INN`, ATC, form, Rx | merge в `drug_normalizer` / sidecar JSON |
| Labels | structured JSON по `reg_id` | `/var/data/rceth/labels/` (+ опц. мелкий seed в repo) |
| Runtime | `load_rceth_label_ctx()` рядом с `load_drug_ctx()` | deep / medication_findings на GCE |
| Findings UI | текст finding + ред. инструкции | разбор случая; сначала shadow |

В git: код пайплайна, схемы, fixtures, фронт вкладки.
В git **не** класть тысячи PDF.

---

## 6. Метрики

| Метрика | Сейчас | Цель v1 |
|--|--|--|
| Действующие `_s` в манифесте | 0 | ≥ 90% от найденных ссылок в Refbank |
| Parse success 4.1+4.3 | 0 | ≥ 80% пилота; ≥ 70% полного корпуса |
| Identity: бренд РБ → INN (top МО) | частично ручной словарь | + покрытие частых белорусских торговых имён |
| False positive off-label на gold 30 МО | n/a | < 15% в shadow до primary |
| Влияние на Dx-зону | не должно | 0 изменений SSOT диагноза |
| Sync latency | n/a | weekly на GCE; changed PDF < 24h после обнаружения |
| Live UI во время job | нет | баннер `done/total` + phase, poll ≤ 3 с |
| История прогонов на вкладке | нет | ≥ 1 запись после первого GCE run |

---

## 7. Архитектура пайплайна

```text
Refbank search (действующие)
  → manifest.jsonl (reg_id, urls, status, nd_dates, inn, atc, form)
  → download _s.pdf + detail HTML (skip if sha256 unchanged)
  → text extract (pdf)
  → section parse (4.1 / 4.2 / 4.3 …)
  → label JSON + identity dictionary
  → (позже) mo_label_check → findings shadow → калибровка → primary
```

### Идентификаторы

- `reg_id` - id карточки Refbank (`21_04_3138`, `11349_24`, …).
- `pdf_url` - `/NDfiles/instr/{reg_id}_s.pdf`.
- Версия: `pdf_sha256` + дата НД / текст «Изменение в нормативной документации».

### Статусы записи (обязательно в manifest)

| Класс строки / фильтр | Смысл | В v1 корпусе |
|--|--|--|
| без класса | обычная / действующая | **да** |
| `unterm` | срок закончился | нет |
| `annul` | прекращено | нет |
| `pause` | приостановлено | нет |
| EAEU notactual / discountinued | служебные пометки | хранить флагами; не путать с «не действующий» |

---

## 8. Шаги

### A. Инфра и манифест (GCE-first)

- [x] Скрипт обхода Refbank: поиск `Start` по буквам / пагинация `FOpt.PageN` (`clinical_knowledge/rceth_sync/`, `scripts/rceth_sync_run.py`).
  Пагинация: `IsPostBack=true` + `QueryStringFind` (без этого сервер отдаёт ту же страницу).
- [x] Фильтр только действующие (не включать VUnTerm/VAn/VPause).
- [x] Сбор `manifest.jsonl` (поля reg_id/trade/inn/form/urls/status; ATC/nd - через detail enrich позже).
- [x] Preflight `/NDfiles/` (`rceth_sync_run.py preflight`) - не стартовать download при fail.
- [x] Throttle + retry/503 + resume (skip existing PDF sha).
- [x] Каркас `_sync/status.json` + `public_rceth_sync_payload` + unit tests.
- [x] GCE job-скрипт `deploy/gcp-app/rceth_sync_job.sh` (пилот `--limit`).
- [ ] Полный crawl всех букв на **GCE** + smoke 5× `_s` с VM (остаток A на GCE).

**Выход A:** код + тесты в PR; полный манифест/smoke - на GCE (следующий прогон).

### B. Пилот на GCE (50-100) + разметка

- [ ] Топ INN/брендов из реальных МО; `--limit 100` на GCE.
- [ ] Скачать `_s` + HTML в `/var/data/rceth/`.
- [ ] Извлечь текст PDF; распарсить 4.1 / 4.2 / 4.3.
- [ ] Ручная проверка 15-20 карточек.
- [x] Schema JSON (§9) + `label_parse` / `parse` CLI; fixtures `oxlp_*_sample.txt` для Mac pytest.
- [ ] Live `status.json` виден через API или файл на VM.

**Выход B:** schema + parse quality; go/no-go на полный корпус.

### C. Полный корпус действующих `_s` (GCE)

- [ ] Smoke с GCE: 5× `_s.pdf` OK.
- [ ] Bulk download **только на GCE** в `/var/data/rceth/` (missing/changed).
- [ ] HTML-карточки; отчёт downloaded / failed / no_pdf / parse_ok.
- [ ] Финальный `rceth_sync_YYYY-MM-DD.json` + history.
- [ ] Summary (counts, sha манифеста) в `docs/reports/` без ПДн.

**Выход C:** ~4k PDF на GCE + manifest + первый history row для UI.

### D. Identity-словарь для drug_normalizer

- [ ] Из карточек: trade_name (ru) → INN, ATC, form keywords.
- [ ] Merge с `_BRAND_TO_INN` без ложных override (регрессия meloxicam).
- [ ] Unit-тесты на 20 белорусских брендов из пилота.

**Выход D:** улучшение нормализации до label findings.

### E. Вкладка аналитики + онлайн-прогресс (обязательно до primary findings)

- [x] `clinical_knowledge/rceth_sync/status.py`: `load_latest`, `public_rceth_sync_payload`.
- [x] `GET /api/methodist/mo/rceth-sync` (+ краткий блок в corpus-stats).
- [x] Page в МО: KPI, freshness, **всегда видимый статус-баннер** (running / idle / interrupted), журнал прогонов (графики/fail-таблицы - после первого GCE прогона).
- [x] Poll 2 с по образцу scoring-strictness job.
- [x] Тесты payload + якоря HTML (`rceth-sync-*`), по аналогии с kp-sync.
- [x] Права methodist: page flag `rceth_sync` как у `kp_sync`.
- [x] Stale/dead-pid: `resolve_live_status` не держит UI в вечном `running` после смерти job.
- [x] Карточка «Замечания» явно не лог синка (findings = шаг F).

**Выход E:** методист видит и историю, и текущий download online.

### F. Label-check в МО (shadow)

- [ ] `load_rceth_label_ctx()` / lookup by INN+form.
- [ ] Findings shadow: `off_label_vs_dx`, `label_contraindication`, `age_outside_label`.
- [ ] В тексте: surface + INN + «инструкция rceth, ред. {date}».
- [ ] **Не** менять Dx-зону / protocol suggest scoring.

**Выход F:** shadow findings + калибровка 30 кейсов.

### G. Weekly sync job на GCE

- [ ] Cron: refresh manifest → diff → download/parse changed → write sync JSON.
- [ ] Inactive mark; алерт 503/parse_fail.
- [ ] Один writer; UI отражает running через `status.json`.

**Выход G:** runbook + cron.

### H. Primary findings (решение владельца)

- [ ] После precision shadow: whitelist / severity.
- [ ] Off-label по умолчанию не выше Умеренно, пока нет gold.
- [ ] Справка МО: label vs КП vs DDI.

---

## 9. Схема разметки (черновик)

```json
{
  "reg_id": "11349_24",
  "status": "active",
  "trade_name_ru": "ИБУПРОФЕН ДАНСОН",
  "inn": "ibuprofen",
  "atc": "M01AE01",
  "forms": ["suspension_oral"],
  "rx_otc": "otc",
  "term_from": "2024-10-23",
  "term_to": "2029-10-23",
  "nd_changes": ["изменение в ОХЛП … пр. №1043 от 12.09.2025"],
  "pdf_s": {
    "url": "/NDfiles/instr/11349_24_s.pdf",
    "sha256": "…",
    "bytes": 0
  },
  "sections": {
    "indications_4_1": ["…"],
    "posology_4_2": ["…"],
    "contraindications_4_3": ["…"],
    "warnings_4_4": [],
    "interactions_4_5": []
  },
  "parse": {
    "ok": true,
    "method": "heading_regex_v1",
    "needs_human": false
  }
}
```

Правила парсера v1: якоря «4.1», «Показания», «4.3», «Противопоказания» (ru); при провале - `needs_human`, finding не штрафует.

---

## 10. Связь с МО Аналитикой (как включать)

```text
treatment_recommendations
  → drug_normalizer (+ rceth identity)
  → существующий _axis_safety (DDI / NSAID / high-alert)
  → NEW label_check(inn, form, age, dx/icd)  [shadow]
  → findings в safety / clinical_concordance
diagnosis
  → КП + concordance (без rceth как судьи Dx)
```

Приоритет: **E (UI/live) параллельно с C** после B; findings **D → F(shadow) → H**; sync **G**.

---

## 11. Риски

| Риск | Митигация |
|--|--|
| NDfiles 503 / таймауты | preflight; resume; не считать fail = «нет инструкции» без retry |
| Off-label шум (часто в практике) | shadow; severity потолок; калибровка 30 МО |
| Ложный INN merge | регрессии drug_normalizer; confidence; не затирать ручной словарь вслепую |
| Путаница label vs КП | справка; разные коды findings; KP остаётся про «что надо при болезни» |
| PDF без чётких 4.x (старый формат) | needs_human; не primary |
| Юридически / ToS сайта | throttle; хранить для внутренней оценки; при возможности запрос к rceth на выгрузку |
| Раздувание git | PDF только на data volume |

---

## 12. Порядок работ (краткий)

1. A манифест + GCE smoke NDfiles.
2. B пилот 50-100 на GCE + schema + status.json.
3. E вкладка аналитики + live poll (можно каркас сразу после A/B).
4. C полный `_s` корпус на GCE (видно online).
5. D identity в normalizer.
6. F shadow findings.
7. G weekly sync.
8. H primary - после метрик §6.

---

## 13. Статус шагов

| Шаг | Статус |
|--|--|
| A Манифест (GCE) | in progress (код+тесты+пагинация; полный crawl на GCE) |
| B Пилот GCE | in progress (parser+fixtures; GCE job запускается) |
| C Полный корпус GCE | pending |
| D Identity | pending |
| E UI аналитика + live progress | done (каркас KPI/live/history; графики после первых прогонов) |
| F Shadow findings | pending |
| G Weekly sync GCE | pending |
| H Primary | pending (решение владельца) |

---

## 14. Координация с параллельными агентами

- Отдельная task-ветка `cursor/rceth-drug-labels-mo-v1-pc1` / свой worktree; не `main`, не чужие ветки.
- Общий файл `docs/plans/README.md` - частый конфликт: перед merge rebase на свежий `origin/main`.
- Не трогать параллельно: night GCE KP sync scripts, `drug_normalizer.py` до шага D (тогда узкий PR).
- Bulk download - один writer на `/var/data/rceth` (GCE job), не два агента одновременно.

---

## 15. Готовность и следующая команда

### Checklist «можно приступать к коду»

| Проверка | Статус |
|--|--|
| Scope зафиксирован (~4k `_s`, не ИМН) | да |
| Bulk предпочтительно на GCE | да (§3) |
| UI/аналитика + live progress в плане | да (§4, шаг E) |
| Параллельные агенты / worktree | да (PR #146, ветка отдельная) |
| PDF не в git | да |
| Dx не через rceth | да |
| План в индексе `docs/plans/README.md` | да |

**Шаг A (код) стартовал** (скелет `scripts/rceth/` + status writer + GCE smoke) после вашего «поехали» / merge или продолжения в этой же task-ветке.

```bash
cd /private/tmp/protocol-task-rceth-drug-labels-mo-pc1
# следующий PR-коммит: scripts/rceth/ manifest crawl + status.json
```

Первый кодовый deliverable: `scripts/rceth/` (manifest crawl + download resume + `_sync/status.json`) и fixtures 3 карточек (Фенибут, Ибупрофен Дансон, Нимесулид Фармлэнд).
