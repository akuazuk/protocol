# Daily КП МЗ: только новые PDF + точный подбор и оценка (v2)

Дата: 2026-08-13  
Статус: **active**  
Преемник: `2026-08-13-minzdrav-kp-daily-sync-v1.md` (там был полный apply «всего подряд»; здесь - узкий прогон и качество match/score)  
Связанные: `2026-08-08-mo-icd-first-kp-suggest-v1.md`,
`2026-08-08-mo-analytics-mz-sheet-layers-v2.md`,
`2026-08-07-by-home-gcp-llm-split-v1.md`.

---

## 1. Ответ на вопрос прогона

**Да. На GCE каждый день обрабатываем только новые и изменённые PDF.**  
Не пересобираем чанки 345 старых протоколов. Не гоняем OCR/LLM по всему корпусу.

Чанки нужны не «для RAG ради RAG», а потому что **оценка плана уже читает их**:

```text
код/текст Dx
  → protocol_cards + catalog (какой КП)
  → protocol_icd_profiles из чанков/таблиц (что в КП обязательно)
  → clinical KP hit
  → alignment обследования/лечения
  → зона «План по протоколу» (0 / 0.5 / 1)
```

Оформление (№127) КП не использует. Если clinical KP нет - план не штрафуем «не по протоколу» (`n/a` / 0.5), поэтому **дыры в каталоге = слепые оценки**, а не ложный 0.

Первый catch-up: прогон **~61 missing PDF** (посты 04-06.2026 + ревматология КП1-32), не всего архива.

---

## 2. Что реально гонять на GCE (слои)

| Слой | Когда | Что | Блокирует night score? |
|--|--|--|--|
| 0. Crawl + download | daily 01:00 UTC | только added/updated URL | да, если 0 файлов - skip дальше |
| 1. Текст + чанки + таблицы | только changed PDF | `corpus_pipeline` `--changed-only`, merge в jsonl | да для этих path |
| 2. Индекс подбора | после слоя 1 | cards, `protocol_catalog.jsonl`, ICD profiles **merge по path** | **да** |
| 3. Доказательства плана | после слоя 1 | exams/drugs/follow-up из таблиц и typed chunks (без LLM) | **да** |
| 4. Summaries LLM | очередь | `kz_checklist` для grade/UI | **нет** (видима на вкладке как pending) |
| 5. RAG extras | опционально | lex shards / vector sidecar только changed paths | нет для МО |
| 6. Targeted re-score | если слой 2 сменил КП | вчера + визиты, чей МКБ пересекается с added/updated | да, узко; не весь месяц |

Не гоняем: полный `chunks.jsonl` rewrite, OCR всех PDF, LLM по stomatology-пачке, полный recompute июля.

Стоматология (94 PDF): чанки для поиска врача - да, если файл новый; **не** блокируют оценку clinical_visit.

---

## 3. Почему одного «скачать PDF» мало

Сейчас подбор и балл ломаются даже при наличии файла:

| Дыра | Эффект | Пример 13.08 |
|--|--|--|
| Нет 2026 ОКС / АГ / ТЭЛА в каталоге | suggest берёт КП 2017 или `kp_not_matched` | пост 28.04.2026 №44 vs «инфаркт 06.06.2017 №59» |
| Короткое имя файла, ICD только из filename | карточка без кодов → specialty-filler | `КП2_ревматоидный артрит.pdf` |
| Нет веса свежести в `match_score` | старый КП с тем же корнем МКБ побеждает | бронхит 2012 vs №64 2026 |
| Один КП в двух рубриках | дубль path, разный rank | ЗНО кишки в gastro и novoobrazovaniya |
| Rehab / algorithm как clinical | ложный hit, alignment по не той норме | реабилитация ЧМТ vs острый инсульт |
| LLM summary отстаёт | UI беднее, но **score МО уже может работать** с профилем из таблиц | не ждать Gemini |

v1 это не закрывал. v2 добавляет слой качества **на тех же новых чанках**.

---

## 4. Усовершенствования прогона (точность подбора и оценки)

Делать **только на added/updated path** (+ successor старого path).

### 4.1 Карточка из текста PDF, не из имени файла

- Заголовок: первые страницы / «Клинический протокол «…»».
- Пост: `ДД.ММ.ГГГГ №N` из текста, если в имени нет (ревматология КП1-32).
- Аудитория: маркеры в теле, не только `взр_нас` в filename.
- `protocol_kind`: `clinical` / `rehab` / `algorithm` / `admin`. Suggest и plan-zone берут только `clinical` (как ICD-first v5).

### 4.2 ICD и обязательства из нужных чанков

Не размазывать коды из приложений с лекарствами.

- `icd10_primary`: главы классификация / диагностика / критерии.
- Exams / treatment / follow-up: таблицы + chunk_type diagnostics/pharmacotherapy (уже есть `protocol_icd_profiles`).
- Merge профиля в индекс **по path**, без пересборки 396 профилей.

Это напрямую кормит `consult_alignment` и `_score_plan` (балл 1.0 только при clinical KP **и** alignment блока ≥60).

### 4.3 Граф замен и свежесть

- `superseded_by`: файл пропал с сайта **или** тот же нозологический ключ + более новый пост.
- Suggest: не брать superseded как top-1, если successor проходит ICD/audience.
- В `match_score` добавить небольшой вес **recency** (год/дата поста), не ломая ICD-first. Старый 2017 не должен бить 2026 при том же I21.
- Канонический path при дубле рубрик: один `canonical_path`, второй `alias`.

### 4.4 Оценка только по актуальному clinical KP

После apply:

1. Suggest reload cards+catalog+profiles.
2. Для вчерашнего inbound: re-score **только** случаев, у которых код/корень МКБ пересекается с ICD новых КП (или ранее `kp_not_matched` по той же рубрике).
3. В результат случая писать `kp_corpus_generation` (sha манифеста) - видно, по какой редакции считали.
4. Не пересчитывать зону оформления: она без КП.

### 4.5 Что не трогать в том же cron

Gemini summaries, полный vector index, MIS extract. Summaries - следующий слот / очередь на вкладке.

---

## 5. Метрики

| Метрика | Было | Цель |
|--|--|--|
| Night: PDF, по которым считаем чанки | 0 или «всё руками» | = числу added+updated за сутки (обычно 0-5; catch-up ~61 один раз) |
| Покрытие сайт ↔ корпус | 335/396 | ≥99% или явный superseded |
| I20-I25 / I10 / I26 top-1 | старые КП или нет hit | clinical КП постов 2026 №38/43/44/47 |
| Plan 1.0 без clinical KP | запрещено (уже в коде) | сохранить |
| Alignment exams с новым КП | пустой профиль → 0.5 «следующий этап» | профиль из таблиц в ту же ночь |
| Re-score объём | нет | только пересечение ICD, не весь месяц |
| Вкладка МО | нет | сверка + added/updated + pending summaries |

---

## 6. Архитектура night

```text
01:00 UTC  crawl+diff+download          (GCE, без MIS DSN)
           chunk+tables  --changed-only
           merge cards / catalog / icd_profiles
           successor + recency metadata
           reload app caches
01:40      retry если fail
02:00      night MIS extract+score      (уже новый catalog)
           если сегодня были КП с ICD: точечный re-score вчера
           по пересечению кодов
```

PDF SSOT: `/var/data/protocol_corpus`. Git - скрипты и тесты, не бинарники.

---

## 7. Вкладка «Протоколы МЗ»

Как в v1: 7-й пункт меню. Дополнительно колонки, без которых качество не проверить:

- обработано чанков / path за ночь (должно быть = added+updated, не 345);
- есть ли ICD-профиль и таблицы (готово к оценке);
- summaries pending (не блокер);
- сколько вчерашних МО пересчитано из-за новых КП.

---

## 8. Шаги внедрения

### A. Catch-up на GCE (разово)

- [ ] A1. Скачать missing 61, не unlink 10 старых.
- [ ] A2. Чанки+таблицы+cards+ICD **только этих path**.
- [ ] A3. Smoke: ОКС/АГ/ТЭЛА/ЗНО молочной железы открываются; suggest на фикстуре I21 / I10 даёт КП 2026.

### B. Daily incremental (код)

- [x] B1. Crawl/diff/download как v1 §B, тесты на фикстуре HTML.
- [x] B2. `run_pipeline --changed-only` + merge jsonl по `source_path`.
- [x] B3. Merge catalog + icd_profiles; запрет full rewrite.
- [x] B4. Заголовок/пост/kind/audience из текста PDF (тесты на фикстурах).

### C. Точность match/score

- [x] C1. `superseded_by` + recency в rank (тесты: 2017 vs 2026, тот же корень МКБ).
- [x] C2. Suggest не primary на rehab/admin/superseded.
- [x] C3. Canonical path при дубле рубрик.
- [~] C4. Targeted re-score по ICD overlap; поле `kp_corpus_generation`. (generation в status JSON; yesterday score идёт после 01:00 sync)
- [x] C5. Не ждать LLM summary для 1.0/0.5 плана, если профиль из таблиц уже есть.

### D. Cron + UI

- [x] D1. `night_kp_sync.sh` 01:00 UTC, owner `pavel`, Telegram fail.
- [x] D2. Вкладка МО + API (v1 §E + счётчики слоя 1-3).
- [x] D3. `/api/corpus-stats`: last sync, changed_n, pending_summaries.

### E. Хвост

- [ ] E1. LLM summaries очередь на GCE (не Mac).
- [ ] E2. Периодический PR `index.csv` без PDF.
- [ ] E3. Выборочный re-score 7/30 дней по запросу, не daily.

---

## 9. Риски

| Риск | Митигация |
|--|--|
| `--changed-only` затрёт весь chunks.jsonl | merge по path; тест «345 строк на месте, +1 новая» |
| Recency побьёт единственный точный старый КП | вес меньше ICD; recency только при сопоставимом icd_fit |
| Ложный successor (разные болезни, соседний № поста) | successor только при пересечении ICD roots **или** явном «в ред. поста» в имени/тексте |
| Catch-up 61 PDF не уложится в 01:00-02:00 | catch-up вручную днём, daily cap N файлов |
| SSL/вёрстка МЗ | фикстуры + алерт; не валить MIS night |

---

## 10. PR

1. **PR1** - crawl/diff + `--changed-only` merge + тесты (без UI).
2. **PR2** - title/ICD/kind/successor/recency + targeted re-score.
3. **PR3** - вкладка МО.
4. Catch-up на GCE после PR2. Deploy: `deploy_to_gce.sh` + smoke suggest I21/I10 на `protocol.kravira.by`.
