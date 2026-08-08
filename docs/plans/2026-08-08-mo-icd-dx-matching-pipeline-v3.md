# МО: полный пайплайн сверки диагноза со справочником МКБ (v3)

Дата: 2026-08-08  
Статус: active  
Преемник: `2026-08-08-mo-icd-name-match-v2.md` (шаги 1-4 сделаны; v3 = калибровка,
усиление матчинга, единый оркестратор, серая зона LLM).  
Связанные (не смешивать оси):

| План | Роль |
|--|--|
| `2026-08-07-mo-dx-text-suggest-icd-directory-eval-v1.md` | КП только по тексту Dx; directory helper v1 |
| `2026-08-08-mo-icd-name-match-v2.md` | name_only + чип + full-text fallback Dx (шаги 1-4) |
| `2026-08-06-mo-icd-full-document-search-v1.md` | код искать по всему МО; soft-fill warehouse - отдельно |
| `2026-08-08-mo-patient-history-bundle-v2.md` | история пациента **после** стабилизации МКБ на визите |
| `2026-08-05-mo-llm-action-queue-judge-v1.md` | LLM-судья; сюда же серая зона Dx↔МКБ |
| `clinical_knowledge/diagnosis_icd.py` | seed нозология→код (ОРВИ, ГБ, …) - вшить в пайплайн |

Источник требования: владелец - exact match формулировки с title МКБ почти всегда врёт;
нужна устойчивая сверка кода и «своими словами» написанного диагноза.

---

## 1. Контекст: что уже в проде (не переделывать)

Production GCE: https://protocol.kravira.by  
База: `origin/main` после #49 / #50 / #51 (`BUILD_VERSION` см. handoff
`docs/reports/2026-08-08-handoff-mo-icd-steps-1-4.md`).

| Компонент | Что делает |
|--|--|
| `data/icd_reference/icd10_ru_mkb10su.json` + `.meta.json` | RU-справочник ~15.6k `{code, title_ru}` |
| `icd_mkb._canonicalize_icd_like_token` | `,`/`/`/`-` → `.`, кириллица К→K |
| `mo_icd_resolve.resolve_icd_codes_from_mo` | коды по всему документу |
| `mo_icd_resolve.resolve_diagnosis_text_from_mo` | слоты → «Диагноз: …» → snippet у кода |
| `mo_icd_directory_eval` | код ∈ справочник + token coverage с `ru_title` |
| `mo_icd_name_match` + `clinical_text_similarity` | текст Dx ↔ title_ru (Jaccard + coverage + fuzz) |
| `mo_icd_visit_status` | чип: `МКБ ✓` / `нет Dx` / `не в МКБ` / `слабо МКБ` |
| Protocol Suggest v3 | **не** ищет КП по МКБ |

Shadow default: `MO_ICD_DIR_IN_PRIMARY=0`, `MO_ICD_NAME_IN_PRIMARY=0`.

### 1.1 Почему exact match не подходит (зафиксировать)

| Ситуация | Exact | Нужно |
|--|--|--|
| `K29,3` / `К29.3` | fail | canonicalize → `K29.3` ∈ JSON |
| `K293` без точки | сейчас fail | compact → `K29.3` |
| «хронический гастрит» vs «Хронический поверхностный гастрит» | fail | coverage / combined_score |
| «ОРВИ» vs длинный title J06.9 | fail | словарь аббревиатур / seed |
| «гастрит» + код M60 | «код ок» врёт | text↔title mismatch |
| Несколько Dx через `\|` / перенос | одна строка шум | матчить каждый элемент списка |

---

## 2. Продуктовые оси (не смешивать findings)

```text
A. Наличие Dx в МО          B_dx_absent
B. Код: формат + ∈ справочник   B_icd_invalid / B_icd_dir_code_unknown
C. Текст ↔ справочник (name)    B_icd_name_*
D. Текст ↔ выбранный код        B_icd_dir_text_mismatch
E. Код текста ↔ код МИС         B_icd_mismatch_mis
F. Клиника поддерживает Dx      B_dx_* / concordance (не МКБ)
G. Подбор КП                    protocol_suggest - только текст Dx
```

Чип таблицы дня агрегирует **только A-D** (visit status). E и F - отдельные
замечания. История пациента (лента) - отдельный план, читает уже сохранённые чипы.

---

## 3. Целевой алгоритм на один визит

Единый оркестратор (новый модуль, тонкая обёртка - не дублировать scorers):

`clinical_knowledge/mo_icd_match_pipeline.py` → `evaluate_mo_icd_match(case) -> dict`

```text
case
  │
  ├─1─ resolve_diagnosis_text_from_mo → diag_text, source
  ├─2─ resolve_icd_codes_from_mo → main, all[]
  ├─3─ canonicalize каждый код (+ compact без точки, см. §4.1)
  │
  ├─4─ expand_clinical_aliases(diag_text)  # ОРВИ→…, хр.→хронический
  │       aliases не заменяют evidence; идут параллельным query
  │
  ├─5─ если нет текста и нет кодов → B_dx_absent / chip missing_dx
  │
  ├─6─ directory_eval(text_or_alias, codes)     # оси B,D
  ├─7─ name_match(text_or_alias)                # ось C
  ├─8─ mis_agreement(codes vs mkb_code_mis)     # ось E (починить значения)
  │
  ├─9─ merge verdict (см. §3.1)
  └─10 серая зона review → опционально LLM judge (§5) только offline/GCE
```

Выход (для case detail / warehouse):

```json
{
  "engine": "mo_icd_match_pipeline_v3",
  "diag_text": "...",
  "diag_source": "slots|label_line|near_code|empty",
  "codes": ["K29.3"],
  "alias_expanded": "острая инфекция верхних дыхательных путей",
  "directory": { "verdict", "text_rubric_fit", "code_checks", "findings" },
  "name_only": { "verdict", "name_fit", "best_code", "best_title_ru", "findings" },
  "mis_agreement": "match|partial|mismatch|unknown|skip",
  "chip": { "status", "label_ru", "title_ru" },
  "pipeline_verdict": "ok|review|fail|skip",
  "score_pct": 0-100,
  "needs_llm_review": false,
  "findings": []
}
```

### 3.1 Слияние вердиктов (приоритет)

| Приоритет | Условие | chip / pipeline_verdict |
|--|--|--|
| 1 | нет текста и нет кодов | `missing_dx` / fail |
| 2 | код unknown **и** name_only fail | `not_in_directory` / fail |
| 3 | код unknown, но name_only ok/review | `not_in_directory` (код) + name finding; chip `not_in_directory` |
| 4 | код ok, name fail, fit &lt; review | `not_in_directory` или `weak_name` - калибровать |
| 5 | directory text_mismatch или name weak | `weak_name` / review |
| 6 | иначе | `ok` |

Правило для методиста: **красный** только когда и код, и название не
сопоставились; **жёлтый** - слабое согласие или противоречие текст↔код;
**зелёный** - хотя бы одна ось уверенно ok и нет жёсткого contradiction.

Contradiction (отдельный finding, не терять):

- `B_icd_dir_text_mismatch`: код ∈ справочник, но title не про тот текст
  (пример: «гастрит» + M60). Даже при name_only hit на K29.\* - оставлять
  mismatch по **указанному** коду.

---

## 4. Нюансы реализации (чеклист)

### 4.1 Коды: опечатки и формы

| Кейс | Действие | Где |
|--|--|--|
| `K29,3` `K29/3` `K29-3` | уже canonicalize | `icd_mkb` |
| `К29.3` (кириллица) | уже | `icd_mkb` |
| `K293` / `K2930` | **новое**: letter+digits → insert `.` после 2 цифр; принять только если ∈ JSON | `_canonicalize` или `mo_icd_resolve._normalize_code` |
| `k29.3` регистр | upper | уже |
| Лишний хвост `K29.3.` | strip trailing `.` | уже частично |
| Несколько кодов | `all[]`; main из слота диагноза | уже |
| Код только вне графы Dx | presence ok; для text_fit брать near_code snippet | уже #50 |
| Плейсхолдер «см. МКБ» | не считать кодом | фильтр в extract |

Тесты: `K293`→`K29.3` hit; `K9999` compact не создавать ложный код; кириллица+запятая.

### 4.2 Текст: нормализация до similarity

Уже в `clinical_text_similarity`: lower, ё→е, strip кодов, strip `CODE -` из title,
пунктуация, Jaccard / coverage / fuzz.

Добавить (осторожно, с тестами):

| Нормализация | Пример | Заметка |
|--|--|--|
| Сокращения с точкой | `хр.` → `хронический`, `о.`/`остр.` → `острый` | словарь, не regex «все точки» |
| Ё/й стабильность | уже ё→е | |
| Множественные пробелы / переносы MIS `::` | схлопнуть | при сборке diag_text |
| Латиница в RU тексте | `GERD` → через alias | §4.3 |
| Слишком короткий текст (&lt;3) | skip name_only | уже |

Не стеммить агрессивно («гастрит»≠«гастриты» ок через fuzz; не резать до 3 букв
для всех токенов - шум).

### 4.3 Словарь клинических алиасов (главный пробел)

Источник правды для seed уже есть: `diagnosis_icd._DIAGNOSIS_ICD_SEED`
(орви, орз, гэрб, хобл, гипертоническая болезнь, …). Сейчас он **не** кормит
name_match / directory_eval как расширенный query.

Сделать слой:

`clinical_knowledge/mo_icd_aliases.py`

```text
expand(diag_text) -> {
  original,
  expanded_phrases: ["острая инфекция верхних дыхательных путей"],
  seed_codes: ["J06.9"],   # мягкая подсказка, не факт
  match_method: "alias_seed"
}
```

Правила:

1. Match по подстроке / целому токену (длинные стемы раньше коротких - как в seed).
2. Expanded phrase **добавляется** к запросу suggest/name_match; original остаётся
   в evidence.
3. `seed_codes` не выставляют chip ok сами по себе: только boost кандидатов /
   clinical_hint. Иначе «ОРВИ» без сверки со справочником станет ложным ok.
4. Файл данных: начать с выноса seed + таблица аббревиатур
   (`data/icd_reference/dx_aliases_ru.json`); правки без релиза кода - по возможности.
5. Стоп-слова и слабые прилагательные - как в `icd_mkb._RU_STOP` / `_RU_WEAK_ADJ`.

Минимальный v3 набор аббревиатур (из живых прогонов):

| Alias | Expanded / seed |
|--|--|
| ОРВИ, ОРЗ | J06.9 + полная формулировка title |
| ГБ, АГ | I10 / гипертензивная болезнь |
| ИБС | I25.9 |
| ХОБЛ | J44.9 |
| ГЭРБ / GERD | K21.9 |
| ИМВП | N39.0 |
| СД2 / СД 2 тип | E11.9 |
| хр. / хрон. | хронический |
| о. / остр. | острый |

### 4.4 Similarity и пороги

Стартовые (из v2; **калибровать** на gold, не менять вслепую):

| Ось | ok | review |
|--|--|--|
| name_only `combined` | ≥ 0.42 | ≥ 0.28 |
| directory `text_rubric_fit` (coverage title) | ≥ 0.35 | ≥ 0.25 |
| lexicon `DIR_HIT_SCORE_MIN` | ≥ 0.12 | - |

Известные ложные зоны (эталоны в тесты):

| Пример | Ожидание v3 |
|--|--|
| «острый цистит» + N30.0 | ok |
| «хронический гастрит» + K29.3 | ok (fit ~0.67) |
| «гастрит» + M60 | review + text_mismatch |
| «ОРВИ» + J06.9 | ok **после** alias (сейчас review) |
| «гипертоническая болезнь 2 ст» + I11.9 | review или ok после alias/coverage на I10/I11 |
| «боль в животе» + R10.4 | review/ok после улучшения coverage («живот» vs «живота») - морфология лёгкая или fuzz |
| мусор «ааа ббб» | fail |
| пустые слоты, код в статусе + near snippet | не `missing_dx` |

Морфология: не подключать pymorphy в P0. Сначала alias + fuzz; если «живот/живота»
ломает метрику на калибровке - добавить лёгкий stem (отрезание 1-2 окончаний только
для токенов ≥6) под флагом.

### 4.5 Несколько диагнозов

MIS: `diagnosis_list` через `|`, main index в `##[3]`.

| Правило | |
|--|--|
| Main | оценивать обязательно (chip) |
| Прочие | findings с `linked_fields` / evidence; не затирать chip main |
| Строка «K29.3. Гастрит \| N30.0 Цистит» | split → два независимых match |
| Suggest КП | по **main** clinical text (как сейчас) |

### 4.6 MIS agreement (баг + семантика)

Экспорт пишет `match|partial|mismatch|unknown`.  
`kz_deep_eval` B3 ждёт `0/1/true/false` → `B_icd_mismatch_mis` почти никогда не
срабатывает.

Фикс в v3:

1. Принимать оба словаря значений.
2. `partial` (совпал 3-значный stem) → не finding mismatch; опционально info/shadow.
3. `mismatch` → `B_icd_mismatch_mis`.
4. Не путать с directory: MIS - про два источника кода, не про title_ru.

### 4.7 Warehouse soft-fill (координация с full-doc планом)

Из `2026-08-06-mo-icd-full-document-search-v1.md` P3:

- `mkb_code_main` пуст, но full-doc нашёл код → soft-fill для KPI/UI.
- **Не** подменять код для `mkb_code_agreement` (сверка с `mis_diagnos` остаётся
  на исходных полях экспорта).
- Помечать `mkb_code_main_source=slot|soft_fill_full_doc`.

Делать отдельным PR после оркестратора или сразу за ним - не блокировать калибровку.

### 4.8 Primary vs shadow

Порядок включения в балл:

1. Калибровка на ≥20 ручных + gold packs (3650612 / 3643304 / цистит-эталоны).
2. Метрики §6 в зелёной зоне.
3. Флаги: сначала `MO_ICD_NAME_IN_PRIMARY=1`, затем directory; или общий
   `MO_ICD_PIPELINE_IN_PRIMARY=1` после оркестратора.
4. Чип уже виден; primary меняет только влияние на overall score.

### 4.9 LLM только серая зона

Не гонять Gemini на каждый визит с Mac (правило `gemini-via-render` / GCE).

| Когда | Что |
|--|--|
| `pipeline_verdict == review` или contradiction текст↔код | опциональный judge |
| ok / явный fail (мусор, unknown code без name) | без LLM |

Контракт judge (JSON, коротко):

```json
{
  "agree": "yes|partial|no",
  "reason_ru": "≤160 chars",
  "suggested_code": "K29.3|null"
}
```

Вход: diag_text, code, ru_title(code), top-3 name candidates.  
Выход → shadow finding `B_icd_llm_review_*` или снятие weak при `yes` (под флагом).  
Запуск: night/action-queue на GCE (`deploy/gcp-llm/run_on_gce.sh`), не local Mac.

### 4.10 UI / копирайт

- Чип и KPI «МКБ / диагноз» уже есть - не дублировать.
- В разборе явно три подписи, если findings с разных осей:
  - «нет диагноза в МО»
  - «код / название не в справочнике»
  - «формулировка слабо совпадает с рубрикой»
- Не писать «подбор КП по МКБ».
- Evidence ≤400 символов, без лишнего PHI.

### 4.11 Производительность

- Не full-scan 15k на каждый визит: `suggest_icd_from_russian` top-N + prune.
- Alias expand - O(словарь).
- Кэш `ru_title(code)` уже в процессе загрузки JSON.
- Pipeline на день N визитов должен укладываться в текущий recompute budget
  (замерить на одном дне GCE до primary).

### 4.12 Что сознательно вне v3

| Тема | Куда |
|--|--|
| История пациента / «код привычен врачу» | `mo-patient-history-bundle-v2` |
| Dx ↔ жалобы/план (клиника) | phase D в name-match v2 / concordance |
| WHO ICD-API Docker | deep-eval backlog; RU JSON остаётся SoT |
| Embeddings title_ru | только если калибровка lex+alias не вытягивает recall; отдельный spike |
| Смена правила «КП без МКБ» | запрещено без запроса владельца |

---

## 5. Фазы работ

### Фаза 0 - контракт (docs) - этот файл

- [x] Зафиксировать оси A-G и целевой pipeline
- [x] Список нюансов §4
- [x] Индекс `docs/plans/README.md`: v3 active; v2 → archived (преемник v3)

### Фаза 1 - оркестратор + фикс MIS (P0)

- [x] `mo_icd_match_pipeline.py` + unit-тесты на эталонах §4.4
- [x] Wire в `kz_deep_eval` / case detail рядом с текущими merge
  (сначала pipeline вызывает существующие helpers, не ломая флаги)
- [x] Починить `mkb_code_agreement` ∈ {match, partial, mismatch, unknown}
- [x] Чип питается из `pipeline.chip` (один источник)

### Фаза 2 - коды compact + aliases (P0/P1)

- [x] Compact ICD без точки → только если код ∈ JSON
- [x] `mo_icd_aliases` + JSON; связать seed `diagnosis_icd`
- [x] ОРВИ / ГБ / хр. эталоны зелёные
- [x] Тесты на ложный boost (alias не ставит ok без справочника)

### Фаза 3 - калибровка → primary (P1)

- [x] Прогон выборки дня с GCE warehouse (без печати PHI в отчёт)
- [x] Таблица confusion: chip vs мнение методиста (≥20 кейсов)
- [x] Подкрутка порогов в одном месте (constants / env)
- [x] `MO_ICD_*_IN_PRIMARY` или общий pipeline flag
- [x] Обновить handoff + метрики в этом плане

Решение 2026-08-08 (см. `docs/reports/2026-08-08-mo-icd-pipeline-calibration.md`):

- Эталоны 22/22 chip accuracy=1.0; `not_in_directory` P/R=1.0/1.0.
- День 2026-08-04 (n=200): `not_in_directory` share≈0.365; `missing_dx`≈0.005.
- **Включаем** `MO_ICD_NAME_IN_PRIMARY=1` (deploy GCE default).
- **Не включаем** `MO_ICD_DIR_IN_PRIMARY` / `MO_ICD_PIPELINE_IN_PRIMARY` (ждём ручных labels дня).
- Пороги не трогали (дефолты v3); вынесены в `mo_icd_thresholds.py` + env override.
- Критичный fix deploy: RU JSON `data/icd_reference/icd10_ru_mkb10su.json` в Docker image
  (раньше на GCE `ru_valid_codes=0` → ложный not_in_directory на всём дне).

### Фаза 4 - LLM серая зона (P2)

- [x] Контракт judge + интеграция в action-queue / night на GCE
- [x] Флаг `MO_ICD_LLM_REVIEW=0` default off
- [x] Не включать в overall, пока нет ≥30 размеченных review-кейсов

Решение 2026-08-08:

- Модуль `mo_icd_llm_review.py` + batch `scripts/run_mo_icd_llm_review.py`.
- Night: `mo_llm_range_runner.sh` вызывает batch только при `MO_ICD_LLM_REVIEW=1`.
- Findings `B_icd_llm_review_{yes,partial,no}` всегда shadow; clear-weak только
  при `MO_ICD_LLM_CLEAR_WEAK=1` (default off).
- Живой Gemini - только GCE (`deploy/gcp-llm/run_on_gce.sh`), не Mac.
- Включить batch на день: на VM/container `MO_ICD_LLM_REVIEW=1` +
  `MO_ICD_LLM_REVIEW_LIMIT=50` перед range-runner.

### Фаза 5 - soft-fill warehouse (P2, координация)

- [x] P3 из full-document plan; source-маркер; не ломать MIS agreement
- [x] Recompute day smoke на GCE

Решение 2026-08-08:

- `soft_fill_mkb_for_warehouse` в `mo_icd_resolve.py`.
- `upsert_warehouse` пишет `diagnosis_code` + `mkb_code_main_source`
  (`slot|soft_fill_full_doc|empty`) + `mkb_code_main_slot`.
- `mkb_code_agreement` / CSV слот **не** переписываются.
- UI: `mo_backend` отдаёт source/slot на списке и в case detail.

### Фаза 6 - (опционально) лёгкая морфология / embeddings spike

- [x] Лёгкий stem под `MO_ICD_LIGHT_STEM` (default off); без pymorphy
- [x] Spike-отчёт `docs/reports/2026-08-08-mo-icd-light-stem-spike.md`
- [x] Embeddings spike не нужен: stem закрыл «живот/живота» на фикстуре
- Не в primary по умолчанию - включать флаг на GCE после smoke

---

## 6. Метрики

| Метрика | Было (после #50) | Цель v3 |
|--|--|--|
| Exact-string зависимость | нет (уже) | сохранить |
| «ОРВИ»+J06.9 → ok | review / weak | ok (alias) |
| «гастрит»+M60 → contradiction finding | review | review + явный text_mismatch |
| `K293` → directory hit | fail unknown | ok если K29.3 в JSON |
| `B_icd_mismatch_mis` на export `mismatch` | не стреляет | стреляет |
| Ложный `missing_dx` при near_code тексте | снижен #50 | 0 на gold |
| Precision chip `not_in_directory` (эталоны ≥20) | unknown | ≥ 0.85 → **1.0** (2026-08-08) |
| Recall «реально нет в МКБ» | unknown | ≥ 0.80 → **1.0** (эталоны) |
| Primary без калибровки | выкл | фаза 3: **NAME=1**, DIR/pipeline=0 |
| LLM на 100% визитов | нет | нет (только review) |
| Suggest reasons `icd_fit` | 0 | 0 |

Отчёт калибровки: `docs/reports/YYYY-MM-DD-mo-icd-pipeline-calibration.md`
(агрегаты, без сырых ФИО/текстов КЗ).

---

## 7. Что изменится в проде (после merge+deploy фаз)

| Компонент | Изменение |
|--|--|
| `mo_icd_match_pipeline.py` | единый выход + chip |
| `icd_mkb` / resolve | compact codes |
| `mo_icd_aliases` + JSON | аббревиатуры |
| `kz_deep_eval` B3 | match/mismatch vocabulary |
| flags env | pipeline primary / llm review |
| UI | без новой колонки; те же чипы, меньше ложных weak на ОРВИ |
| Suggest / КП | без изменений правила |

Deploy: GCE canonical (`deploy/gcp-app/deploy_to_gce.sh`); Render backup.  
LLM phases - только GCE llm runner, не Mac.

---

## 8. Риски

| Риск | Митигация |
|--|--|
| Alias раздувает ложные ok | seed_codes только boost; вердикт от score справочника |
| Compact `A12`→мусор | принимать только после `is_code_in_ru_reference` |
| Двойной штраф directory+name | pipeline merge; один chip; findings с разными code |
| Primary убивает overall врачей | shadow → калибровка → flag |
| PHI в калибровочном отчёте | только mis_id / агрегаты |
| Путаница с историей пациента | история не в этом плане |
| Регресс suggest ICD-first | тесты `icd_fit` reasons == 0 |

---

## 9. Тесты (минимум)

| Файл | Фокус |
|--|--|
| `tests/test_mo_icd_match_pipeline.py` | эталоны §4.4 + merge chip |
| `tests/test_icd_compact_canonicalize.py` | K293 / отказ мусора |
| `tests/test_mo_icd_aliases.py` | ОРВИ, хр., порядок стемов |
| дополнить `test_mo_icd_name_match` / directory | alias-expanded query |
| `tests/test_kz_deep_eval_mis_agreement.py` | mismatch/partial/match |

---

## 10. Definition of Done v3

1. План в индексе; v2 помечен archived с преемником v3.
2. Оркестратор + compact codes + aliases + фикс MIS agreement в main.
3. Эталоны §4.4 зелёные в CI.
4. Калибровочный отчёт ≥20 кейсов; решение по primary зафиксировано в плане.
5. LLM review - либо за флагом off, либо shadow-only с контрактом.
6. Suggest по-прежнему без МКБ; история пациента не блокирует и не подменяется.
7. Handoff в `docs/reports/` + `/api/version` на GCE после deploy.

---

## 11. Безопасная следующая команда

```bash
# после commit/push этого плана - реализация фазы 1 в отдельной task-ветке:
scripts/ops/git_task_start.sh mo-icd-match-pipeline-p0 --pc=pc1 \
  --branch=cursor/mo-icd-match-pipeline-p0-agent1-pc1
```

Не включать `MO_ICD_*_IN_PRIMARY` до фазы 3.
