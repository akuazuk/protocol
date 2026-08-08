# МО: сверка названия диагноза со справочником МКБ (name_only) v2

Дата: 2026-08-08  
Статус: active  
Преемник v1: `2026-08-07-mo-dx-text-suggest-icd-directory-eval-v1.md` (v1 остаётся для code+title; v2 добавляет ось без кодов).  
Источник: владелец - обновить RU-справочник; сравнивать **названия** диагноза с `title_ru` (опечатки, несовпадение формулировок), не равенство кодов.

Связанные:

- `clinical_knowledge/mo_icd_directory_eval.py` - v1 (код в справочнике + token overlap с title)
- `icd_mkb.py` / `data/icd_reference/icd10_ru_mkb10su.json`
- `mo_concordance_findings` - клиническая согласованность (отдельная ось)
- будущий контур: диагноз ↔ жалобы / анамнез / обследования / рекомендации / лечение
- фундамент привычности кода: `2026-08-08-mo-prior-dx-usage-baseline-v1.md` (врач / специальность → одно МО + context для других осей)

---

## 1. Продуктовое правило

| Ось | Вход | Что сравниваем | Коды МКБ |
|--|--|--|--|
| **v1 directory** | Dx ± код | код есть в справочнике; overlap текста с `ru_title(code)` | да |
| **v2 name_only (это)** | свободный текст Dx | формулировка ↔ лучшее `title_ru` в справочнике | **нет** - коды из текста вырезаются перед match |
| **Prior usage (отдельный план)** | ICD / name_key + врач + специальность | был ли ID ранее у врача / в специальности → **одно** МО | ID = код (или name_key) |
| **Клиническая согласованность (дальше)** | Dx + слоты КЗ | формулировка Dx ↔ жалобы, анамнез, статус, обследования, план | нет |

Подбор КП по-прежнему **только по тексту диагноза** (v1 product rule). Name_only - дополнительная оценка качества формулировки относительно справочника (опечатки, «не та нозология», пустой/мусорный Dx).

---

## 2. Зачем общий слой текста сейчас

Сейчас строим **одну** нормализацию и similarity для Dx ↔ `title_ru`. Тот же контракт потом кормит сравнение Dx с клиническими слотами без переписывания матчера:

```text
mo_prior_dx_context          ← фундамент (отдельный план)
  usage_tier / counts по врачу и специальности
       │
       ├─ одно МО B_dx_prior_usage
       └─ feature для name_only / concordance / judge / очереди

clinical_text_similarity
  normalize_for_match / strip_icd_codes / token_jaccard / fuzz_ratio / combined_score
       │
       ├─ mo_icd_name_match (фаза B): Dx → каталог title_ru
       └─ mo_clinical_section_align (фаза D, позже): Dx → complaints|anamnesis|exams|recs|treatment
```

Правила переиспользования:

1. Не смешивать «есть в справочнике МКБ», «клиника поддерживает диагноз» и «код привычен врачу» в один finding.
2. Пороги калибровать отдельно (справочник vs клиника), API similarity общий; prior-usage меняет **веса**, не клонирует замечания.
3. Не гонять LLM на bulk match - только lex + token + SequenceMatcher (stdlib).
4. В evidence не тащить PHI сверх короткого snippet (как в других shadow findings).

---

## 3. Фазы

### A. Справочник МКБ RU (обновление артефакта) - сделано

- Пересобрать JSON из `mkb10_ru_mkb10su.xlsx` (`scripts/export_icd_ru_from_xlsx.py`).
- Добавить `icd10_ru_mkb10su.meta.json`: count, sha256, source xlsx, exported_at_utc.
- README: как обновить справочник; что title часто содержит префикс `CODE - …` и для name match префикс снимается.

Метрики A:

| | Было | Цель |
|--|--|--|
| Строк в JSON | 15616 | ≥ текущего после re-export |
| meta.json | нет | есть, sha совпадает с JSON |
| Расхождение xlsx→json | unknown | 0 (idempotent re-export) |

### B. Name-only match (shadow) - сделано (код в PR; калибровка - C)

Helper: `clinical_knowledge/mo_icd_name_match.py` + shared `clinical_text_similarity.py`.

Алгоритм:

1. Собрать текст Dx из слотов (как directory eval).
2. `strip_icd_codes` + `normalize_for_match`.
3. Кандидаты: `suggest_icd_from_russian` (top N) + при слабом hit - token-prune по каталогу.
4. Для каждого кандидата: score по **очищенному** `title_ru` (без ведущего кода).
5. Verdict: `ok` / `review` / `fail` по `combined_score` (token Jaccard + difflib ratio).

Findings (shadow default):

| code | Когда |
|--|--|
| `B_icd_name_no_match` | нет кандидата выше review |
| `B_icd_name_weak_match` | лучший hit в зоне review (опечатка / неточная формулировка) |

Флаги: `MO_ICD_NAME_MATCH=1` (default on), `MO_ICD_NAME_IN_PRIMARY=0` (default off).

Пороги (стартовые, калибровка на gold):

| | combined |
|--|--|
| ok | ≥ 0.42 |
| review | ≥ 0.28 |
| fail | &lt; 0.28 |

Метрики B:

| | Было | Цель |
|--|--|--|
| Ось name_only в case detail | нет | shadow findings |
| Ложных fail на «Острый цистит» | n/a | 0 на эталоне |
| Ложных ok на мусоре | n/a | 0 на эталоне |

### C. Калибровка / primary (дальше)

- Прогон на gold / выборке дней GCE.
- Подкрутка порогов; при стабильности - опция primary.
- UI-подпись оси (не смешивать с directory v1).

### D. Клиническая секция-alignment (дальше, дизайн)

Использовать тот же `combined_score(Dx, section_text)` для слотов:

| section | ключи case |
|--|--|
| complaints | `complaints` |
| anamnesis | `anamnesis_doctor`, `anamnesis_auto` |
| exams | данные обследований / `exam_*` |
| recommendations | `exam_recommendations` |
| treatment | `treatment_recommendations` |

Выход: профиль `support_by_section` + findings вроде `B_dx_weak_support_complaints` (отдельно от МКБ). Concordance v1 может постепенно переехать на этот слой вместо ad-hoc токенов.

### E. Prior-usage врача/специальности (дизайн вынесен)

Полный план: `2026-08-08-mo-prior-dx-usage-baseline-v1.md`.

Кратко: ID = МКБ (fallback name_key); lookup в warehouse `fact_mo_case` по `doctor_key` и `specialty`; **одно** finding `B_dx_prior_usage` с tier; тот же context читают name_only (C), section-align (D), LLM judge и очередь. Не путать с prior пациента (`load_prior_clinical`).

---

## 4. Что изменено в проде (после merge+deploy)

| Компонент | Изменение |
|--|--|
| `data/icd_reference/*.meta.json` | манифест сборки справочника |
| `clinical_text_similarity.py` | общий normalize/score |
| `mo_icd_name_match.py` | name_only eval + merge shadow |
| `rag_server` / `kz_deep_eval` | подключение merge рядом с directory v1 |
| UI labels | `mo_icd_name_match_v1` |

Deploy: GCE `deploy/gcp-app/deploy_to_gce.sh` после merge; Render - backup.

---

## 5. Шаги

| # | Шаг | Статус |
|--|--|--|
| A1 | Re-export JSON + meta.json + README | сделано |
| B1 | `clinical_text_similarity.py` (+ `score_against_sections` stub) | сделано |
| B2 | `mo_icd_name_match.py` + тесты | сделано |
| B3 | Wire rag_server + kz_deep_eval + labels | сделано |
| C1 | Калибровка на gold / день | дальше |
| D1 | section-align helper на том же similarity | дальше |
| E1 | Prior-usage фундамент (отдельный план) | дальше |

---

## 6. Риски

| Риск | Митигация |
|--|--|
| Title в JSON с префиксом кода → ложный boost | strip ведущего `CODE -` и всех кодов из Dx |
| 15k full scan медленно | suggest top-N + token prune; кэш индекса |
| Двойные findings v1+v2 | разные `code` / `source_ref`; UI подписи разные |
| Primary раньше калибровки | default shadow |
| PHI в evidence | truncate ≤400, как directory |

---

## 7. Тесты

- `tests/test_clinical_text_similarity.py`
- `tests/test_mo_icd_name_match.py` - цистит ok, мусор fail, опечатка review/ok, код в тексте не решает match
