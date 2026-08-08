# МО: prior-usage диагноза у врача / специальности (фундамент оценок) v1

Дата: 2026-08-08  
Статус: active  
Связано: `2026-08-08-mo-icd-name-match-v2.md` (потребитель), `mo_backend._doctor_breakdown` (peer specialty×chapter), `mo_case_document.load_prior_clinical` (prior **пациента**, не врача).  
Источник: владелец - при анализе учитывать, встречался ли этот ID диагноза ранее у **этого врача** или у **этой специальности**, и на этой базе строить **одно** МО-замечание / контекст, от которого зависят и другие оценки.

---

## 1. Продуктовое правило

| Вопрос | Смысл для методиста |
|--|--|
| Код/нозология уже была у этого врача? | «Привычный» диагноз vs первый раз / редкий |
| Код/нозология типична для специальности? | Ожидаемый case-mix vs вне профиля |
| Одно МО | Не плодить отдельные findings «не у врача» + «не у специальности» - **одна** ось с уровнями |

**ID** в этом плане = идентификатор нозологии случая:

1. **Primary:** `diagnosis_code` / `mkb_code_main` (нормализованный МКБ, напр. `N30.0`).
2. **Fallback (без кода):** стабильный ключ формулировки - `name_key` = нормализованный текст Dx через `clinical_text_similarity.normalize_for_match` (тот же слой, что name_only).
3. **Не путать** с `visit_id` / `patient_id` / `mis_id` - это не «ID диагноза».

Ось называется **prior Dx usage** (`mo_prior_dx_usage_v1`). Она **не** заменяет:

- справочник МКБ (directory / name_only);
- клиническую согласованность Dx ↔ жалобы/план;
- peer score specialty×chapter (`_doctor_breakdown`).

Она даёт **контекст привычности**, чтобы остальные оценки были точнее (строже для «нового» кода у врача, мягче для habitual при мелких опечатках и т.п.).

---

## 2. Одно МО: уровни и finding

Один код finding (shadow → потом primary): `B_dx_prior_usage`.

| Уровень `usage_tier` | Условие (черновик порогов) | Severity (shadow) | Смысл |
|--|--|--|--|
| `habitual_doctor` | count у врача ≥ 3 за lookback (напр. 180д), исключая текущий визит | - (passed / info) | врач уже ставил |
| `seen_doctor` | count у врача 1-2 | P3 info | редко, но был |
| `novel_doctor_known_specialty` | у врача 0, у специальности ≥ P50 или count ≥ N_spec | P3 | новый для врача, обычен в специальности |
| `novel_both` | у врача 0 и у специальности редко / 0 | P2 | новый для врача и редкий для профиля |
| `unknown` | нет doctor_key / specialty / кода и name_key | - | не штрафовать |

**Одно** замечание в UI, detail_ru собирает оба факта, например:

> Код N30.0: у врача ранее 12 раз за 180д; в специальности «урология» - частый (p80).

или

> Код M60: у врача ранее не встречался; в специальности «терапия» - редкий (&lt; p10). Проверьте обоснованность.

Не делать два finding `B_dx_prior_doctor` + `B_dx_prior_specialty`.

---

## 3. Фундамент: один контекст для всех оценок

```text
mo_prior_dx_context (lookup, без PHI в логах)
  keys: doctor_key | doctor_id_hash, specialty, diagnosis_code | name_key
  lookback_days, exclude_mis_id
  → {
      doctor_prior_n, specialty_prior_n, specialty_share,
      specialty_percentile, usage_tier, lookback_days,
      id_kind: "icd"|"name_key"
    }
         │
         ├─ finding B_dx_prior_usage          (одно МО)
         ├─ mo_icd_name_match                 (фаза C: пороги)
         ├─ mo_icd_directory_eval             (вес text_mismatch)
         ├─ mo_clinical_section_align (D)     (строгость support)
         ├─ mo_concordance / LLM judge        (prior как feature)
         └─ risk-gate / queue priority        (novel_both выше)
```

### 3.1 Как потребители используют контекст (не дублируя finding)

| Потребитель | Правило (черновик) |
|--|--|
| **name_only** | `habitual_doctor` + слабая опечатка → не поднимать выше review; `novel_both` + weak name → легче fail |
| **directory v1** | `novel_both` усиливает `B_icd_dir_text_mismatch`; habitual не снимает unknown code |
| **section-align / concordance** | при `novel_both` требовать более высокий support жалоб/статуса; при habitual - не ослаблять P0 safety |
| **LLM action-judge** | передавать в prompt только tier + counts (без FIO/PHI): «код новый для врача, редкий в специальности» |
| **очередь методиста** | boost priority для `novel_both` при низком overall |
| **peer breakdown** | не смешивать: chapter-peer = ожидаемый **балл**; prior-usage = ожидаемость **кода** |

Контекст считается **один раз** на case (в `kz_deep_eval` / case detail) и кладётся в `case["_mo_prior_dx"]` или отдельный блок ответа `prior_dx_usage`, чтобы все оси читали одно и то же.

---

## 4. Как реализовать (данные и код)

### 4.1 Источник правды по истории

Предпочтительно warehouse SQLite (`MO_ANALYTICS_DB` / `warehouse/mo_analytics.sqlite`):

- Уже есть: `fact_mo_case(doctor_key, specialty, diagnosis_code, visit_date, mis_id, …)`, индексы по `doctor_key`, `specialty`.
- `doctor_key` = hash FIO (как сейчас); для стабильности MIS лучше дополнительно хранить `doctor_id` (или hash) при upsert - **фаза A схемы**.
- Specialty: `doctor_specialization` / `specialty` (как в dim).

Fallback без полного warehouse: сканировать `secure_cases/YYYY/MM/mo_*.csv` за lookback (медленнее; только backfill/rebuild).

**Не** ходить в живую MariaDB MIS на каждый case detail с Mac/GCE app - только склад МО.

### 4.2 Агрегат (чтобы lookup был O(1))

Новая таблица (или materialized view в sqlite):

```sql
fact_mo_dx_prior (
  doctor_key TEXT,
  specialty TEXT,
  diagnosis_code TEXT,   -- или name_key при id_kind=name
  id_kind TEXT,          -- 'icd' | 'name_key'
  window_end DATE,       -- день пересчёта / партиция
  doctor_n INTEGER,
  specialty_n INTEGER,
  specialty_cases INTEGER,
  PRIMARY KEY (doctor_key, specialty, diagnosis_code, id_kind, window_end)
)
```

Пересчёт: nightly / после `upsert_warehouse` за день (incremental: обновить counts для кодов дня).  
Lookback: скользящее окно 180д (env `MO_PRIOR_DX_LOOKBACK_DAYS`).

Альтернатива v0 без таблицы: SQL на лету

```sql
SELECT COUNT(*) FROM fact_mo_case
 WHERE doctor_key=? AND diagnosis_code=?
   AND visit_date BETWEEN ? AND ? AND mis_id <> ?
```

плюс аналогичный count по `specialty`. Для case detail на день ок; для batch score дня - лучше агрегат или кэш в памяти на процесс.

### 4.3 Модуль

`clinical_knowledge/mo_prior_dx_usage.py`:

- `build_prior_dx_context(case, *, as_of_date, db=…) -> dict`
- `evaluate_mo_prior_dx_usage(case) -> list[finding]`  # одно finding max
- `merge_prior_dx_into_findings(...)`
- флаги: `MO_PRIOR_DX_USAGE=1`, `MO_PRIOR_DX_IN_PRIMARY=0`

Резолв ключей:

1. `doctor_key` из case / `doctor_key_for(doctor_fio)`; иначе hash `doctor_id`.
2. `specialty` нормализованная строка.
3. ICD из `resolve_icd_codes_from_mo` → main; иначе `name_key` из clinical_diagnosis.

Логи/telemetry: только `doctor_key`, specialty label, code/name_key, counts, tier - **без** FIO, patient_id, текста Dx.

### 4.4 Точки встраивания

1. `kz_deep_eval.evaluate_case` - посчитать context → shadow finding → передать в name_match / concordance.
2. `rag_server` case detail live merge (рядом с directory / name_match).
3. Позже: feature в LLM judge payload; priority в action queue.

---

## 5. Фазы

| # | Шаг | Статус |
|--|--|--|
| A0 | Зафиксировать ID = ICD (+ name_key fallback); lookback; tier-таблица | сделано (этот план) |
| A1 | При upsert_warehouse писать `doctor_id`/`doctor_id_hash` (если ещё нет) + индекс `(doctor_key, diagnosis_code, visit_date)` | дальше |
| A2 | `mo_prior_dx_usage.py`: context + одно shadow finding + тесты на фикстурах | дальше |
| A3 | Wire `kz_deep_eval` + case detail; RU label | дальше |
| B1 | Потребитель name_only: пороги от `usage_tier` | дальше (после merge name-match v2) |
| B2 | Потребитель concordance / section-align | дальше |
| B3 | LLM judge feature + queue boost `novel_both` | дальше |
| C1 | Калибровка порогов count/percentile на складе GCE; primary | дальше |
| C2 | Incremental `fact_mo_dx_prior` если live SQL медленный | дальше |

---

## 6. Метрики

| | Было | Цель |
|--|--|--|
| Учёт «код новый для врача» в МО | нет | shadow finding + context на case |
| Два разных finding doctor/specialty | риск | ровно одно `B_dx_prior_usage` |
| Потребители читают один context | 0 | name_only + concordance ≥1 feature |
| PHI в логах prior-lookup | n/a | 0 (только key/code/counts) |
| Latency case detail | baseline | +&lt;50ms p95 при индексе / кэше дня |

---

## 7. Риски

| Риск | Митигация |
|--|--|
| Смена FIO → новый `doctor_key` | писать стабильный `doctor_id` из MIS; lookup по id_hash с fallback на key |
| Редкая специальность / мало истории | tier `unknown` / не штрафовать при specialty_cases &lt; min_n |
| Код пустой | name_key; если и он пуст - unknown |
| Путаница с prior пациента | разные модули и названия; patient prior не использовать здесь |
| Двойной штраф novel + name fail | consumers меняют **вес**, не клонируют finding |
| Утечка FIO в evidence | в finding только code + counts + specialty label |

---

## 8. Тесты (когда дойдём до кода)

- habitual: врач с n=5 → нет P2, tier `habitual_doctor`
- novel_doctor_known_specialty: doctor 0, specialty high → один finding P3
- novel_both → один finding P2
- exclude текущего `mis_id` из count
- без doctor/specialty → unknown, findings []
- name_key path когда ICD пуст
- consumer stub: name_only получает тот же context object

---

## 9. Связь с name-match v2

План name_only остаётся про справочник. Prior-usage - **ортогональный фундамент**:

1. Сначала (или параллельно) context prior.
2. Name_only / section-align / judge **читают** context.
3. Методист видит одно МО про привычность кода + отдельные МО про формулировку и клинику.
