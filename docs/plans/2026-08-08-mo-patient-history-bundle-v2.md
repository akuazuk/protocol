# МО: сначала история пациента у врача и специальности, потом анализ (v2)

Дата: 2026-08-08  
Статус: active  
Преемник: `2026-08-08-mo-prior-dx-usage-baseline-v1.md` (там был только «код привычен врачу» - это **часть** шага 2, не фундамент).  
Связано: `mo_case_document.load_prior_clinical` (сейчас берёт **один** прошлый визит пациента, без разреза врач/специальность), name-match v2, concordance.

---

## Простыми словами: как всё будет

**Сейчас ошибка подхода**, если сразу считать «название похоже на МКБ» или «код новый», не собрав контекст пациента.

**Правильный порядок:**

1. **Собрать историю пациента** до текущего визита.
2. **Разложить её на две полки** (и третью «прочее»):
   - визиты **к этому же врачу**;
   - визиты **к другим врачам этой же специальности**.
3. **Склеить в один объект** `patient_history_bundle` - это фундамент.
4. Из бандла сделать **одно понятное МО** про историю (не десять разных).
5. **Только потом** включать анализаторы (название↔справочник, клиника, LLM) - они **читают бандл**, а не лезут в CSV сами.

```text
Пациент + текущий визит
        │
        ▼
┌───────────────────────────────────────┐
│  ШАГ 1. Собрать историю (фундамент)   │
│  patient_history_bundle               │
│                                       │
│  • same_doctor[]   - к этому врачу    │
│  • same_specialty[]- к другим этой    │
│                      специальности    │
│  • other[]         - остальные        │
│  • summary         - счётчики, коды   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  ШАГ 2. Одно МО по истории            │
│  «что уже было у этого врача /        │
│   у специальности по этому пациенту»  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  ШАГ 3. Анализаторы (читают бандл)    │
│  • тот же диагноз уже ставили?        │
│  • название ↔ справочник МКБ          │
│  • Dx ↔ жалобы / план                 │
│  • LLM-судья / очередь методиста      │
└───────────────────────────────────────┘
```

Без шага 1 шаги 2-3 будут неточными и каждый раз будут заново искать файлы.

---

## 0. Точно: как МО «объединяются» и где хранится история

### 0.1 Что значит «объединить» (и чего это НЕ значит)

| Не так | Так |
|--|--|
| Склеить все старые замечания в одно finding текущего визита | У каждого визита свои МО (`fact_mo_finding` как сейчас) |
| Отдельная SQL-таблица на каждый `patient_id` | **Одна** общая таблица визитов + ключ пациента |
| При оценке заново читать все CSV месяца | При оценке: `SELECT` по `patient_key` → собрать бандл в памяти |

**Объединение** = на момент оценки текущего визита **прочитать** историю пациента из склада, разложить по полкам (этот врач / другие специальности), получить один объект `patient_history_bundle` и из него:

1. одно историческое МО для текущего случая;
2. вход для остальных анализаторов.

Старые визиты **не пересчитываются** и их findings не переписываются. Меняется только оценка **текущего** визита с учётом прошлого.

```text
Склад (постоянно):
  визит_1 → свои scores + findings
  визит_2 → свои scores + findings
  визит_N (текущий) → считается сейчас

При оценке визит_N:
  история = все визиты пациента с date < N
  бандл = история разложенная по полкам
  МО_N = f(текст_N, бандл)     ← здесь «объединение»
```

### 0.2 Отдельная таблица на каждого пациента? Нет

Плохо:

```text
patient_12345_history
patient_67890_history
... десятки/сотни тысяч таблиц
```

Почему нельзя:

- нельзя нормально индексировать и обновлять при дневном upsert;
- нельзя сделать один SQL «все пациенты специальности»;
- `patient_id` в имени таблицы = утечка идентификатора;
- миграции и бэкапы ломаются.

**Правильно - общие таблицы, строки различаются ключом пациента.**

### 0.3 Целевая схема склада (две таблицы, не N)

Сейчас в `fact_mo_case` **нет** `patient_id` / `patient_key` - это главный пробел.

**A. Лента визитов (источник правды по истории)** - расширяем существующий `fact_mo_case` или добавляем узкую таблицу:

```sql
-- вариант: колонки в fact_mo_case (предпочтительно)
ALTER ... ADD patient_key TEXT;   -- hash(patient_id), не сырой id
ALTER ... ADD doctor_id TEXT;     -- стабильный MIS id врача (если есть)

CREATE INDEX idx_case_patient_date
  ON fact_mo_case(patient_key, visit_date);
CREATE INDEX idx_case_patient_doctor
  ON fact_mo_case(patient_key, doctor_key, visit_date);
CREATE INDEX idx_case_patient_specialty
  ON fact_mo_case(patient_key, specialty, visit_date);
```

Каждая строка = один документ/случай МО (как сейчас `mis_id`).  
Findings по-прежнему в `fact_mo_finding(mis_id, …)`.

**B. Опциональный кэш бандла (не обязательно в v1 кода)** - одна строка на пациента (или на пациента+день пересчёта), не таблица на пациента:

```sql
fact_mo_patient_history_cache (
  patient_key TEXT PRIMARY KEY,
  lookback_days INTEGER NOT NULL,
  as_of_date TEXT NOT NULL,          -- до какой даты собран
  summary_json TEXT NOT NULL,        -- counts, codes_by_doctor, codes_by_specialty
  visit_index_json TEXT NOT NULL,    -- список {mis_id, date, doctor_key, specialty, diagnosis_code, overall_pct}
  updated_at TEXT NOT NULL
)
```

Кэш обновляется при `upsert_warehouse` дня (для затронутых `patient_key`) или лениво при первом запросе case detail.

Тексты жалоб/плана в кэш **не обязаны** лежать целиком: для тяжёлых анализаторов при необходимости дочитываем secure row по `mis_id` из индекса.

### 0.4 Что происходит при оценке (псевдокод)

```text
evaluate(current_case):
  patient_key = hash(current_case.patient_id)   # наружу не отдаём
  rows = SELECT mis_id, visit_date, doctor_key, doctor_id, specialty,
                diagnosis_code, overall_pct
         FROM fact_mo_case
         WHERE patient_key = ?
           AND visit_date < current_case.visit_date
           AND visit_date >= current_case.visit_date - lookback
         ORDER BY visit_date

  bundle.same_doctor    = [r for r in rows if same doctor]
  bundle.same_specialty = [r for r in rows if same specialty and other doctor]
  bundle.other          = [r for r in rows if else]
  bundle.summary        = aggregate(bundle)

  history_finding = one_mo_from(bundle, current_case.diagnosis_code)  # 0 или 1

  # остальные анализаторы только читают bundle.summary (+ при нужде тексты)
  name_only = ...
  concordance = ...
  return findings including history_finding
```

Если кэш есть и `as_of_date` свежий - вместо SELECT по ленте читаем `summary_json` / `visit_index_json`.

### 0.5 Как «вся история» попадает в таблицу

Не отдельный ручной ввод, а побочный эффект дневного пайплайна:

1. Пришёл день → score → `upsert_warehouse` пишет/обновляет строки в `fact_mo_case` (+ `patient_key`).
2. Findings дня - в `fact_mo_finding`.
3. (Опционально) инвалидация/пересчёт `fact_mo_patient_history_cache` для patient_key этого дня.

История = накопленные строки склада за месяцы, не отдельный «архив на пациента».

### 0.6 Итог решения по хранению

| Вопрос | Ответ |
|--|--|
| Таблица на каждый patient_id? | **Нет** |
| Где вся история? | Строки в общей `fact_mo_case` (+ findings) с `patient_key` |
| Нужна ли ещё таблица? | Позже кэш `fact_mo_patient_history_cache` - **одна строка на patient_key** |
| Когда объединяем МО? | Только в момент оценки текущего визита, в памяти → бандл |
| Переписываем старые МО? | Нет |

---

## 1. Что такое «объединить МО по истории»

Не «склеить все замечания в одну строку», а:

> Построить **единый контекст пациента** относительно текущего врача и его специальности, и уже из него:
> - показать методисту **одно** историческое МО;
> - дать всем остальным оценкам один и тот же вход.

| Полка | Кто в визитах | Зачем |
|--|--|--|
| `same_doctor` | тот же `doctor_id` / `doctor_key` | динамика ведения у «своего» врача |
| `same_specialty` | та же специальность, **другой** врач | что уже делали коллеги профиля по этому пациенту |
| `other` | прочие специальности | фон (редко нужен в UI, полезен анализаторам) |

Текущий визит в бандл **не входит** (только lookback).

---

## 2. Что лежит в каждом прошлом визите (минимальный набор)

Без лишнего PHI в логах; в бандле для анализа - только нужные поля.

| Поле | Зачем |
|--|--|
| `visit_date`, `mis_id` / `visit_id` | порядок, исключение дублей |
| `doctor_id` / `doctor_key`, `specialty` | полка same_doctor / same_specialty |
| `diagnosis_code`, короткий `diagnosis_text` | тот же ID / формулировка |
| ключевые слоты: жалобы, статус, план (укороченно) | потом Dx↔клиника |
| если уже посчитано: `overall_pct`, топ finding codes | «МО раньше» по этому визиту |

Итоговый `summary` бандла (то, что видят анализаторы чаще всего):

```text
summary:
  n_same_doctor
  n_same_specialty
  codes_same_doctor:     {N30.0: 3, ...}
  codes_same_specialty:  {N30.0: 5, ...}
  last_same_doctor_date
  last_same_specialty_date
  current_code_seen_by_doctor: true|false
  current_code_seen_in_specialty: true|false
```

Именно `summary` - контракт для шага 3. Полные тексты визитов - по запросу анализатора, не тащить везде.

---

## 3. Одно МО по истории (шаг 2)

Один finding, например `B_patient_history_context` (название кода уточним в коде).

Примеры текста для методиста:

- «У этого врача по пациенту уже 4 визита за 180д; код N30.0 ставился 2 раза. У других урологов - ещё 1 визит с тем же кодом.»
- «К этому врачу пациент впервые; у других терапевтов за 180д - 2 визита с другими кодами (J06.9, I10).»
- «Истории недостаточно (нет patient_id / мало визитов) - историческое МО не штрафует.»

Не делать отдельные findings «нет у врача» и «нет у специальности» - уровни внутри одного.

Черновик уровней:

| tier | Смысл |
|--|--|
| `known_to_doctor` | текущий код уже был у этого врача по пациенту |
| `known_in_specialty_only` | у врача по пациенту кода не было, у коллег специальности был |
| `new_for_profile` | ни у врача, ни у специальности по пациенту такого кода не было |
| `first_contact` | визитов same_doctor = 0 (первый контакт с врачом) |
| `insufficient` | нет данных - не штрафовать |

---

## 4. Как реализовать правильно (техника)

### 4.1 Откуда брать визиты

Источник: склад МО (`secure_cases` / warehouse), **не** живой запрос в MariaDB на каждый клик.

Сейчас `load_prior_clinical` умеет только **один** прошлый визит пациента за N дней - этого мало. Нужна новая функция:

```text
build_patient_history_bundle(
  patient_id,          # только внутри, наружу не отдавать
  as_of_date,          # дата текущего визита
  doctor_id|doctor_key,
  specialty,
  exclude_ids,         # текущий mis_id / visit_id
  lookback_days=180,
) -> PatientHistoryBundle
```

Алгоритм:

1. Найти все визиты пациента с `date < as_of_date` за lookback (warehouse SQL предпочтительнее обхода всех CSV).
2. Для каждого визита определить полку: same_doctor / same_specialty / other.
3. Посчитать `summary`.
4. Вернуть бандл; `patient_id` в API/UI/логи **не** класть.

Стабильный врач: предпочитать `doctor_id` из МИС; `doctor_key` (hash FIO) - запасной.

### 4.2 Где считать один раз

| Когда | Где |
|--|--|
| Batch score дня | в `kz_deep_eval` / pipeline до findings: положить бандл в case context |
| Разбор случая (UI) | case detail: тот же builder, кэш на запрос |
| Night LLM / judge | читает готовый `summary` из case, не собирает историю сам |

Правило: **бандл считается один раз на case**, дальше все оси только читают.

### 4.3 Модуль (целевой)

`clinical_knowledge/mo_patient_history_bundle.py`

- `build_patient_history_bundle(...)`
- `history_summary_for_analyzers(bundle) -> dict`  # только summary
- `evaluate_history_mo(bundle, current_code) -> list[finding]`  # 0 или 1 finding
- `attach_bundle_to_case(case) -> case`  # `case["_patient_history"] = ...`

Флаги: `MO_PATIENT_HISTORY_BUNDLE=1`, `MO_PATIENT_HISTORY_IN_PRIMARY=0`.

### 4.4 Склад (чтобы не тормозить)

Минимум для быстрого lookup:

```sql
-- уже почти есть в fact_mo_case:
-- patient нужен! сейчас в fact_mo_case patient_id может отсутствовать
-- фаза A: гарантировать patient_id_hash или join к secure row при upsert
INDEX (patient_key, visit_date)
INDEX (patient_key, doctor_key, visit_date)
INDEX (patient_key, specialty, visit_date)
```

Если `patient_id` в warehouse нет - фаза A1: добавить `patient_key` (hash) при `upsert_warehouse`, без сырого id в отчётах.

---

## 5. Что потом анализируют поверх бандла (шаг 3)

Все анализаторы **не** ходят в историю сами.

| Анализатор | Что берёт из бандла | Что делает |
|--|--|--|
| Историческое МО (шаг 2) | summary + current code | одно замечание |
| Привычность кода у врача вообще* | можно позже отдельным lookup по всем пациентам врача | не путать с историей **этого** пациента |
| Name-only МКБ | `current_code_seen_*`, тексты prior Dx | мягче/строже порог опечатки |
| Dx ↔ жалобы/план | prior слоты same_doctor / same_specialty | «продолжает линию» vs «внезапный новый Dx» |
| Concordance / LLM | summary tier + короткие prior snippets | feature в промпт без ФИО/patient_id |
| Очередь методиста | `new_for_profile` / `first_contact` | выше приоритет |

\*«Код вообще часто ставит этот врач (любым пациентам)» - отдельный сигнал case-mix; в v2 он **вторичен**. Сначала - история **пациента**.

---

## 6. Фазы работ

| # | Шаг | Статус |
|--|--|--|
| 0 | Зафиксировать порядок: бандл → одно МО → анализаторы | сделано |
| 0b | Зафиксировать хранение: общие таблицы + patient_key, не table-per-patient | сделано (§0) |
| A1 | `patient_key` (+ опц. `doctor_id`) в `fact_mo_case` + индексы; писать при upsert | дальше |
| A2 | `build_patient_history_bundle` из SQL ленты + тесты | дальше |
| A3 | Одно shadow МО `B_patient_history_context` + UI label | дальше |
| A4 | Wire в deep_eval и case detail (бандл один раз) | дальше |
| A5 | (Опц.) `fact_mo_patient_history_cache` - 1 строка на patient_key | дальше |
| B1 | Name-only читает summary (веса) | дальше |
| B2 | Section-align / concordance читают prior слоты | дальше |
| B3 | LLM judge + очередь по tier | дальше |
| C | Калибровка lookback/порогов на GCE | дальше |

---

## 7. Метрики

| | Было | Цель |
|--|--|--|
| Prior пациента | 1 прошлый визит, без полок врач/спец | бандл с same_doctor / same_specialty |
| Исторических МО | разрозненно / нет | ровно одно по истории |
| Анализаторы сами ищут CSV | да (частично) | нет - только бандл |
| patient_id в логах/UI | риск | 0 (только hash внутри) |

---

## 8. Риски

| Риск | Митигация |
|--|--|
| Путаница «история пациента» vs «код привычен врачу по всем» | разные модули; в UI разные подписи |
| Смена ФИО врача | ключ `doctor_id` |
| Нет patient_id в строке | tier `insufficient`, не штрафовать |
| Тяжёлый обход CSV | warehouse + индексы; кэш бандла на case |
| Утечка PHI | в finding - даты/коды/counts; тексты - укороченно только в разборе |

---

## 9. Тесты (когда будет код)

- Пациент с 3 визитами к врачу A и 2 к врачу B той же специальности → полки верные.
- Текущий визит не попадает в бандл.
- Код текущего визита уже был у A → tier `known_to_doctor`, одно finding max.
- Код только у коллеги специальности → `known_in_specialty_only`.
- Нет patient_id → `insufficient`, findings [].
- Name-only mock получает тот же `summary` object.
