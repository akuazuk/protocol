# Задание для машины с доступом к БД: глубокая оценка качества КЗ и поиск ошибок диагностики/лечения

**Статус:** active
**Дата:** 2026-07-22
**Исполнитель:** машина/агент с доступом к MariaDB МИС (`~/CURSOR/sql_epam/.env`) и к интернету
**Родственные планы:** `2026-07-22-kz-scoring-methodology-v1.md` (оси A/B/C, gold, калибровка),
`2026-07-22-kz-data-separation-viz-v1.md` (kz_kind, гигиена данных, №55).

> Этот файл - самодостаточное ТЗ. Его нужно выполнить на компьютере, где есть креды БД
> (на текущей машине их нет). Все шаги, источники и скрипты описаны так, чтобы можно было
> выполнять по порядку. Данные пациентов (ПДн) не коммитим - см. раздел 9.

---

## 0. Цель

Поднять оценку консультативных заключений (КЗ) с эвристической шкалы до **клинически
валидной, объяснимой и защитимой перед регулятором** оценки, которая:

1. **правильно оценивает КЗ** по осям документирования, клинического соответствия и
   безопасности;
2. **находит потенциальные ошибки/неточности** в двух самых важных местах:
   - **определение болезни** (диагноз не следует из жалоб/анамнеза/осмотра; диагноз не
     кодируется по МКБ; пропущен «красный флаг»; диагноз не соответствует обследованию);
   - **лечение** (назначение не по протоколу МЗ РБ; вне Республиканского формуляра;
     опасные лекарственные взаимодействия; high-alert-препарат без дозы/контроля;
     противопоказание при диагнозе; доза вне диапазона);
3. **опирается на всё доступное**: клинические протоколы МЗ РБ, коды МКБ, Постановление
   №55, Республиканский формуляр ЛС, материалы сайта Минздрава РБ, и **мировой опыт**
   (методики, базы взаимодействий, гайдлайны) - см. раздел 2.

Оценка нужна и как **число/статус**, и как **список конкретных находок** (что не так, где,
ссылка на протокол/пункт №55/МКБ/формуляр) для методиста и врача.

---

## 1. Что уже есть в проекте (на этом строим, не начинаем с нуля)

- **Выгрузка МИС**: `scripts/export_mis_protocol_month.py` (парсит `mis_protocol.result` по
  `::`, тянет ФИО/специализацию из `mis_data`, классифицирует `kz_kind`). Правило БД -
  `.cursor/rules/mis-mariadb.mdc`.
- **L1-батч**: `scripts/run_mis_protocol_l1_batch.py` (детерминированная оценка + summary +
  worst_visits + reg55).
- **Движок оценки**: `clinical_knowledge/compliance_engine.py`, `consult_alignment.py`,
  `scoring.py` (веса `config/compliance_weights.yaml`), 3 оси + axes-режим
  (`config/axes_thresholds.yaml`, флаг `CONSULT_AXES_OVERALL`, пока OFF).
- **№55**: `data/regulations/mz_2021_55.json` + `clinical_knowledge/reg55_criteria.py`
  (`regulatory_compliance_pct`, дефекты P0-P3).
- **Протоколы МЗ**: `minzdrav_protocols/*` (скачанные PDF по специальностям),
  `data/protocol_summaries/json/*.json` (структурированные `ConditionSummary` с
  `kz_checklist`, `required_exams`, `treatment`, `red_flags`, `diagnostic_criteria`),
  rich-chunks + векторный индекс для RAG.
- **Синонимы**: `data/catalog/exam_drug_synonyms.json` + `clinical_knowledge/term_catalog.py`.
- **LLM-разбор одного КЗ**: `clinical_knowledge/mis_kz_quality.py::review_one_visit_full`.
- **Gold**: `scripts/build_kz_gold_sample.py`, `label_kz_gold_llm.py`,
  `calibrate_axes_thresholds.py`.
- **Регуляция-наблюдение**: `data/regulations/mz_2015_127.json` + `dispensary_regulations.py`.

---

## 2. Источники данных (что подключаем; мировой опыт + РБ)

### 2.1. Регуляторика и протоколы РБ (первичный источник истины)
| Источник | Что берём | Доступ |
|---|---|---|
| **Постановление МЗ РБ 21.05.2021 №55** | критерии/дефекты экспертизы качества | уже в `data/regulations/mz_2021_55.json` (case-level); полный PDF в корне |
| **Клинические протоколы МЗ РБ** | обязательные обследования, критерии диагноза, лечение по Dx | `minzdrav.gov.by/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/` (24 специальности); юр. тексты - `pravo.by` / `etalonline.by`; в проекте уже `minzdrav_protocols/` + `protocol_summaries/json` |
| **Республиканский формуляр ЛС** | перечень разрешённых ЛС + ATC-классификация | Постановление МЗ **27.03.2026 №22** (актуальный); PDF на `minzdrav.gov.by/upload/lcfiles/...`; структура по ATC (ВОЗ) |
| **Инструкция №127** (медосмотры) | требования к заключению, наблюдение по главам МКБ | уже в `data/regulations/mz_2015_127.json` |

### 2.2. Кодирование диагнозов (МКБ)
| Источник | Что берём | Доступ |
|---|---|---|
| **WHO ICD-API** (`id.who.int`) | валидация кода МКБ-10, название по коду, синонимы, глава | OAuth2 (регистрация на `icd.who.int/icdapi`); есть endpoints ICD-10; **можно развернуть локально в Docker** (offline, без отправки ПДн наружу) |

### 2.3. Лекарственная безопасность (мировые базы)
| Источник | Что берём | Доступ |
|---|---|---|
| **DDInter 2.0** | 302k пар взаимодействий ЛС: тяжесть, механизм, тактика; + drug-disease, дубли терапии | скачать CSV по ATC-группам: `ddinter2.scbdd.com/download/` (offline-датасет) |
| **ISMP High-Alert Medications** | список high-alert (антикоагулянты, инсулин, опиоиды, метотрексат...) | публичный список ISMP; курируемый seed в проекте |
| **ONC High Priority DDI list** + **CredibleMeds** | узкий список критичных DDI + риск QT/Torsades | публичные списки |
| **RxNorm** (NLM) | нормализация названий ЛС → RxCUI (для матча с DDInter/ATC) | `findRxcuiByString` (работает); RxNorm public domain |
| **openFDA drug label** | секция `drug_interactions`, дозирование (DailyMed) | free, 240 req/min; резерв к DDInter |

> NLM Drug Interaction API закрыт (янв. 2024). Основа - **офлайн DDInter 2.0** (скачиваемый
> CSV) + курируемые ISMP/ONC. Так пайплайн работает без отправки ПДн во внешние сервисы.

### 2.4. Методики оценки (на чём строим рубрику и грейдер)
- **PDQI-9 / PDSQI-9** (Physician/Provider Documentation Quality Instrument) - 9 атрибутов
  качества записи (актуальность, точность, полнота, полезность, организованность,
  понятность, сжатость, синтез, согласованность/«cited»), Likert/бинарно. Валидирован;
  для **коротких** консультативных заметок надёжность ниже - нужны якоря и обучение
  разметчиков. → ось A (документирование).
- **SaferDx Instrument** (Singh) - скрининг диагностических ошибок по 5 этапам
  диагностического процесса: (1) встреча врач-пациент (сбор/осмотр/назначение тестов),
  (2) выполнение и трактовка тестов, (3) наблюдение/отслеживание во времени,
  (4) факторы направления/субспециальности, (5) факторы пациента. → детектор «ошибка
  определения болезни».
- **MedCheckLLM** (guideline-in-the-loop) - извлечь Dx → подобрать гайдлайн → превратить
  в чек-лист → LLM сверяет запись с чек-листом → человек подтверждает. **Это ровно наша
  архитектура** (`kz_checklist` + RAG + `review_one_visit_full`). → грейдер.
- **LLM-as-judge best practices (2026)**: рубрика с явными уровнями; **атомарные бинарные
  пункты**; chain-of-thought; few-shot с одним concordant + одним discordant примером;
  human-in-the-loop для низкой уверенности. GPT-o3-mini/Claude достигают ICC ~0.82 с
  экспертом при валидной рубрике. → LLM НЕ финальный судья, а масштабируемый «второй ридер».
- **CRICO/ISMP/NICE**: «Big Three» диагностических ошибок (онко, сосудистые катастрофы,
  инфекции/сепсис) = наши red_flags; high-alert-препараты; safety-netting.

---

## 3. Целевая модель оценки КЗ (что считаем и что ищем)

Итог по каждому КЗ - структура:

```
KZ_evaluation = {
  scores: {
    documentation (ось A, PDQI-9-lite),
    clinical_concordance (ось B, протокол МЗ + МКБ),
    safety (ось C, red flags + препараты),
    regulatory_55 (reg55_criteria),
    overall (калиброванный, с risk-gate)
  },
  findings: [                       # ГЛАВНОЕ: конкретные находки
    {kind: "diagnosis_error"|"treatment_error"|"doc_defect"|"safety",
     severity: P0|P1|P2|P3,
     message, evidence(цитата из КЗ), source_ref(протокол/№55/МКБ/формуляр/DDInter),
     confidence, needs_human}
  ],
  status, percentile_in_specialty
}
```

### 3.1. Детектор ошибок ОПРЕДЕЛЕНИЯ БОЛЕЗНИ (приоритетно)
1. **Диагноз не обоснован**: нет связки жалобы+анамнез/осмотр → диагноз (SaferDx этап 1;
   №55 п. 4.2.10). Детерминированно (поля) + LLM-суждение о логической связи.
2. **Код МКБ невалиден/отсутствует/не соответствует тексту диагноза**: валидировать код
   через WHO ICD-API (локально); сверять название кода с текстом Dx (семантически).
3. **Диагноз ↔ обследование не бьются**: для Dx протокол требует обязательный метод, а его
   нет в КЗ (ось B, `required_exams` + `kz_checklist.must_have`).
4. **Пропущен red flag / «нельзя исключить» без маршрутизации** (Big Three) → P0.
5. **Диагноз-специфичные критерии не выполнены** (`diagnostic_criteria` протокола).
6. **Подозрительный/неполный диагноз** («?», «под вопросом») без плана дообследования → P0.

### 3.2. Детектор ошибок ЛЕЧЕНИЯ (приоритетно)
1. **Назначение не соответствует протоколу** по данному Dx (нет обязательной группы, или
   назначена группа вне протокола) - ось B, `treatment` протокола.
2. **Препарат вне Республиканского формуляра** РБ (флаг: не входит в формуляр → пометка,
   не всегда ошибка, но повод для внимания).
3. **Опасное лекарственное взаимодействие** между назначенными ЛС - DDInter 2.0
   (severity major/contraindicated) → P0/P1.
4. **High-alert-препарат без дозы/длительности/контроля** (ISMP) → P0.
5. **Противопоказание препарат ↔ диагноз/состояние** (drug-disease, DDInter DDSI).
6. **Доза вне диапазона / дубль терапии** (openFDA/DailyMed dosing; DDInter therapeutic
   duplication).

### 3.3. Risk-gate
Любой необработанный **P0** → cap итогового балла (45-55%) + `manual_review_required`.
P3 (формальные) не топят клинически хорошую запись.

---

## 4. Предварительные шаги на машине с БД (окружение и загрузки)

### 4.1. Проверить доступ к БД (правило `mis-mariadb.mdc`)
```bash
cd ~/CURSOR/sql_epam && python3 -c "
from pathlib import Path; import os; from dotenv import load_dotenv
from sqlalchemy import create_engine, text
load_dotenv(Path('.env')); pw=os.environ['KRAVIRA_DB_PASSWORD'].strip()
e=create_engine(f'mysql+pymysql://kravira_mc_user:{pw}@178.163.240.131:6330/kravira_mc?charset=utf8mb4', pool_pre_ping=True, connect_args={'connect_timeout':30})
print(e.connect().execute(text('SELECT COUNT(*) FROM mis_protocol')).scalar())
"
```

### 4.2. Скачать мировые базы (интернет; скрипты создать)
- `scripts/fetch_ddinter.py` - качает 8 CSV с `ddinter2.scbdd.com/download/` в
  `data/drug_safety/ddinter/`; собирает единый `ddinter_pairs.parquet`
  (drug_a, drug_b, atc_a, atc_b, level, mechanism, management).
- `scripts/fetch_belarus_formulary.py` - качает PDF формуляра (Пост. №22 2026), парсит в
  `data/regulations/rb_formulary_2026.json`: {inn, atc, forms, doses, group}.
- `scripts/build_high_alert_list.py` - курируемый ISMP high-alert + ONC/CredibleMeds seed →
  `data/drug_safety/high_alert.json` (+ маппинг на ATC/RxCUI).
- `scripts/setup_icd_api_local.sh` - разворачивает WHO ICD-API локально в Docker
  (`icd.who.int/docs/icd-api` local deployment), чтобы **валидировать коды офлайн** без
  отправки диагнозов наружу. Проверка: `GET /icd/release/10/{releaseId}/{code}`.
- (опц.) `scripts/refresh_minzdrav_protocols.py` - дозагрузка новых протоколов по 24
  специальностям (проверить обновления 2026: Пост. №16, №22, №24, №33).

### 4.3. Нормализация ЛС
- `clinical_knowledge/drug_normalizer.py` - имя ЛС из КЗ → канон (формуляр INN) → ATC →
  (опц.) RxCUI через RxNorm `findRxcuiByString`. Кэш в `data/drug_safety/rxnorm_cache.json`.

---

## 5. Экспорт данных заново (с БД) - устраняем потери текущих CSV

Текущие CSV на Render собраны старым экспортёром: нет структурного разбора автора/типа,
diagnostic (УЗИ) не отделяется. На машине с БД:

1. Прогнать `scripts/export_mis_protocol_month.py` для нужных месяцев (январь; при наличии
   февраль; июль) с уже добавленными `doc_type`, `kz_kind`, автор по `doctor_id`.
2. **Дополнительно вынести структурные слоты** в отдельные колонки (для детекторов):
   - `diagnosis_structured` (поле 22: `##`/`|`, основной Dx + код МКБ, если есть в слоте);
   - `treatment_recommendations` и `exam_recommendations` как есть;
   - список ЛС (эвристический парс назначений: имя + доза + путь + длительность).
3. Проверить покрытие `mkb_code` в структурном диагнозе (для детектора 3.1.2).

Результат: `data/mis_protocol/mis_protocol_YYYY-MM.csv` (+ meta) - только на /var/data и
локально (ПДн), в git не коммитим.

---

## 6. Реализация детекторов (движок)

Новый модуль `clinical_knowledge/kz_deep_eval.py` (stdlib + опц. эмбеддинги), функция
`evaluate_kz_deep(case, protocol_ctx, drug_ctx, icd_client) -> findings + scores`.
Подключается в `run_mis_protocol_l1_batch.py` (флаг `--deep-eval`) и в
`review_one_visit_full` (для полного отчёта одного КЗ).

### 6.1. Диагностические детекторы
- `dx_substantiation(case)` - детерминированно + опц. LLM (логическая связь жалобы→Dx).
- `icd_validate(dx_code, dx_text, icd_client)` - код валиден? название кода ≈ тексту Dx?
- `dx_vs_exams(case, protocol)` - покрытие `required_exams`/`must_have` семантически
  (term_catalog + эмбеддинги).
- `red_flag_scan(case, protocol.red_flags)` - «нельзя исключить»/тревожные формулировки без
  маршрутизации → P0.
- `dx_criteria(case, protocol.diagnostic_criteria)`.

### 6.2. Лечебные детекторы
- `drugs_extract(case)` → список назначенных ЛС (drug_normalizer).
- `tx_vs_protocol(drugs, protocol.treatment)` - соответствие группам протокола.
- `formulary_check(drugs, rb_formulary)` - вне формуляра → флаг.
- `ddi_check(drugs, ddinter)` - взаимодействия (severity) → P0/P1 с механизмом+тактикой.
- `high_alert_check(drugs, high_alert, case)` - доза/длительность/мониторинг обязательны.
- `drug_disease_check(drugs, dx_codes, ddinter_ddsi)` - противопоказания.
- `dose_range_check(drugs, dailymed/formulary)` - доза вне диапазона (best-effort).

### 6.3. Оси и итог
- Ось A: PDQI-9-lite (детерминированные прокси заполненности/согласованности + LLM для
  «синтез/полезность»).
- Ось B: клиническое соответствие (dx_vs_exams + tx_vs_protocol + dx_criteria).
- Ось C: safety (red flags + ddi + high_alert + drug_disease + vitals).
- reg55: как есть.
- overall: калиброванный (axes_thresholds) с risk-gate.

### 6.4. LLM-грейдер (масштабируемый второй ридер, MedCheckLLM-стиль)
`scripts/grade_kz_llm.py`:
1. извлечь Dx → выбрать протокол (RAG по protocol_summaries) → взять `kz_checklist`;
2. подать в LLM: текст КЗ + чек-лист (атомарные бинарные пункты) + рубрика PDQI-9-lite +
   правило вывода в JSON; few-shot: 1 хороший + 1 плохой пример; chain-of-thought;
3. LLM возвращает per-item pass/fail + finding + confidence;
4. `needs_human=true` при низкой уверенности или расхождении с детерминированными
   детекторами; такие уходят в очередь методисту.
Хранить сырые ответы LLM только на /var/data (могут содержать ПДн).

---

## 7. Gold-выборка и калибровка (валидность)

1. `scripts/build_kz_gold_sample.py` - стратифицированная выборка 600-800 КЗ (специальность
   × банд × pay_type × red-flag), манифест на /var/data.
2. **Двойная человеческая разметка методистами** по кодбуку (PDQI-9 + оси A/B/C + флаг
   «potential harm» + класс ошибки + вердикт) с LLM-предразметкой; арбитраж при
   расхождении > 1 балла; согласие - weighted kappa (цель κ ≥ 0.6).
3. `scripts/calibrate_axes_thresholds.py` - пороги статусов из ROC/квантилей по gold, не
   «90/75/50»; веса осей - ordinal/isotonic регрессия к вердикту методиста.
4. Валидация на held-out 20%: **QWK** (статус vs методист) цель ≥ 0.70; **recall по
   potential-harm** цель ≥ 0.90 (не пропустить опасное важнее точности).
5. После валидации - включить `CONSULT_AXES_OVERALL=1`.

---

## 8. Метрики успеха (было / цель)

| Метрика | Сейчас | Цель |
|---|---|---|
| Соответствие №55 (январь) | 67.7% | пересчитать на чистых данных |
| exams / treatment block | 24 / 21 | реалистично после структурных items + семантики |
| Диагнозы с валидным кодом МКБ | ~86% (эвристика) | измерить через WHO ICD-API |
| Находки «ошибка диагноза/лечения» на 100 КЗ | не измеряется | измерить, ранжировать по severity |
| Recall potential-harm (vs методист) | не измеряется | ≥ 0.90 |
| QWK (статус vs методист) | 0.32-0.36 (proxy) | ≥ 0.70 |
| Покрытие red_flags/high-alert в протоколах | 44% / seed | ≥ 90% топ-Dx |

---

## 9. ПДн и что коммитим

- **Только на /var/data и локально (не в git):** CSV выгрузки, `cases.jsonl`,
  сырые ответы LLM, gold-манифест с `visit_id`/`patient_id`, `rxnorm_cache`.
- **В git можно:** агрегаты summary (уже включают `patient_id` в worst_visits - как принято
  в проекте), справочники без ПДн (`data/regulations/*`, `data/drug_safety/*` кроме сырых
  дампов с лицензионными ограничениями - проверить лицензии DDInter/ISMP перед коммитом),
  скрипты, кодбук разметки, отчёты `data/ml/reports/*` (агрегаты).
- **Лицензии:** DDInter - academic/open (проверить условия перед коммитом сырых CSV; при
  сомнении держать только на /var/data и коммитить производный индекс). WHO ICD-11 -
  открытая лицензия; ICD-10 - использовать локальный деплой. Формуляр РБ - гос. НПА.
- Правило дефисов (`hyphen-dash.mdc`) и `BUILD_VERSION` (`build-version.mdc`) - соблюдать.

---

## 10. Порядок выполнения (чек-лист для машины с БД)

1. [ ] Проверить доступ к БД (4.1).
2. [ ] Скачать мировые базы: DDInter, формуляр, high-alert; поднять ICD-API локально (4.2).
3. [ ] `drug_normalizer` + RxNorm-кэш (4.3).
4. [ ] Пере-экспорт mis_protocol с автором/типом/структурой (раздел 5).
5. [ ] `kz_deep_eval.py`: диагностические + лечебные детекторы + оси + risk-gate (раздел 6).
6. [ ] LLM-грейдер `grade_kz_llm.py` (MedCheckLLM-стиль) (6.4).
7. [ ] Прогнать `run_mis_protocol_l1_batch.py --deep-eval` по месяцам; собрать summary с
       `findings` и новыми осями.
8. [ ] Gold-выборка + двойная разметка + калибровка порогов/весов (раздел 7).
9. [ ] Валидация (QWK, recall harm); при достижении целей включить axes-режим.
10. [ ] Обновить дашборд: панель «Находки» (ошибки диагноза/лечения по severity + ссылки),
        оси A/B/C, №55, перцентиль по специальности.
11. [ ] Обновить этот план (факт/метрики), поднять `BUILD_VERSION`, `git push`.

---

## 11. Риски

- **Матч названий ЛС** (свободный текст врача ↔ INN/ATC) - главный технический риск;
  смягчаем нормализатором + RxNorm + синонимами; при неуверенности - `needs_human`.
- **Ложные срабатывания DDI** (назначения в разное время, не одновременный приём) -
  помечать как «проверить», не как факт вреда.
- **ПДн наружу**: диагнозы/ЛС не отправлять во внешние API - ICD-API локально, DDInter
  офлайн, RxNorm - только имена препаратов (без пациента).
- **Смещение LLM**: финальное слово - методист; замер согласия; CoT + рубрика + few-shot.
- **Актуальность протоколов/формуляра**: фиксировать дату версии НПА в `source_ref`.
- **Стоимость разметки**: снижаем стратификацией + LLM-предразметкой.
