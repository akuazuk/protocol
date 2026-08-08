# МО Аналитика: трёхслойная модель оценок + дашборд «только внимание»

Статус: **active** (план, код не меняем до согласования фаз)  
Дата: 2026-08-08  
Владелец продуктового требования: пользователь (сессия 2026-08-08)  
База кода: `origin/main` @ `a34b6070`  
Связанные:  
`docs/methodist/mo-evaluation-catalog.md`,  
`docs/plans/2026-08-03-mo-rubric-mz-scoring-viz-v1.md`,  
`docs/plans/2026-08-08-mo-action-queue-precise-signals-v2.md`,  
`docs/plans/2026-08-08-mo-patient-history-bundle-v2.md`,  
`docs/plans/2026-08-08-mo-icd-diag-slots-only-v1.md`,  
`docs/plans/2026-08-05-mo-llm-action-queue-judge-v1.md`.

---

## 0. Зачем новый план

Сейчас на основном дашборде и в таблицах одновременно живут:

- сводный `overall_pct` (deep/v3/v4),
- четыре оси A/B/C/D,
- балл №55,
- shadow-рубрика «Как оценивать» (№127/№55),
- coverage/confidence («Полнота» / «Надёжность» - путают с клинической полнотой),
- чипы МКБ и истории,
- очередь по whitelist safety/Dx,
- LLM-судья (Полнота / Диагноз / План) только в разборе случая.

Методист не видит **одной клинической логики**: сначала насколько заполнен МО, потом
верный ли диагноз по клинике, потом верны ли назначения относительно КП и истории
пациента. Вместо этого - смесь индексов, чек-листов и meta-метрик.

**Цель продукта:** оценка МО = три последовательных слоя; главный дашборд показывает
только сигналы «сюда смотреть», а не все возможные проценты.

---

## 1. Продуктовая модель (источник истины)

### Слой A - Полнота оформления разделов

**Вопрос:** заполнены ли и достаточно ли детализированы разделы МО?

**Входы:** жалобы, анамнез (врач), факторы риска, объективный статус, данные
обследований (если есть), клинический диагноз, дата/время, план обследования/лечения
(факт наличия текста).

**МКБ здесь:** только как часть оформления диагноза - код в слотах
`clinical_diagnosis` / `mis_diagnos` / `mkb_code_main` (правило slots-only).
Нет кода при наличии текста Dx - **не** дефект слоя A.
Нет ни текста Dx, ни кода - дефект A (и кандидаты в очередь).

**Регламенты:** Инструкция №127 (структура/глубина) + часть присутствия из №55
(жалобы/анамнез/осмотр/диагноз как «есть поле»).

### Слой B - Правильность диагноза

**Вопрос:** следует ли клинический диагноз из жалоб + анамнеза + осмотра + данных
обследований?

**Входы:** текст Dx + МКБ из слотов диагноза; жалобы; анамнез; objective; exam_data;
история пациента (новый код / смена линии Dx).

**Не путать с:** «код есть в справочнике МКБ» (это подслой B-coding, вторичный).

**Регламенты:** №55 «диагноз обоснован»; формулировка рубрики «вытекает из…»;
concordance / LLM diagnosis_assessment.

### Слой C - План (обследование / лечение / наблюдение) + история

**Вопрос:** соответствуют ли рекомендации найденным клиническим протоколам МЗ и
динамике пациента?

**Входы:** exam/treatment/follow_up рекомендации; подобранные КП (suggest + L1
block_scores); история визитов (полки врач/специальность, prior therapy, коррекции);
safety-оверлей (DDI, red flags) как отдельный hard-gate, не как «качество текста».

**Регламенты:** №55 обследование/лечение по КП; №127 п.10 кратность наблюдения;
рубрика correction/follow_up.

### Safety (поперечный hard-gate)

Не слой качества записи, а **немедленный риск**: Major DDI, red flag, high-alert без
дозы, NSAID dup. Всегда сверху очереди, отдельный KPI «критично», не смешивать со
средним % полноты.

---

## 2. Карта регламентов №127 и №55 → слои

### 2.1 Инструкция №127 (`data/regulations/mz_2015_127.json` + рубрика YAML)

| Тема | Слой | Как используем |
|--|--|--|
| МО проведён и оформлен | A | `mo_complete` |
| Дата/время | A | `datetime` |
| Жалобы детализированы | A | `complaints` (0/0.5/1) |
| Анамнез достаточный | A | `anamnesis` |
| Факторы риска | A | `risk_factors` |
| Объективный осмотр полный | A | `objective` |
| Данные обследований (если есть) | A | `exam_data` |
| Кратность наблюдения (п.10) | C | `follow_up` + chapter hints |
| Динамика / коррекция планов | C | `exam_correction`, `treatment_correction` + история |

### 2.2 Постановление №55 (`data/regulations/mz_2021_55.json`)

| Группа №55 | Критерии | Слой |
|--|--|--|
| Сбор информации | complaints/anamnesis/objective present | A (бинарное присутствие) |
| Диагноз | diagnosis_present, diagnosis_substantiated | A + B |
| Обследование и лечение | exams_per_protocol, treatment_per_protocol | C |
| Преемственность | follow_up_present (ссылка на №127.10) | C |
| Служебное | no_unhandled_red_flag (вне формулы %) | Safety |

**Важно:** №55 даёт pass/fail %; рубрика №127 даёт глубину 0/0.5/1. Это **не один KPI**.
В новой модели:

- слой A берёт глубину из рубрики (или LLM completeness) и presence из №55 как floor;
- слой B - обоснованность Dx (№55 substantiated + concordance/LLM);
- слой C - alignment к КП (№55 + L1/suggest) + динамика (рубрика + история);
- отдельная карточка «Чек-лист №55» остаётся в разборе / вторичном блоке, не hero.

### 2.3 Что не смешивать (жёстко)

1. Coverage «полнота проверки модели» ≠ слой A.  
2. Балл №55 ≠ «качество МО целиком».  
3. Справочник МКБ (directory/name) ≠ правильность диагноза по клинике.  
4. Safety ≠ слой C (план по КП).  
5. Сводный deep overall - legacy-тренд, не продуктовый смысл трёх слоёв.

---

## 3. Как собрать слои из того, что уже есть в коде

Контракт уже почти есть в **разборе случая**: LLM action-judge
(completeness / diagnosis / plan) и вердикты методиста (Полнота / Диагноз / Рекомендации).
На дашборде его нет. Делаем единый `layer_scores` на каждый clinical_visit.

### 3.1 Поля склада / API (новые)

На `fact_mo_case` (или рядом `fact_mo_layer_score`):

| Поле | Тип | Смысл |
|--|--|--|
| `layer_a_pct` | 0–100 / null | полнота оформления |
| `layer_b_pct` | 0–100 / null | правильность диагноза |
| `layer_c_pct` | 0–100 / null | план vs КП + история |
| `layer_a_band` | ok / weak / bad | пороги для внимания |
| `layer_b_band` | ok / weak / bad | |
| `layer_c_band` | ok / weak / bad | |
| `attention_primary` | enum | главный слой внимания: safety / a / b / c / none |
| `attention_reasons` | json/text | 1–3 человекочитаемых причины |
| `history_tier` | уже есть | контекст для C и B |
| `reg55_pct` | уже есть | вторичный чек-лист |
| `rubric_pct` | желательно persist | вторичная методика МЗ |
| `overall_pct` | уже есть | legacy «сводный индекс» |

### 3.2 Формулы v1 (детерминированный каркас, 100% покрытие)

Пока LLM judge есть не на всех днях - **всегда** считаем proxy; LLM, если есть,
перекрывает/уточняет band (не обязательно % до калибровки).

#### Слой A (proxy)

Источники: deep `axes.documentation`; рубрика group documentation+clinical
(complaints, anamnesis, risk_factors, objective, mo_complete, datetime, exam_data,
diagnosis presence); №55 presence fails.

Предложение v1:

```
A_pct = round(0.55 * deep_doc + 0.45 * rubric_doc_clinical_pct)
```

где `rubric_doc_clinical_pct` - среднее scored критериев A-affinity (n/a вне знаменателя).  
Band: `bad` если A_pct &lt; 55 или нет Dx-текста и нет кода; `weak` если &lt; 75; иначе `ok`.

МКБ в A: chip статуса кодирования (`icd_visit_status`) как аннотация, не драйвер %.

#### Слой B (proxy)

Источники: findings `B_dx_no_support`, `B_dx_absent`; deep concordance; рубрика
`diagnosis`; LLM `diagnosis_assessment` если есть; history `new_for_profile` /
`history_dx_line_break` как soft.

```
если B_dx_absent → band=bad, B_pct≤40
если B_dx_no_support → band=bad, B_pct≤50
иначе B_pct ≈ deep_concordance (с потолком, если rubric diagnosis = 0)
```

Directory/name mismatch → `weak` максимум (не bad), чтобы не засорять очередь.

#### Слой C (proxy)

Источники: №55 fail exams/treatment/follow_up; L1 `block_scores.exams/treatment`;
рубрика exam_plan/treatment_plan/follow_up; dynamics correction при наличии prior;
LLM `plan_assessment`; history tier.

```
C_base = mean(available: №55_plan_pct, rubric_plan_pct, l1_exams, l1_treatment)
если history_tier in {first_contact, new_for_profile} и C_base < 75 → band не выше weak
если prior есть и correction = 0 → штраф / band weak|bad
Safety findings не входят в C_pct (отдельный gate)
```

#### Attention primary (для дашборда и сортировки)

Порядок приоритета:

1. Safety whitelist (critical)  
2. Layer B bad (`B_dx_*`)  
3. Layer C bad (план/КП + при необходимости история)  
4. Layer A bad (пустое оформление)  
5. иначе none  

Это **согласуется** с `mo_action_queue_select` v2 (не возвращаем №55-тикеты как
единственную причину очереди).

### 3.3 LLM и человек

| Источник | Роль |
|--|--|
| LLM action-judge 3 вопроса | эталон семантики слоёв; заполняет % когда есть JSONL |
| Вердикты методиста C/D/R | gold; агрегаты «согласие с моделью» на overview (фаза позже) |
| Night Gemini grader | обучение/BI, не hero дашборда |

---

## 4. История пациента - где обязательна

| Место | Использование |
|--|--|
| Слой B | смена линии Dx / новый код у врача → осторожнее к «новому» диагнозу |
| Слой C | коррекция планов; сравнение therapy vs prior; эпизод для подбора КП |
| Очередь | boost first_contact / new_for_profile при слабом C или B |
| UI overview | распределение tier среди attention-кейсов (не отдельный % «истории») |
| UI разбор | блок истории **выше** LLM (уже сделано); связка «почему C weak» |

Без prior: критерии correction = n/a (не zero); first_contact не штрафуется за
«нет коррекции», но может получить attention, если план слабый при первом контакте.

---

## 5. Новый главный дашборд (Overview)

Принцип: **только то, по чему находятся МО «где что не то».**

### 5.1 Структура первого экрана (desktop)

```
┌─────────────────────────────────────────────────────────────┐
│ Период · филиал · специальность · врач · [только clinical]  │
├───────────────┬───────────────┬───────────────┬─────────────┤
│ OPS (вторично)│ Оценено MTD   │ Свежесть склада│ Прогноз объёма│
├───────────────┴───────────────┴───────────────┴─────────────┤
│ ATTENTION (hero)                                              │
│  [В очереди critical] [important]                             │
│  [A плохо] [B плохо] [C плохо] [Safety P0]                    │
│  клик → очередь с фильтром слоя                               │
├───────────────────────────────────────────────────────────────┤
│ ТРЕНД СЛОЁВ (3 линии A/B/C + тонкий safety count)             │
│  НЕ пять линий «четыре индекса + итог» по умолчанию           │
├────────────────────────────┬──────────────────────────────────┤
│ Топ причин внимания        │ Куда смотреть (врачи / спец.)    │
│ (whitelist + layer tags)   │ (n≥порог, delta layer bad %)     │
├────────────────────────────┴──────────────────────────────────┤
│ <details> Методика МЗ: рубрика top-fail · №55 % · сводный deep│
└───────────────────────────────────────────────────────────────┘
```

### 5.2 Что убрать с hero / дефолта

| Сейчас на overview | Решение |
|--|--|
| Итоговая оценка как главный KPI | Демотировать в `<details>` «Сводный индекс (deep)» |
| Четыре индекса A/B/C/D одной серией | Заменить на тренд **трёх слоёв**; ось Safety - count |
| Рубрика МЗ avg в верхнем ряду | Вторичный блок методики |
| Соответствие №55 рядом с итогом | Вторичный чек-лист |
| Heatmap спец×МКБ, case-mix врачей, CRM funnel | Ниже fold или страницы Аналитика |
| «Полнота» как coverage | Переименовать везде в «Полнота проверки»; с hero убрать |

### 5.3 Yesterday (операционный день)

Оставить: completeness pipeline (получено/оценено).  
Hero дня = те же Attention KPI за вчера + таблица очереди (уже близко к нужному).  
Добавить колонки/фильтр **слой** (A/B/C/safety).  
Убрать акцент на «четыре индекса vs вчера», заменить мини-спарклайнами слоёв.

### 5.4 Очередь (Work)

Оставить whitelist v2.  
Добавить:

- тег слоя на строке;
- фильтры: слой, history_tier, есть LLM judge / нет;
- сортировка: safety → B → C → A → overall.

Причина строки = человекочитаемый сигнал (`B_dx_no_support`, Major DDI…), не «P0 №55».

### 5.5 Таблица документов

**Дефолтные колонки:**

Визит · Дата · Врач/спец · Диагноз · МКБ-chip · История-chip ·
**A** · **B** · **C** · Attention · Статус CRM.

**Скрыть по умолчанию (column manager):** Итог deep, №55 %, coverage, confidence.

### 5.6 Разбор случая (drawer)

Уже почти правильный каркас. Целевая иерархия:

1. Клинический текст (якоря на слоты Dx).  
2. **Три слоя** крупно (A/B/C %) + safety badge.  
3. История пациента (как влияет на B/C).  
4. Подбор КП + почему C.  
5. LLM 3 вопроса (если есть) vs детерминизм.  
6. Замечания с тегом слоя.  
7. `<details>`: №55 пункты, рубрика 13 критериев, оси deep, сводный индекс.  
8. Решение методиста (вердикты C/D/R = слои).

### 5.7 Страницы Аналитика (не hero)

| Страница | Содержание после редизайна |
|--|--|
| Specialties | boxplot **по слоям** (не одному overall) |
| Diagnoses | treemap: volume × **B-bad %** (или C), не только overall |
| Doctors | scatter: volume × attention rate / layer bad |
| Safety | без изменений по смыслу (P0–P3 / whitelist) |
| Data quality | coverage/confidence/freshness - сюда, не на overview |
| Settings | легенда слоёв, пороги band, methodology |

---

## 6. Визуализация - спецификация

### 6.1 Attention strip

- 6 плиток: Queue critical, Queue important, A bad, B bad, C bad, Safety P0.  
- Число + % от оценённых clinical.  
- Цвет: critical=красный, important/bad=янтарь, ok-фон нейтральный.  
- Клик: `queue?layer=b&band=bad` и т.п.  
- Без em/en dash в подписях (дефис).

### 6.2 Layer trend

- ECharts: 3 series 0–100 (A/B/C) по дням MTD.  
- Отдельная ось/бар: число safety critical.  
- Drill по клику на день → yesterday/documents с фильтром даты.

### 6.3 Pareto причин

- Только коды из queue whitelist + агрегированные «A: пустые жалобы», «C: exams≠КП».  
- Не сырой Pareto всех D_reg55_*.

### 6.4 Coaching (вторично)

- Таблица top-fail критериев рубрики (как сейчас) - полезна для обучения врачей.  
- №55 top failed points - отдельно.

### 6.5 Именование (обязательный словарь UI)

| Было | Стало |
|--|--|
| Итоговая оценка | Сводный индекс (deep) |
| Полнота (в таблице, если coverage) | Полнота проверки |
| Балл №55 | Чек-лист №55 |
| Рубрика МЗ | Методика «Как оценивать» (черновик/калибровка) |
| Оформление / Согласованность / … | Слой A / Слой B / Слой C (+ подпись методики) |
| Четыре индекса | убрать из hero |

---

## 7. Фильтры (единый контракт)

Глобальные (уже есть): период, филиал, спец, врач, document_kinds=clinical(+consultation).

**Новые:**

| Фильтр | Значения |
|--|--|
| `layer` | a / b / c / safety |
| `layer_band` | bad / weak / ok |
| `attention_only` | 1 (default на overview drill) |
| `history_tier` | first_contact / new_for_profile / known_to_doctor / … |
| `has_llm_judge` | 0/1 |
| `rubric_criterion` | уже есть - оставить для coaching |

Default overview drill и queue landing: `attention_only=1`.

---

## 8. Фазы внедрения

### Фаза 0 - Согласование (этот документ)

- [ ] Утвердить 3 слоя + safety gate  
- [ ] Утвердить hero overview (attention strip)  
- [ ] Утвердить словарь названий  
- [ ] Зафиксировать пороги band v1 (55 / 75) или скорректировать

### Фаза 1 - Слойные поля без ломки UI (backend)

1. Модуль `clinical_knowledge/mo_layer_scores.py`: считает A/B/C из deep+rubric+№55+findings+history.  
2. Persist в warehouse при upsert/recompute.  
3. API cases/overview отдают `layer_*` + aggregates.  
4. Тесты на фикстурах (пневмония J18.9; пустой Dx; dx no support; plan fail; first_contact).  
5. Не менять hero UI ещё - только API + optional debug.

**Метрика фазы 1:** 100% clinical_visit за день имеют `layer_*` или явный null+reason.

### Фаза 2 - Overview / Yesterday / Queue UI

1. Attention strip + layer trend.  
2. Демотировать four-index / overall / №55 / rubric avg.  
3. Queue + documents: колонки A/B/C, фильтр слоя.  
4. Переименовать coverage.  
5. Smoke GCE.

**Метрика:** клик с overview bad-B ведёт в очередь, где ≥90% строк имеют B-сигнал.

### Фаза 3 - Drawer + история в слое C

1. Крупные A/B/C в drawer.  
2. Явная связь history → B/C («почему weak»).  
3. Dynamics correction завязать на bundle prior (не только single prior).  
4. KP suggest объясняет C.

### Фаза 4 - LLM как усилитель слоёв

1. Если есть action-judge - показывать рядом с proxy и расхождение.  
2. Backfill judge не блокирует дашборд.  
3. Калибровка: где LLM и proxy расходятся >X - очередь «на обучение».

### Фаза 5 - Аналитика / врачи / диагнозы

1. Перевести scatter/boxplot/treemap на layer bad rates.  
2. Отчёты методиста: экспорт A/B/C.  
3. Expert portal: yesterday attention в тех же терминах.

### Фаза 6 - Калибровка методики МЗ

1. Решить, становится ли рубрика primary для A (глубина №127).  
2. Не сливать №55 и рубрику в один % без отдельного решения.  
3. Обновить `mo-evaluation-catalog.md` под слои как канон дашборда.

---

## 9. Метрики успеха

| Метрика | Сейчас | Цель |
|--|--|--|
| Число «главных %» на overview hero | 5–7 | ≤1 ops-ряд + attention (без смеси смыслов) |
| Доля кликов overview → очередь с понятной причиной | низкая / шум | ≥80% строк с layer/safety reason |
| Согласованность очереди и overview | расходится (Pareto≠whitelist) | одинаковые семантики внимания |
| Покрытие layer_scores на clinical | 0 | 100% после recompute |
| Ложные «плохие» из ICD/№55 alone в hero | да | 0 |
| Использование истории в C | почти невидимо | видно в band/reason при prior/first_contact |

---

## 10. Риски

| Риск | Митигация |
|--|--|
| Путаница A vs coverage | жёсткий словарь UI + тесты на строки |
| Proxy B слишком грубый (presence≠entailment) | LLM/human для bad-B; proxy только для triage |
| C зависит от качества KP match | trust-aware; при низком trust C=n/a или weak max |
| Два % (рубрика vs №55) снова смешают | разные блоки; запрет merge без фазы 6 |
| Регресс «четыре индекса» для привыкших | `<details>` + settings toggle «классические оси» |
| Recompute нужен для истории soft-fill ICD | ops runbook после фазы 1 |

---

## 11. Что сознательно не делаем в v1

- Не делаем один «супербалл МЗ» из №55+№127.  
- Не возвращаем full-document ICD search.  
- Не тикетуем очередь по любому D_reg55_*.  
- Не кладём night LLM grader в hero.  
- Не оцениваем non-clinical.

---

## 12. Definition of Done (весь план)

1. В коде и UI есть понятия Слой A/B/C + Safety.  
2. Overview hero = attention strip + layer trend; legacy индексы вторичны.  
3. Очередь и overview говорят на одном языке причин.  
4. История пациента явно входит в объяснение B/C.  
5. Каталог оценок обновлён: слои - канон дашборда; остальные метрики - приложения.  
6. Есть фазы 1–2 в проде на GCE с smoke на реальных clinical_visit.

---

## 13. Первая безопасная команда после согласования

```bash
# после approve фазы 1
scripts/ops/git_task_start.sh mo-layer-scores-v1 --pc=pc1 \
  --branch=cursor/mo-layer-scores-v1-pc1
```

Реализовать `mo_layer_scores.py` + warehouse columns + API aggregates **без** ломки
текущего overview (feature flag `MO_LAYER_SCORES_UI=0`).

---

## Приложение A - Маппинг старых осей → слои (шпаргалка)

| Старое | Новое |
|--|--|
| deep documentation | вклад в A |
| deep clinical_concordance | вклад в B (+ часть C protocol exams) |
| deep safety | Safety gate |
| deep regulatory / reg55_pct | чек-лист; куски A/B/C по группам критериев |
| rubric documentation/clinical | A (глубина №127) |
| rubric diagnosis | B |
| rubric regulatory/dynamics | C |
| LLM completeness/diagnosis/plan | A/B/C semantic overlay |
| overall_pct | сводный legacy |

## Приложение B - Пример внимания (пневмония)

- Клинический диагноз: «Внебольничная … пневмония…»  
- Диагноз МИС: `J18.9`  
- A: смотрим детализацию жалоб/анамнеза/осмотра (№127), не коды из анамнеза.  
- B: пневмония следует ли из клиники; МКБ J18.9 - coding chip.  
- C: план обследования/лечения vs КП пневмонии; если были prior визиты - учтена ли динамика.  
- Safety: отдельно (например NSAID/DDI в плане).  
На overview такой случай попадает в attention только если какой-то слой bad/weak или safety.
