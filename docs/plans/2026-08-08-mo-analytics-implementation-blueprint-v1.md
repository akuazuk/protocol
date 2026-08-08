# МО Аналитика: подробный blueprint изменений (UI + оценки + данные)

Статус: **active**
Дата: 2026-08-08
Методика-канон: `2026-08-08-mo-analytics-mz-sheet-layers-v2.md`
**Канон экранов и меню:** `2026-08-08-mo-analytics-ui-target-v2.md` (короткое русское меню; §5–§10 этого blueprint выполнять только в его объёме)
Каталог метрик (справочник): `docs/methodist/mo-evaluation-catalog.md`
Draft PR планов: https://github.com/akuazuk/protocol/pull/77

Этот документ - **пошаговый план реализации** склада, движка и API.
Поверхность UI не раздувать: меню из 6 пунктов, без отдельных экранов «Специальности / Диагнозы / Safety / Data quality».

Каждый шаг: цель → артефакты → контракт API/UI → приёмка → риски → зависимость.

**Не начинать код фазы N, пока не закрыта приёмка N−1** (кроме явно параллельных шагов).

---

## Оглавление

0. Исходное состояние и целевой каркас
1. Словарь UI (обязательный)
2. Модель данных склада
3. Движок зон (оценки)
4. API контракты
5. Overview (месяц) - дашборд
6. Yesterday (день)
7. Очередь разбора
8. Таблица документов
9. Разбор случая (drawer)
10. Страницы Аналитика (врачи / спец / диагнозы / safety / data-quality)
11. Фильтры, URL, сохранённые виды
12. Feature flags и миграция без даунтайма
13. Тесты и калибровка
14. Порядок релизов (GCE) и DoD
15. Чеклист владельца перед стартом кода

---

## 0. Исходное состояние → целевой каркас

### 0.1 Что есть сейчас (кратко)

| Поверхность | Сейчас |
|--|--|
| Overview | KPI: записи, оценено, итоговая, рубрика, attention, P0, прогноз; график 4 осей+итог; heatmap; врачи; Pareto; funnel; №55; рубрика top-fail |
| Yesterday | completeness pipeline; 4 индекса; findings; врачи; P0/P1 actions; flow |
| Documents | Итог, №55, Полнота(=coverage), Надёжность |
| Queue | whitelist сигналов; итог; причина |
| Drawer | текст → №55 → details(оси/рубрика) → история → LLM → KP → findings → решение |
| Оценки | deep A/B/C/D, overall, reg55 binary %, rubric ternary shadow, ICD chips, history |

### 0.2 Целевой каркас продукта

```
Оценка случая
├── Зона 1 Оформление (№127)     0/0.5/1 → zone1_pct / band   [КП нет]
├── Зона 2a Диагноз              0/0.5/1 → zone2a_pct / band  [КП да]
├── Зона 2b+c План по КП         0/0.5/1 → zone2b_pct / band  [КП да; 2c+история]
└── Safety gate                  critical/important/none      [не %]

Дашборд внимания
├── плитки: Оформление↓ · Диагноз↓ · План по КП↓ · Safety · Очередь
├── тренд 3 зон
└── <details> legacy: deep / binary №55 / методика подробно

Разбор случая
├── 3 зоны крупно + safety
├── таблица 13 критериев «Как оценивать»
├── история (для 2c / эпизода КП)
├── подобранные КП (вход зоны 2)
└── LLM / решение методиста в тех же трёх ярлыках
```

---

## 1. Словарь UI (зафиксировать до вёрстки)

| Запрещено на hero / дефолтных колонках | Обязательное имя |
|--|--|
| Итоговая оценка | Сводный индекс (deep) - только secondary |
| Полнота (если это coverage) | Полнота проверки модели |
| Балл №55 / Соответствие №55 как главный | Чек-лист №55 (binary) - secondary |
| План / Рекомендации без уточнения | **План по КП** |
| Оформление без уточнения | **Оформление (№127)** |
| Согласованность / ось B как главный | **Диагноз** (зона 2a) |
| «Не соответствует КП» на жалобах/осмотре | запрещено текстом findings |

Подписи band:

| band | RU | цвет |
|--|--|--|
| `ok` | в норме | green/neutral |
| `weak` | слабо | amber |
| `bad` | плохо | red |
| `na` | нет данных / КП не сопоставлен | muted |

Дефисы в текстах: короткий `-`, без em/en dash (`hyphen-dash.mdc`).

---

## 2. Модель данных склада

### Шаг 2.1 - DDL

**Файл:** `clinical_knowledge/mo_daily.py` (`initialize_warehouse` + migrate ALTER).

Добавить в `fact_mo_case`:

| Колонка | Тип | Описание |
|--|--|--|
| `zone1_pct` | REAL NULL | Оформление №127, 0–100 |
| `zone2a_pct` | REAL NULL | Диагноз |
| `zone2b_pct` | REAL NULL | План по КП (2b+2c scored) |
| `zone1_band` | TEXT | ok/weak/bad/na |
| `zone2a_band` | TEXT | |
| `zone2b_band` | TEXT | |
| `zone2b_kp_status` | TEXT | `matched` / `unmatched` / `not_applicable` |
| `attention_primary` | TEXT | safety / zone1 / zone2a / zone2b / none |
| `attention_reason_ru` | TEXT | 1 строка для таблицы/очереди |
| `rubric_json` | TEXT NULL | сжатый JSON 13 критериев (опционально фаза 1b) |
| `rubric_pct` | REAL NULL | среднее ternary (persist; сейчас часто live-only) |
| `layer_engine` | TEXT | `mo_zones_v1` |
| `layer_updated_at` | TEXT | ISO UTC |

Индексы:

```sql
CREATE INDEX IF NOT EXISTS idx_case_attention
  ON fact_mo_case(attention_primary, visit_date);
CREATE INDEX IF NOT EXISTS idx_case_zone_bands
  ON fact_mo_case(zone1_band, zone2a_band, zone2b_band, visit_date);
```

**Приёмка 2.1:** чистый warehouse и прод-migrate на GCE поднимают колонки без потери строк; `PRAGMA table_info` содержит все поля.

### Шаг 2.2 - Запись при upsert / recompute

**Где:** `upsert_warehouse` в `mo_daily.py`; `scripts/recompute_mo_days.py`.

Порядок внутри case upsert (после soft-fill МКБ и history):

1. Собрать clinical slots + findings + block_scores + suggest hint (если уже есть) + history tier.
2. Вызвать `compute_mo_zone_scores(case_ctx) -> dict`.
3. Записать zone_* / attention_* в row.
4. Не затирать CRM.

**Приёмка 2.2:** recompute одного дня заполняет `zone*` у всех `clinical_visit|consultation`; у non-clinical - NULL и `attention_primary=none`.

### Шаг 2.3 - Обратная совместимость

`overall_pct`, `axis_*`, `reg55_pct`, `history_*` **не удалять**.
Новые отчёты читают zone_*; старые графики читают overall до фазы UI.

---

## 3. Движок зон (оценки)

### Шаг 3.1 - Модуль

**Новый файл:** `clinical_knowledge/mo_zone_scores.py`
**Конфиг:** расширить `config/mo_rubric_mz.yaml`:

```yaml
# на каждый criterion дополнительно:
requires_protocol: false   # true для 8–13
zone: documentation        # documentation | diagnosis | plan
optional: false            # true для exam_data
```

Карта id → zone (зафиксировать в YAML, не хардкодить в UI):

| zone | ids |
|--|--|
| documentation | mo_complete, datetime, complaints, anamnesis, risk_factors, objective, exam_data |
| diagnosis | diagnosis |
| plan | exam_plan, treatment_plan, follow_up, exam_correction, treatment_correction |

### Шаг 3.2 - Алгоритм `compute_mo_zone_scores`

Вход `case_ctx`:

```python
{
  "clinical": {...},           # слоты МО
  "meta": {...},               # date/time, codes
  "block_scores": {...},
  "findings": [...],
  "patient_history": {...},    # public bundle / summary
  "protocol_suggest": {...},   # optional; items[]
  "icd_visit_status": {...},
  "llm_action_judge": {...},   # optional overlay
}
```

Процедура:

1. `rubric = evaluate_mo_rubric_mz(...)` (уже есть) - получить 13 scores.
2. Для каждого criterion с `requires_protocol=true`:
   - если нет matched KP (`zone2b_kp_status=unmatched`) и rule требует alignment → score=`n/a` (не 0), reason=`kp_not_matched`.
   - если rule = presence-only и текста плана нет → 0.
3. Для correction (`requires_prior_visit`): нет prior → `n/a`.
4. Агрегаты:

```
zone1_pct = mean(scored documentation) * 100
zone2a_pct = score(diagnosis) * 100   # с подправилами §3.3
zone2b_pct = mean(scored plan) * 100
```

5. Bands (§3.4).
6. `attention_primary` приоритет: safety whitelist → zone2a bad → zone2b bad → zone1 bad → none.
7. `attention_reason_ru` из главного сигнала (1 фраза).

### Шаг 3.3 - Подправила диагноза (2a)

Из таблицы владельца:

| Условие | Итог criterion diagnosis |
|--|--|
| Нет текста Dx в слотах и нет кода | **0** |
| Есть Dx, нет опоры жалобами/анамнезом/осмотром | **0.5** |
| Есть Dx + опора, но МКБ/класс слабо / directory weak | **0.5** |
| Есть Dx + опора + МКБ/класс ок | **1** |

Код МКБ только из слотов диагноза (уже в коде).
Directory/name - chip, не отдельный hero %.

### Шаг 3.4 - Пороги band (v1, калибруемые)

| band | Условие |
|--|--|
| `bad` | pct &lt; 50 **или** обязательный критерий зоны = 0 |
| `weak` | 50 ≤ pct &lt; 85 **или** есть 0.5 при отсутствии нулей |
| `ok` | pct ≥ 85 и нет нулей по обязательным |
| `na` | в зоне 0 scored критериев (например plan без KP и без текста) |

Пороги вынести в `config/mo_zone_bands.yaml` + env override позже.

### Шаг 3.5 - LLM overlay (не блокер)

Если есть `llm_action_judge`:

| LLM блок | Зона |
|--|--|
| completeness | documentation |
| diagnosis_assessment | diagnosis |
| plan_assessment | plan |

Правило v1: LLM **не переписывает** % склада; добавляет в API
`zones.llm_overlay = {zone1, zone2a, zone2b}` и в drawer показывает рядом.
Расхождение |proxy−llm| > 25 п.п. → тег `needs_calibration` (фильтр позже).

### Шаг 3.6 - Тесты движка

**Файл:** `tests/test_mo_zone_scores.py`

Фикстуры минимум:

1. Полный хороший МО + KP match → все ok.
2. Пустые жалобы/осмотр → zone1 bad; тексты findings без «КП».
3. Dx есть, клиника пустая → zone2a weak/bad.
4. План пустой при KP match → zone2b bad.
5. План пустой без KP → zone2b na или weak presence, **не** «не по КП».
6. Correction без prior → n/a.
7. Pneumonia clinical + J18.9 mis → diagnosis использует эти слоты.
8. Non-clinical → zones null.

**Приёмка 3:** pytest green; снимок JSON фикстуры в `tests/fixtures/mo_zones/`.

---

## 4. API контракты

### Шаг 4.1 - Case detail

`GET /api/methodist/mo/cases/{id}` дополнить:

```json
"zones": {
  "engine": "mo_zones_v1",
  "zone1": {"pct": 72.0, "band": "weak", "label_ru": "Оформление (№127)", "requires_protocol": false},
  "zone2a": {"pct": 50.0, "band": "bad", "label_ru": "Диагноз", "requires_protocol": true},
  "zone2b": {"pct": null, "band": "na", "label_ru": "План по КП", "kp_status": "unmatched", "requires_protocol": true},
  "safety": {"band": "none", "codes": []},
  "attention_primary": "zone2a",
  "attention_reason_ru": "Диагноз слабо опирается на жалобы и осмотр",
  "criteria": [ /* 13 items from rubric + requires_protocol + zone */ ],
  "llm_overlay": null
}
```

Live: пересчёт zones в detail (как rubric сейчас), даже если склад старый.

### Шаг 4.2 - Overview aggregates

`GET /api/methodist/mo/overview` добавить:

```json
"attention": {
  "n_evaluated": 1200,
  "zone1_bad": 80, "zone1_bad_pct": 6.7,
  "zone2a_bad": 45, "zone2a_bad_pct": 3.8,
  "zone2b_bad": 110, "zone2b_bad_pct": 9.2,
  "safety_critical": 12,
  "queue_critical": 20, "queue_important": 55
},
"zone_trends": [
  {"date": "2026-08-01", "zone1_avg": 78.1, "zone2a_avg": 71.0, "zone2b_avg": 66.4, "safety_critical": 2}
],
"attention_pareto": [
  {"key": "zone2b", "reason_ru": "План обследования не покрывает КП", "n": 40}
],
"legacy": { "avg_overall": ..., "reg55": ..., "rubric_mz": ... }
```

### Шаг 4.3 - Cases list filters

Новые query params:

| param | values |
|--|--|
| `zone` | zone1 / zone2a / zone2b / safety |
| `zone_band` | bad / weak / ok / na |
| `attention_only` | 0/1 |
| `kp_status` | matched / unmatched |
| `history_tier` | уже частично есть через chips - сделать серверный фильтр |

Default для drill с overview: `attention_only=1`.

### Шаг 4.4 - Daily report

`build_daily_report` / yesterday payload: те же attention aggregates за день + actions с `zone` tag.

**Приёмка 4:** контрактные тесты FastAPI/TestClient на overview+cases+detail; OpenAPI/ручной curl на GCE staging.

---

## 5. Overview (месяц) - пошаговая вёрстка

**Файлы:** `frontend/web/methodist/mis-kz-quality.html`, `mo-app.js`, `mo-charts.js`, `mo-ui.css`.

### Шаг 5.1 - Каркас HTML

Заменить/обернуть `#month-kpis` и блоки графиков:

```
section[data-page=overview]
  #month-ops-kpis          (вторичный ряд)
  #month-attention-strip   (hero)
  #month-zone-trend        (chart host)
  #month-attention-pareto  (chart + table)
  #month-attention-doctors (compact)
  details#month-secondary
    #month-legacy-trend    (старые 4 оси - скрыто по умолчанию)
    #month-reg55-card
    #month-rubric-card     (top-fail критериев)
    #month-heatmap
    #month-crm-funnel
```

### Шаг 5.2 - Ряд OPS (не hero)

Плитки: Записи MTD · Оценено · Свежесть склада · Прогноз объёма.
Стиль: меньше шрифт, без красных акцентов.

### Шаг 5.3 - Attention strip (hero)

6 плиток:

1. В очереди: критично (`queue_critical`)
2. В очереди: важно (`queue_important`)
3. Оформление плохо (`zone1_bad` + %)
4. Диагноз плохо (`zone2a_bad` + %)
5. План по КП плохо (`zone2b_bad` + %)
6. Safety (`safety_critical`)

Поведение клика:

| Плитка | Навигация |
|--|--|
| очередь | `switchPage('queue')` + filters |
| zone1/2a/2b | `queue` или `documents` с `zone=&zone_band=bad` |
| safety | `queue` + safety filter / page safety |

Tooltip плитки «План по КП»:
`Считается при подобранном КП. Жалобы и осмотр сюда не входят.`

### Шаг 5.4 - График тренда зон

ECharts line:

- series: Оформление, Диагноз, План по КП (avg pct по дням)
- secondary axis/bar: safety_critical count
- click day → yesterday с датой

**Убрать с дефолта:** пятисерийный график «Итог+4 оси». Перенести в `details#month-secondary` как «Классические оси (deep)».

### Шаг 5.5 - Pareto внимания

Горизонтальный bar: top reasons из `attention_pareto` (не все findings).
Клик → filter queue by reason key.

### Шаг 5.6 - Компакт «куда смотреть»

Топ-5 специальностей / врачей по `zone_*_bad_pct` при n≥порогу (как сейчас n≥5).
Клик → documents с doctor/specialty + attention_only.

### Шаг 5.7 - Secondary details

Внутри `<details>`:

1. Чек-лист №55 (avg binary % + top failed points).
2. Методика «Как оценивать»: avg rubric + top-fail criteria table (как сейчас).
3. Heatmap спец×МКБ (оставить).
4. CRM funnel (оставить).
5. Сводный deep overall sparkline.

**Приёмка 5:** первый экран overview без скролла на 1440×900 показывает OPS+Attention+Trend; нет слова «Итоговая» в hero; клики ведут с нужными query.

---

## 6. Yesterday (день) - детально

### Шаг 6.1 - Верх

Оставить pipeline completeness (получено / допущено / оценено) - это data quality дня.
Рядом Attention strip **за вчера** (те же 6 плиток, дневные числа).

### Шаг 6.2 - Заменить «4 индекса»

Блок `renderYesterdayIndices` → `renderYesterdayZones`:

- 3 карточки zone avg + Δ vs предыдущий день / vs тот же weekday 8w
- mini bar: доля bad

### Шаг 6.3 - Таблица действий дня

Расширить `renderYesterdayActions`:

Колонки: Приоритет · Слой · Визит · Пациент · Дата · Врач · Диагноз · Причина · История · PDF.

Слой = `attention_primary` → RU.

### Шаг 6.4 - Findings chart дня

Фильтр series: только attention-eligible / zone-tagged.
Отдельная вкладка «все findings» в details.

**Приёмка 6:** вчерашний экран эксперта (`expert.html`) показывает зоны без поломки; expert по-прежнему только yesterday+reports, но с новыми KPI.

---

## 7. Очередь разбора - детально

### Шаг 7.1 - Колонки

| Колонка | Источник |
|--|--|
| ☐ | select |
| Приоритет | whitelist band |
| Слой | attention_primary |
| Визит / Пациент / Дата | as now |
| Филиал / Врач | as now |
| Диагноз + ICD chip + history chip | as now |
| Оформл. | zone1_band (chip) |
| Диагноз-зона | zone2a_band |
| План КП | zone2b_band (+ kp_status icon) |
| Причина | attention_reason_ru |
| Ответственный / Срок / Статус / PDF | as now |

Колонку «Итог %» - optional, скрыта по умолчанию.

### Шаг 7.2 - Фильтры очереди

- Слой (multi)
- Band bad/weak
- KP: matched/unmatched
- History tier
- «только критические» (как сейчас)
- rubric criterion chip (сохранить)

### Шаг 7.3 - Согласование с whitelist

Не расширять whitelist всеми zone1 weak - иначе шум.
Правило попадания в queue:

1. Safety codes (как v2).
2. zone2a bad (`B_dx_*` или diagnosis score 0).
3. zone2b bad **и** `kp_status=matched`.
4. zone1 bad только если `mo_complete=0` или нет Dx и нет кода (missing_both) - узкий набор.

**Приёмка 7:** выборка 50 строк очереди вручную - ≥90% причин понятны методисту за 3 секунды; нет «не по КП» на оформлении.

---

## 8. Таблица документов - детально

### Шаг 8.1 - Default columns

Визит · Пациент · Дата · Врач/спец · Филиал · Диагноз · ICD · История ·
**Оформл.** · **Диагноз** · **План КП** · Attention · Статус.

### Шаг 8.2 - Column manager (ensureColumnState)

Скрыть по умолчанию: Итог, №55, Полнота проверки, Надёжность.
Пользователь может вернуть.

### Шаг 8.3 - Сортировка

Ключи: `zone1_pct`, `zone2a_pct`, `zone2b_pct`, `attention`, date, (legacy overall).

### Шаг 8.4 - Empty / na display

Band `na` для План КП → chip «КП не сопоставлен» (muted), не «0%».

**Приёмка 8:** screenshot/checklist column set; URL `?columns=` сохраняет выбор.

---

## 9. Разбор случая (drawer) - пошагово

**Канон поверхности:** `2026-08-08-mo-analytics-ui-target-v2.md` §9 (лаконичный русский разбор).
Ниже - техническая раскладка под этот канон; не возвращать длинный «музей метрик».

### Шаг 9.1 - Новая иерархия DOM

```
.case-workspace-grid
  .case-workspace-main
    [1] шапка (визит / пациент / врач / диагноз / метка Риск)
    [2] три оценки (Оформление · Диагноз · План по протоколу)
    [3] Что не так (findings + сжатое заключение ИИ до 3 строк)
    [4] текст МО (только непустые поля)
    [5] история - сводка + раскрытие
    [6] протоколы - одна строка
  .case-workspace-decision (sticky)
    [7] решение методиста (3 вердикта + комментарий + сохранить)
  details.case-workspace-more
    [8] таблица критериев (сначала слабо/плохо)
    [9] полное заключение ИИ / список протоколов
  details.case-workspace-service
    [10] №55 · deep-оси · CRM / пакеты обучения
```

### Шаг 9.2 - Три оценки

Карточки с RU band (в норме / слабо / плохо / протокол не подобран), одна фраза why.
Проценты - мелко или только в «Подробнее».
Метка **Риск** в шапке, не четвёртая «зона-%».
Клик → фильтр «Что не так».

### Шаг 9.3 - Что не так + критерии

На первом экране - findings с русскими метками разделов (не zone1/2a).
Таблица 13 критериев - в `details`, по умолчанию только слабо/плохо/не оценивается; колонки Параметр · Оценка · Что не так.
Запрет copy «не соответствует протоколу» для оформления (жалобы/осмотр).

### Шаг 9.4 - История

Сводка одной строкой + раскрытие списка.
Явно: prior есть/нет для коррекций плана.

### Шаг 9.5 - Протоколы

Одна строка top-1 или «протокол не подобран…». Список - в «Подробнее».

### Шаг 9.6 - Заключение ИИ

Не отдельный hero-блок. До трёх русских строк в «Что не так» или полный текст в «Подробнее».
Не показывать proxy vs LLM на первом экране.

### Шаг 9.7 - Решение методиста

Три русских вердикта: Оформление / Диагноз / План по протоколу + комментарий + сохранить.
CRM/packs - только в «Служебное».
`zones_snapshot` в decision_json при сохранении - ок, UI не раздувать.

**Приёмка 9:** как в ui-target §9.12; gold 3650612 / 3643304 открываются; коррекции без prior = не оценивается; все видимые подписи на русском.

---

## 10. Страницы Аналитика - детально

### Шаг 10.1 - Doctors

Scatter: X = volume, Y = `attention_rate` или `zone2b_bad_pct` (переключатель).
Таблица: n, zone1/2a/2b bad %, safety, delta vs expected (пересчитать expected на zone или оставить deep secondary).

### Шаг 10.2 - Specialties

Boxplot по выбранной зоне (tabs: Оформление / Диагноз / План по КП).
Callouts: спец с высоким bad %.

### Шаг 10.3 - Diagnoses

Treemap: area = volume, color = zone2a_bad_pct (default) или zone2b_bad_pct.
Легенда явно: «цвет = доля плохого диагноза / плана».

### Шаг 10.4 - Safety

Без смены смысла; добавить связку «доля safety среди attention».

### Шаг 10.5 - Data quality

Сюда переехать: coverage, confidence, freshness, dupes, parse_ok.
Убрать эти KPI с overview hero (если ещё остались).

### Шаг 10.6 - Settings

Новый блок «Модель оценки МО»:

- легенда зон + КП да/нет
- пороги band
- toggle «Показывать классические оси deep»
- ссылка на план v2 / catalog

**Приёмка 10:** каждая страница имеет ≥1 график на zone-метриках; data-quality не дублирует hero.

---

## 11. Фильтры, URL, сохранённые виды

### Шаг 11.1 - State

В `mo-app.js` state добавить:

```js
zones: [],          // selected zones
zoneBand: "",       // bad|weak|...
attentionOnly: false,
kpStatus: "",
```

### Шаг 11.2 - URL sync

`?zone=zone2b&zone_band=bad&attention_only=1&kp_status=matched`

### Шаг 11.3 - Saved views

Пресеты:

1. «Внимание: диагноз»
2. «Внимание: план по КП»
3. «Оформление слабо»
4. «Первый контакт + слабый план» (`history_tier=first_contact&zone=zone2b&zone_band=bad`)

**Приёмка 11:** share URL открывает тот же filter set на другом ПК.

---

## 12. Feature flags и миграция

| Flag | Default | Смысл |
|--|--|--|
| `MO_ZONE_SCORES=1` | 1 на GCE после фазы 3 | считать/писать zone_* |
| `MO_ZONE_SCORES_UI=0` | 0 → 1 в фазе UI | новый overview/queue/drawer |
| `MO_ZONE_SCORES_UI_LEGACY=1` | 1 | показывать secondary deep/№55 |

Порядок включения на GCE:

1. Deploy кода с `UI=0`, `SCORES=1` → recompute 7–14 дней.
2. Проверить SQL распределения bands.
3. `UI=1` на staging hostname / том же GCE в off-peak.
4. Смоук разбора + overview.
5. Оставить legacy details ≥1 неделю.

Откат: `MO_ZONE_SCORES_UI=0` без отката склада.

---

## 13. Тесты и калибровка

### Шаг 13.1 - Автотесты

| Набор | Что |
|--|--|
| `test_mo_zone_scores.py` | движок |
| `test_mo_zone_api.py` | overview/cases/detail contracts |
| `test_mo_frontend_structure.py` | строки UI словаря, absence «Итоговая» в hero selectors |
| `test_mo_rubric_mz.py` | requires_protocol / n/a |
| update `test_mo_daily_pipeline.py` | persist columns |

### Шаг 13.2 - Ручная калибровка (методист)

Выборка 30 clinical_visit (10 ok / 10 dx issues / 10 plan issues):

| Вопрос | Цель |
|--|--|
| Зона совпала с ощущением эксперта? | ≥80% |
| Ложное «не по КП» на №127? | 0 |
| Коррекции без prior = n/a? | 100% |
| Очередь полезнее старой? | субъективно да |

Зафиксировать в `docs/reports/YYYY-MM-DD-mo-zones-calibration.md`.

---

## 14. Порядок релизов (GCE only)

### Релиз A - данные (без UI)

Шаги 2–4 + flag UI=0.
Smoke: SQL counts zone bands; detail JSON has `zones`.

### Релиз B - drawer + documents columns

Шаги 8–9 (разбор и таблица) при UI partial flag или том же UI=1 только для drawer.
Предпочтительно: полный UI=1 сразу на overview+queue+drawer, если A стабилен.

### Релиз C - overview/yesterday/queue

Шаги 5–7.

### Релиз D - analytics pages + settings

Шаг 10–11.

### Релиз E - catalog + archive old mental model

Обновить `mo-evaluation-catalog.md`: зоны = канон дашборда; deep/№55 binary = приложения.

Каждый релиз: task branch → PR → merge → `deploy/gcp-app/deploy_to_gce.sh` → `/api/version` + feature smoke.

---

## 15. Чеклист владельца перед стартом кода

Владелец дал старт реализации 2026-08-08 («начинай реализовывать…»). Принятые умолчания:

1. [x] Подписи UI: **Оформление** / **Диагноз** / **План по протоколу** (ui-target-v2)
2. [x] Primary шкала = 0/0.5/1; binary №55 secondary
3. [x] Пороги band 50 / 85 (`config/mo_zone_bands.yaml`)
4. [x] Queue: zone1 bad только узко (mo_complete=0)
5. [x] Expert: те же зоны на «Сегодня»/yesterday (UI-фаза)
6. [x] Recompute окно после деплоя движка: **14 дней**

Старт фазы данных выполнен: ветка `cursor/mo-zone-scores-engine-pc1`.

### Прогресс релизов

| Релиз | Содержание | Статус |
|--|--|--|
| A | шаги 2–4: DDL, движок, API, тесты; UI flag off | **в работе / код готов к PR** |
| B | разбор + колонки таблиц (ui-target §9, §6–7) | дальше |
| C | Сегодня / Период / Очередь | дальше |
| D | Врачи + фильтры URL | дальше |
| E | catalog + handoff | дальше |

---

## Приложение A - Wireframe Overview (текст)

```
+------------------------------------------------------------------+
| Фильтры: период | филиал | спец | врач | [clinical only]         |
+------------------------------------------------------------------+
| OPS:  Записи 1240 | Оценено 1180 | Свежесть OK | Прогноз 2100    |
+------------------------------------------------------------------+
| ВНИМАНИЕ                                                         |
| [Очередь крит 20] [важно 55] [Оформл↓ 6.7%] [Диагноз↓ 3.8%]     |
| [План по КП↓ 9.2%] [Safety 12]                                   |
+------------------------------------------------------------------+
| Тренд зон (линии)                    | Safety count (bars)       |
| ~~~~ zone1 ~~~~ zone2a ~~~~ zone2b   | ▐▌ ▐ ▐▌▌                  |
+------------------------------------------------------------------+
| Топ причин внимания          | Куда смотреть (спец/врачи)        |
| ############## План≠КП  40   | Терапия  18% план↓                |
| ########## Диагноз слаб 28   | Неврология 12% диагноз↓           |
+------------------------------------------------------------------+
| ▸ Подробнее: чек-лист №55 · методика 13 критериев · deep · CRM   |
+------------------------------------------------------------------+
```

## Приложение B - Wireframe Drawer

```
+---------------------------+------------------------------+
| Клинический текст МО      | История пациента             |
| (якоря полей)             | полки + влияние на 2c        |
|                           | КП для плана (suggest)       |
| [Оформл 72 weak]          | LLM: Оформл/Диагноз/План КП  |
| [Диагноз 50 bad]          | Замечания [фильтр зон]       |
| [План КП n/a — нет КП]    | Решение методиста (3 вердикта)|
|                           |                              |
| Таблица 13 критериев      |                              |
| № | параметр | КП? | 0.5  |                              |
| ...                       |                              |
| ▸ Чек-лист №55            |                              |
| ▸ Deep оси / сводный      |                              |
+---------------------------+------------------------------+
```

## Приложение C - Соответствие шагов → файлы

| Шаг | Файлы |
|--|--|
| 2 DDL/upsert | `mo_daily.py`, `recompute_mo_days.py` |
| 3 engine | `mo_zone_scores.py`, `mo_rubric_mz.yaml`, `mo_rubric_mz.py` |
| 4 API | `mo_backend.py`, `rag_server.py` |
| 5–8 UI lists | `mis-kz-quality.html`, `mo-app.js`, `mo-charts.js`, `mo-ui.css` |
| 9 drawer | `mo-app.js` (renderCaseWorkspace) |
| 10 analytics pages | `mo-app.js` page renderers |
| 13 tests | `tests/test_mo_zone_*.py`, frontend structure |
| docs | catalog, this blueprint, handoff reports |

## Приложение D - Что явно не входит в v1 реализации

- Миграция binary №55 → ternary (отдельный план после калибровки).
- Объединение рубрики и №55 в один супербалл.
- Full-document ICD search (запрещено).
- Тикеты очереди по любому D_reg55_*.
- Render production deploy (только GCE).

---

**Конец blueprint.** После заполнения §15 чеклиста начинать Релиз A (движок+склад).
