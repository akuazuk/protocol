# МО Аналитика: один полный контур №55 (section pack) вместо двух старых

Дата: 2026-08-09
Статус: active
Автор: агент (cursor) + владелец
Связанные планы:
- `2026-08-03-mo-rubric-mz-scoring-viz-v1.md` → **superseded** этим планом (shadow «Как оценивать» уходит)
- `2026-08-06-mo-icd-full-document-search-v1.md` (МКБ для пунктов диагноза)
- `2026-07-30-mo-analytics-bi-redesign-v1.md` (BI / warehouse)

Источник нормы:
- Пост. МЗ РБ 21.05.2021 № 55 (прил. 1 разд. V + Инструкция об оценке, п.12-13)
- Пост. МЗ РБ 21.12.2015 № 127 - **помощник доказательств** для пунктов №55 про проведение/оформление МО, не отдельный %

---

## 1. Контекст: что сейчас в МО Аналитике

Сейчас по «№55 / рубрика МЗ» живут **два разных контура** плюс заглушка:

| # | Контур | Где | Шкала | Проблема |
|---|--------|-----|-------|----------|
| A | `evaluate_reg55` + deep axis D «Регуляторика» | `reg55_criteria.py`, `mz_2021_55.json`, warehouse `avg_regulatory`, legacy MIS «№55» | binary pass/fail → % | Чеклист «аналог прил.2», не разд. V прил.1; нет роутинга роли |
| B | Shadow `mo_rubric_mz` «Как оценивать» | `mo_rubric_mz.py`, YAML 13 критериев, Dual KPI «Рубрика МЗ» | 0/0.5/1/n/a | Смесь №127+№55 в одном %; не section pack; `primary: false` |
| C | Карточка месяца «Соответствие №55» | `mo_backend` | **unavailable** | Намеренно не подключена |

Итог для пользователя: dual KPI (deep + рубрика), ось D = старый binary №55, отдельная карточка №55 пустая. Это не формула п.12-13 по роли.

**Цель:** убрать A и B как пользовательские/складские метрики №55 и поставить **один** контур `mo_reg55_section` (ниже). Deep A/B/C (документация, клиника, safety) остаются; ось D переподключается к новому %.

---

## 2. Целевая модель (одна оценка)

### 2.1 Пайплайн случая

```text
MO case
  → router: условие + роль + тип документа + возраст + primary/repeat
  → section_pack из прил.1 разд. V (напр. педиатр 41-43, ВОП 38-40, …)
  → для каждого пункта pack: applicable | n/a(+reason)
  → для applicable: score ∈ {0, 0.5, 1} + evidence
  → pct = sum(scores) / n_applicable * 100          # Инструкция п.12
  → band ∈ {80-100, 55-79.9, ≤54.9}                 # Инструкция п.13
  → findings только по applicable с score < 1
```

### 2.2 Роль №127

№127 **не даёт второго процента** в аналитике.

Он - **evidence helper** для пунктов №55, где текст прямо ссылается на проведение/оформление МО (типично `*.МО по Инструкции…`, согласие, структура записи):

| №55 пункт (пример) | Как помогает №127 |
|--------------------|-------------------|
| педиатр 43.6 / ВОП 40.5 «МО по Инструкции №127» | чеклист структуры: жалобы, анамнез, факторы риска, осмотр, vitals |
| 43.1 / 40.1 формы | наличие блоков заключения по №127 |
| кратность наблюдения (если в pack) | `follow_up_by_icd_chapter` из `mz_2015_127.json` как подсказка интервала (не отдельный KPI) |

В UI у таких пунктов бейдж: `№55 · п.43.6` + `доказательства: №127`.  
В знаменатель % входит только критерий №55.

### 2.3 Что удаляем / не показываем

- Binary `regulatory_compliance_pct` / колонка legacy «№55» как отдельная истина
- Shadow KPI «Рубрика МЗ» / таблица 13 «Как оценивать» / dual deep+rubric
- Заглушку month-reg55 `unavailable` - заменить живым `mo_reg55_section`
- Findings `D_reg55_p0` / `D_reg55_gap` от старого JSON - заменить на `D_reg55_section_*` с якорем пункта разд. V

Deep overall остаётся primary клиническим скорером; **единственная метрика «соответствие №55»** = новый section %.

---

## 3. Метрики

| Метрика | Было | Цель |
|---------|------|------|
| Пользовательских «№55/% рубрика» на overview | 2 (+ unavailable) | **1** (`reg55_section_pct` + band) |
| Источник критериев | App2-analog JSON + sheet 13 | прил.1 разд. V packs |
| Шкала | binary и/или смесь 13 | 0/0.5/1 + n/a вне знаменателя |
| Роутинг роли | нет | педиатр / ВОП / гинеколог / specialist-core / diagnostic packs |
| №127 | отдельный хвост в rubric % | только evidence для связанных пунктов №55 |
| Warehouse | `avg_regulatory` = binary | `avg_reg55_section_pct` + band counts + top failed section points |
| Карточка месяца №55 | unavailable | live pct + градация п.13 |
| Главная «Обзор» (month dashboard) | Dual KPI + shadow-таблица 13 + empty №55 | один KPI №55 + диаграммы/таблица по пунктам pack + drill в очередь |
| Фильтр по критерию | `rubricCriterion` client-only (сервер «после warehouse») | серверный `reg55_point` + `reg55_band` + `reg55_pack` |
| Колонки очереди / «Все случаи» | нет колонки №55 section | `№55 %`, `Градация`, опц. `Pack` |

---

## 4. Дашборд главной МО Аналитики («Обзор» / месяц)

Поверхность сегодня: `frontend/web/methodist/mis-kz-quality.html` `#page-overview` + `mo-app.js` `renderOverview` / `loadOverview` (тянет `/month-report` + `/rubric-summary`).

### 4.1 KPI-ряд (`#month-kpis`)

| Сейчас | Станет |
|--------|--------|
| «Итоговая оценка» (deep) | **оставить** (клинический deep) |
| «Рубрика МЗ» (shadow) | **убрать** |
| карточка `#month-reg55` unavailable | **перенести в KPI-ряд**: «Соответствие №55» = `avg_reg55_section_pct` + подпись band (доля случаев 80-100 / 55-79.9 / ≤54.9) |

Подпись KPI: `п.12 · по применяемым пунктам разд. V` (не «shadow»).  
Рядом или второй строкой KPI: **доля случаев по трём градациям** (n и %), клик по доле = фильтр `reg55_band=…`.

### 4.1a Градации п.13 - канон для всего UI (обязательно)

Единый словарь (код → порог → подпись → цвет). Одни и те же значения в KPI, диаграммах, таблицах, фильтрах, chips.

| Код API `reg55_band` | % по п.12 | Подпись UI (кратко) | Подпись п.13 (tooltip) | Цвет (токен) |
|----------------------|-----------|---------------------|------------------------|--------------|
| `compliant_min` | **80-100** | Соответствует (мин. меры) | соответствует; минимальный комплекс мероприятий | `--mo-band-ok` (зелёный) |
| `compliant_measures` | **55-79,9** | Соответствует (нужны меры) | соответствует; комплекс мер по недостаткам | `--mo-band-warn` (янтарь) |
| `noncompliant` | **≤54,9** | Не соответствует | не соответствует; меры по организации | `--mo-band-bad` (красный) |
| `unscored` | нет applicable / нет оценки | Не оценено | вне градации п.13 | нейтральный/серый |

Правила:
- Градация считается **только** от `reg55_section_pct` (не от deep overall).
- Границы включительно как в НПА: `pct >= 80` → `compliant_min`; `pct >= 55` → `compliant_measures`; иначе при наличии pct → `noncompliant`.
- `unscored` не входит в % долей «по градациям» знаменателя «оценённых по №55»; показывается отдельно.
- Легенда градаций **одна** на странице Обзор (под блоком №55) и переиспользуется в таблицах/диаграммах.

### 4.2 Блок вместо `#month-rubric-mz` → `#month-reg55-section` (span-12)

Одна карточка на главной (заголовок: **«№55 по разделам (прил. 1 разд. V)»**):

1. **Сводка по градациям (обязательный виджет)**:
   - stacked bar или 3 KPI: n и % в `compliant_min` / `compliant_measures` / `noncompliant` (+ отдельно unscored)
   - клик по сегменту → очередь `reg55_band=<code>`
   - подпись среднего % с цветным badge градации **среднего по выборке** (если avg попал в зону; avg сам по себе не «случай», badge информативный)
2. **Таблица top-fail пунктов** (замена rubric-table):
   - колонки: `Пункт` · `Название` · `Pack` · `n применимых` · `доля 0` · `доля 0.5` · `доля слабостей` · `ср. балл`
   - доп. колонки **вклада в градации**: `n случаев band≤54.9` / `n band 55-79.9` среди тех, где этот пункт score&lt;1 (чтобы видеть, какие пункты тянут в «не соответствует»)
   - клик по строке → `reg55_point=…`; клик по ячейке band-count → `reg55_point=…&reg55_band=…`
3. **Диаграмма критериев** (ECharts bar horizontal) **с учётом градаций**:
   - основной режим v1: доля слабостей по пункту
   - stacked-режим (preferred): для каждого пункта стек числа случаев в трёх band среди applicable, где пункт score&lt;1 (цвета из таблицы 4.1a)
   - клик по стеку = point + band
4. **Диаграмма распределения градаций** (donut/stacked %): только три band + drill; обязательна на главной рядом со сводкой или как верх виджета п.1
5. **Heatmap специальность × пункт №55**:
   - цвет = fail_pct или avg score пункта
   - **контур/badge ячейки** или второй layer: доминирующая градацияspecialty на этом пункте (majority band среди случаев с weakness) - v1 достаточно tooltip: `band shares`
   - клик → specialty + point; Shift/вторая серия кликов → + band
6. **Heatmap / stacked: специальность × градация №55** (v1 обязательна проще таблицей или bar):
   - строки = специальности, столбцы = 3 band (n или %), цвет по доле `noncompliant`
   - клик → `specialties=…&reg55_band=…`
7. **Слабости по специальностям** - для каждой спец.: avg №55 %, badge градации avg, доли трёх band, top пункты разд. V

Парето (`#month-pareto`): deep не ломать; в блоке №55 bar по пунктам со стеком band (п.3).

Тренд (`#month-trend-chart`):
- серия **avg №55 %**
- плюс **stacked area / 100% stacked** долей трёх градаций по дням (`band_share_daily`) - чтобы было видно не только среднее, но сдвиг в «не соответствует»
- клик по дню → documents за день; клик по слою band → день + `reg55_band`

### 4.3 Что убрать с главной

- весь `renderMonthRubricMz` / fetch `/rubric-summary`
- Dual KPI «Рубрика МЗ»
- unavailable-заглушку «Соответствие постановлению №55» как отдельную пустую card (слить в KPI + блок 4.2)
- chips `Рубрика МЗ: …` → chips `№55: п.42.3`

---

## 5. Таблицы, фильтры, case detail

### 5.1 Таблицы «Очередь» и «Все случаи»

Новые колонки (in columns manager; в queue default visible):

| Колонка | Поле API | Отображение | Сортировка |
|---------|----------|-------------|------------|
| №55 % | `reg55_section_pct` | число + цвет текста/фона по band | да |
| Градация №55 | `reg55_band` | pill с подписью из §4.1a (не сырой код) | да (порядок: noncompliant → measures → min → unscored) |
| Раздел №55 | `reg55_pack_label` | «Педиатр 41-43» | facet |

Обязательные UX-правила градаций в таблицах:
- ячейка «Градация» - всегда цветной pill (§4.1a), tooltip = полный текст п.13
- строка queue: лёгкий row-tint по `noncompliant` (сильнее) / `compliant_measures` (слабее), чтобы градация читалась без открытия карточки
- сортировка «по приоритету»: учитывать band (noncompliant выше measures) вместе с P0 deep
- в «Причина»: якорь `№55 42.3` + краткий band, если band ≠ `compliant_min`

Legacy MIS «№55» → `reg55_section_pct` + тот же pill градации (если колонка одна - `%` с цветным badge).

Агрегатные таблицы (Врачи / Специальности на соответствующих page):
- колонки: avg №55 %, **доли трёх band** (mini stacked или 3 числа), n unscored
- клик по сегменту band → documents с `reg55_band` + doctor/specialty

### 5.2 Фильтры (toolbar + chips + URL)

| Параметр | Смысл |
|----------|--------|
| `reg55_point` | пункт pack, score &lt; 1 (мульти) |
| `reg55_band` | `compliant_min` \| `compliant_measures` \| `noncompliant` (мульти; **не** строки процентов в URL) |
| `reg55_pack` | id pack |
| `reg55_score_max` | `0` \| `0.5` |

UI:
- `filter-pop` «Градация №55» - три опции + счётчики facets (`n` по band в текущем срезе)
- `filter-pop` «Критерий №55»
- chips: `Градация: Не соответствует`, `№55: п.42.3`
- пресеты кнопок на Обзоре/очереди: «Только ≤54,9%» / «55-79,9%» (ставят `reg55_band`)
- серверный фильтр обязателен в v1

### 5.3 Case detail

- крупный badge градации + pct + полный текст п.13 для этой зоны
- если `compliant_measures` или `noncompliant` - блок «Комплекс мероприятий» (пункты score&lt;1) показывается **первым**
- таблица пунктов pack; focus из `reg55_point`
- №127 только evidence-пометка

---

## 6. Warehouse / API под дашборд и градации

| Артефакт | Содержимое |
|----------|------------|
| case / publish | `reg55_section_pct`, **`reg55_band`** (код §4.1a), `reg55_pack`, `reg55_applicable_n`, per-point rows |
| `fact_mo_daily` | `avg_reg55_section_pct`, **`n_band_compliant_min`**, **`n_band_compliant_measures`**, **`n_band_noncompliant`**, `n_reg55_unscored` |
| `fact_mo_reg55_criterion` | day × point: n_applicable, n_zero, n_half, fail_pct, **опц. n_cases_by_band среди weakness** |
| `GET /month-report` → `reg55_section` | `avg_pct`, **`band_share`**: `{compliant_min, compliant_measures, noncompliant, unscored}`, `top_fail[]`, `by_specialty[]` (с band_share), `heatmap_specialty_point[]`, **`heatmap_specialty_band[]`**, `trend_band_share[]` |
| `GET /cases` | фильтры §5.2; сортировка по band; facets counts по band |
| timeseries | avg pct + **daily band shares** |

---

## 7. Шаги реализации

### Этап 0. Канон данных (без UI)
- [x] Снимок прил.1 разд. V в `data/regulations/mz_2021_55_app1_section_v.md`
- [x] `config/mo_reg55_section_packs.yaml` (packs + applicability + `evidence_from_127`)
- [x] Пометить `mz_2021_55.json` и `mo_rubric_mz.yaml` deprecated

### Этап 1. Движок
- [x] `clinical_knowledge/mo_reg55_section.py` (router, evaluate, band п.13)
- [x] №127 только как evidence helper
- [x] Тест fixture mo_1_test → pediatrist pack, ~70-75%, band `compliant_measures`

### Этап 2. Пайплайн + warehouse + API
- [ ] Publish case fields + daily criterion agg
- [x] deep axis D = `reg55_section.pct` (через `to_reg55_detail_payload`)
- [x] Findings `D_reg55_gap` (якоря пунктов в detail; per-point codes - следующий шаг)
- [x] `month-report.reg55` + `GET /reg55-section-summary` (вместо `/rubric-summary` на Обзоре)
- [ ] Серверные фильтры `reg55_*` в cases/queue

### Этап 3. UI главной + таблицы + фильтры (фокус дашборда и градаций)
- [ ] CSS-токены / легенда трёх band (§4.1a) + helper `bandFromPct`
- [x] KPI-ряд: «Соответствие №55» + доли по градациям в `#month-reg55`; shadow «Рубрика МЗ» убрана с Обзора
- [ ] Виджет распределения градаций (donut/stacked) с drill `reg55_band`
- [x] Карточка top-fail пунктов разд. V (id `#month-rubric-mz` пока сохранён); stacked band - след. шаг
- [ ] Heatmap/таблица специальность × градация; specialty × пункт с tooltip band_share
- [ ] Тренд: avg №55 % + stacked доли band по дням
- [ ] Колонки queue/documents: % + **pill градации** + row-tint; сортировка/priority с band
- [ ] Таблицы врачи/спец.: доли трёх band
- [ ] Toolbar: «Градация №55», «Критерий №55», пресеты band; chips (client focus chip есть)
- [x] Case detail: badge градации + текст п.13 + комплекс мер; shadow rubric block убран
- [x] Удалить `/rubric-summary` из `loadOverview` (заменён на `/reg55-section-summary`)

### Этап 4. Выпил старого
- [ ] Удалить hot path `evaluate_mo_rubric_mz` / binary `evaluate_reg55` из UI и month
- [ ] Legacy колонка «№55» = section pct
- [ ] Тесты UI (`test_mo_frontend_structure` и др.) переписать под новые id
- [ ] План rubric → archived (уже)

### Этап 5. Backfill и калибровка
- [ ] Rescore + пересборка month aggregates
- [ ] Gold packs; smoke главной: KPI + таблица + фильтр + колонка очереди

---

## 8. UI-контракт (приёмка главной + градации)

- [ ] Три градации п.13 везде с одним словарём кодов/цветов/подписей (§4.1a)
- [ ] На «Обзор»: KPI №55, виджет **долей band**, stacked/donut, тренд с band shares
- [ ] Диаграммы критериев и specialty учитывают band (стек / tooltip / drill `reg55_band`)
- [ ] Таблицы очереди/случаев: pill градации + цвет; фильтр и пресеты по band
- [ ] Нет «Рубрика МЗ» и empty unavailable №55
- [ ] Клик критерий/band → очередь с серверным фильтром
- [ ] Case detail: badge + полный смысл п.13; №127 только evidence

---

## 9. Риски

| Риск | Митигация |
|------|-----------|
| Частный педиатр ≠ «участковый» | Pack pediatrist с provenance `nearest_app1_role` |
| Нет блока «любой узкий специалист» | `specialist_amb_core` + явная пометка analog |
| Смена оси D сдвинет deep overall | Версия scorer + backfill + сравнение месяца |
| Heatmap пункт×спец. шумная (разные packs) | группировать по `canonical_point` внутри pack; facet «Pack» на главной |
| Фильтр без warehouse снова client-only | Этап 2 блокирует этап 3 UI drill |
| №127 снова смешают в % | тест-инвариант: evidence_127 не в знаменателе |
| Средний % «зелёный», а много noncompliant | на Обзоре всегда рядом **band_share**, не только avg; тренд - stacked доли |
| Путаница кодов band в URL (`80-100` vs enum) | канон §4.1a: только `compliant_min` / `compliant_measures` / `noncompliant` |

---

## 10. Вне скоупа

- Оргкритерии прил.1 разд. I-IV
- Прил.2 МРЭК/ЭВН
- Замена deep A/B/C и LLM action-judge
- Полный LLM-judge каждого пункта №55 в v1

---

## 11. Одна безопасная следующая команда

```bash
scripts/ops/git_task_start.sh mo-reg55-section-pack --pc=pc1 \
  --branch=cursor/mo-reg55-section-pack-agent1-pc1
```

Первый deliverable: packs YAML + движок + тест на `mo_1_test.pdf`.  
Параллельно можно сверстать `#month-reg55-section` на mock JSON month-report (без выпила rubric, за feature-flag), чтобы дашборд главной не ждал полного backfill.
