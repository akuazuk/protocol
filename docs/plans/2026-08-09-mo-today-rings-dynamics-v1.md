# МО Аналитика: «Сегодня» - кольца оценок + динамика по периоду (v1)

Дата: 2026-08-09  
Статус: active (план до кода; ждать решений §9)  
Связанные:

- `2026-08-09-mo-dashboards-zones-first-v2.md` - канон меню/зон (уточняем: кольца №55 на Сегодня - осознанное расширение)
- `2026-08-09-mo-reg55-section-pack-v1.md` - №55 section-pack, band_share, тренд градаций
- `2026-08-08-mo-analytics-ui-target-v2.md` - 6 пунктов меню, RU
- `docs/methodist/mo-evaluation-catalog.md` - полный каталог оценок
- `2026-08-09-mo-analytics-visual-refresh-v1.md` - цвет/таблицы (уже в проде #100)

База анализа: `origin/main` @ `403b9438`, handoff D7 (`n_evaluated` август ~2242 / день 06.08 ~441).

---

## 1. Диагноз: почему «Сегодня» неинформативен

| Симптом | Корневая причина (код/данные) |
|--|--|
| График «Тренд 14 дней» почти плоский / одна точка | `build_daily_report` вызывает attention с `date_from=date_to=день` → `zone_trends` = **1 день**, хотя UI пишет «14 дней» (`mo_backend.py` ~2069-2071, `mo-app.js` `renderZoneTrendHost` slice -14) |
| Выбор периода сверху (7д / месяц) не меняет Сегодня | `loadYesterday` берёт только «рабочий день» (вчера / `data_through`); `period` из context-bar **игнорируется** |
| Плитки «Критично / Важно в очереди» = «-» | API всегда отдаёт `queue_critical: null`, `queue_important: null` |
| Нет кольцевых диаграмм | В приложении нет ECharts `pie`/`doughnut` для зон/№55; на Сегодня нет band_share |
| №55 и МКБ «где-то в Периоде» | zones-first сознательно убрал №55 с hero; методисту не хватает распределения оценок на стартовом экране |
| Мало свежих дней | Handoff: extract обрывается на `data_through` (напр. 2026-08-06); дни без inbound нельзя «дооценить» без выгрузки МИС. Уже загруженные дни при `layer_engine` + recompute дают зоны |

Итог: проблема и в **продуктовой компоновке**, и в **баге окна тренда**, и в **дырах API** (очередь), и частично в **покрытии данных** (нет новых дней ≠ нет оценок за имеющиеся).

---

## 2. Каталог оценок: что существует и что кладём на дашборд

Полный перечень - `mo-evaluation-catalog.md`. Ниже - решение «на кольца / на линии / не на Сегодня».

### 2.1 Hero «Сегодня» (обязательно)

| ID | RU | Тип виджета | Окно данных |
|--|--|--|--|
| `zone1` band share | Оформление | **кольцо** ok / слабо / плохо (+ na серым) | период из context-bar |
| `zone2a` band share | Диагноз | кольцо | период |
| `zone2b` band share | План по протоколу | кольцо; na = «протокол не подобран», не «плохо» | период |
| `zone*_avg` | Средние % зон | **линия/area** по дням (или неделям при month) | lookback (см. §3) |
| `reg55_section` band share | №55 градации п.13 | **кольцо** 80-100 / 55-79.9 / ≤54.9 (+ unscored) | период |
| `reg55_section_avg` | Средний % №55 | линия на том же графике динамики (пунктир / 2-я ось опционально) | lookback |
| attention counts | плохо по зонам + очередь | плитки (уже есть; починить очередь) | **рабочий день** таблицы |
| coverage strip | записи / оценено / % покрытия / data_through | одна строка KPI | рабочий день + период в подписи |

### 2.2 Secondary («Подробнее» на Сегодня или только Период)

| ID | RU | Виджет | Куда |
|--|--|--|--|
| `icd_visit_status` | Чип МКБ | маленькое кольцо / stacked chips | Подробнее Сегодня **или** Период |
| `kp_unmatched` | План без КП | KPI-число | Период (уже) + опционально meta под кольцом zone2b |
| `safety_critical` | Риск (есть/нет) | не %, а count в meta плитки / линия safety_critical на динамике | динамика optional |
| deep A-D / overall | служебные | не hero | Справка / case detail |
| LLM night / judge | калибровка | не hero | coverage advisory в полноте |
| heatmap / Pareto / CRM funnel | legacy | не возвращать на Сегодня | остаётся hidden |

### 2.3 Что не смешивать

- **Зона «Диагноз» ≠ чип МКБ** - разные кольца/подписи.
- **№55 ≠ зона «Оформление»** - №55 «по инструкции» роли; зона1 - полнота №127.
- **zone2b bad ≠ unmatched** - unmatched не красить как плохо на кольце (отдельный сегмент `na` / подпись).

---

## 3. Продуктовая модель экрана «Сегодня»

### 3.1 Два окна времени (важно)

| Окно | Что показывает | Источник периода |
|--|--|--|
| **A. Рабочий день** | Таблица дня, плитки очереди/плохо «за день разбора», свежесть | вчера Minsк → fallback `data_through`; override только custom date |
| **B. Окно аналитики** | Кольца + динамика | **глобальный `#period`**: yesterday / 7d / month / custom |

Подзаголовок страницы явно:

```text
Рабочий день: 2026-08-06 · в кольцах и динамике: последние 7 дней
```

Так период сверху наконец «оживает» на Сегодня, не ломая смысл таблицы «кого разобрать сегодня».

### 3.2 Целевая компоновка (сверху вниз)

```text
Сегодня
├─ page-head: рабочий день + freshness + подпись окна аналитики
├─ strip KPI: получено · оценено · покрытие % · data_through
├─ attention tiles (5): критично · важно · zone1/2a/2b плохо  [окно A]
├─ блок «Распределение оценок» [окно B]
│    ├─ 3 кольца зон (ok/weak/bad[/na])
│    └─ 1 кольцо №55 (3 градации [+ unscored])
├─ блок «Динамика» [окно B lookback]
│    └─ линии: zone1/2a/2b avg; опционально reg55 avg
│       granularity: день (yesterday/7d) · неделя (month, если дней > 21)
├─ Таблица дня [окно A] + table chrome
└─ details «Подробнее»: полнота/воронка · опц. кольцо МКБ · ссылка в Период
```

Клики:

- сегмент кольца зоны → Все случаи с `zone` + `zone_band`
- сегмент №55 → Очередь/случаи с `reg55_band`
- точка на динамике → переключить рабочий день / custom date на этот день + перезагрузка таблицы

### 3.3 Правила колец (визуал)

- Тип: ECharts **doughnut**, hole 55-62%, центр = средний % или «N оценено».
- Цвета: `--good` / `--warn` / `--bad` / muted для na|unscored (согласовано с visual-refresh умеренно+).
- Легенда под кольцом на русском: «в норме / слабо / плохо».
- Пустые данные: кольцо-заглушка + текст «нет оценок за окно» + CTA «пересчитать» (для admin) / «нет выгрузки».

### 3.4 Правила динамики

| `#period` | Агрегат колец | Ось X динамики | Точек min |
|--|--|--|--|
| yesterday | 1 день (= рабочий, если совпадает) | последние **14 календарных** дней с данными | до 14 |
| 7d | сумма/доля за 7 дней | **7 дней** по одному | 7 |
| month | за месяц | **дни** месяца; если >21 точка → bucket **ISO-недели** | ≥4 |
| custom | date_from..date_to | дни; при >21 → недели | ≥2 |

Подпись графика всегда совпадает с реальным `zone_trends[].date` (чинить баг «14 дней»).

---

## 4. API и склад (что добавить)

### 4.1 Расширить attention / новый контракт `dashboard_scorecards`

Предпочтительно **один** payload для колец+динамики, чтобы Сегодня не дёргал 5 endpoint'ов:

`GET /api/methodist/mo/score-dashboard?period=…&date_from&date_to&…filters`

Ответ (черновик):

```json
{
  "ok": true,
  "window": {"date_from": "…", "date_to": "…", "granularity": "day|week"},
  "coverage": {"source": 0, "evaluated": 0, "coverage_pct": 0, "target_pct": 99},
  "zones": {
    "zone1": {"avg_pct": 72.1, "bands": {"ok": 40, "weak": 30, "bad": 20, "na": 0}, "n": 90},
    "zone2a": {"…": "…"},
    "zone2b": {"…": "…", "kp_unmatched_n": 12}
  },
  "reg55": {
    "available": true,
    "avg_pct": 78.0,
    "band_share": {
      "compliant_min": {"n": 50, "pct": 55.0},
      "compliant_measures": {"n": 25, "pct": 28.0},
      "noncompliant": {"n": 10, "pct": 11.0},
      "unscored": {"n": 5, "pct": 6.0}
    },
    "sample_n": 90
  },
  "trends": [
    {
      "bucket": "2026-08-01",
      "zone1_avg": 70, "zone2a_avg": 65, "zone2b_avg": 60,
      "zone1_bad_pct": 22, "reg55_avg": 77,
      "n_evaluated": 80
    }
  ],
  "queue": {"critical": 12, "important": 34}
}
```

Реализация склада:

- bands зон: `GROUP BY` / `SUM(CASE zone*_band=…)` по `fact_mo_case` (как сейчас bad, расширить ok/weak/na);
- trends: тот же SQL что `zone_trends`, но **lookback** отдельным параметром `trend_from`/`trend_to` (для yesterday-period: to=day, from=day-13);
- reg55: переиспользовать `mo_reg55_section.aggregate_*` / уже существующий `/reg55-section-summary`;
- queue: посчитать из action-queue / CRM статусов за **рабочий день** (не null).

Краткий путь без нового route: расширить `daily-report.attention` + отдельный `GET /zone-trends?date_from&date_to` и дергать с клиента. Минус - больше roundtrips. **Рекомендация: один `score-dashboard`.**

### 4.2 Починить daily attention trend

Даже до нового UI:

```text
attention aggregates: date_from=date_to=working_day
zone_trends:          date_from=working_day-13 .. working_day
queue_*:              counts for working_day
```

Это закрывает «нет динамики» при любом периоде = yesterday.

---

## 5. Покрытие данных: «мало оценённых» - чеклист

### 5.1 Развести две причины

| Причина | Признак | Действие |
|--|--|--|
| **Нет extract/inbound** за день | нет CSV/secure_cases, freshness не двигается | выгрузка МИС → publish; recompute бесполезен |
| **Есть случаи, нет зон/№55** | `fact_mo_case` есть, `layer_engine` null / нет `reg55_section_*` | `recompute_mo_days.py` на диапазоне; проверить флаги движка |
| **Coverage < 99%** | funnel.evaluated / eligible | добить scoring errors; смотреть `scoring_coverage` в completeness |
| **Малая выборка** | n < `MO_SUPPRESSION_N` (5) | UI: «мало данных», не пустые кольца как 0% |

### 5.2 Операционный прогон (после согласования плана)

На GCE `protocol-app` (не Mac для Gemini; recompute склада - ок на GCE):

```bash
# 1) свежесть
curl -sS localhost:8000/api/methodist/mo/freshness

# 2) пересчёт всех имеющихся дней склада (пример)
docker exec protocol-web python3 scripts/recompute_mo_days.py \
  --from 2026-08-01 --to 2026-08-09

# 3) проверить
# overview?month=2026-08 → n_evaluated, zone_trends length
# daily-report?date=<data_through> → attention.zone_trends length ≥ 2
# reg55-section-summary?date_from&date_to → band_share
```

Цель данных до UI-merge:

- за каждый день с extract: `coverage_pct ≥ 99` среди eligible **или** явный banner почему нет;
- `zone_trends` за 14д lookback: ≥ числа дней с `n_evaluated>0` в окне;
- `reg55.band_share` available на том же окне, что кольца.

LLM night **не блокирует** кольца зон/№55 (зоны из warehouse; LLM - advisory в полноте).

---

## 6. Волны реализации

### Волна 0 - данные и API-фиксы (1 PR, без большого UI)

1. Lookback `zone_trends` в daily attention (14 дней).
2. Заполнить `queue_critical` / `queue_important`.
3. Endpoint или расширение: `bands` ok/weak/bad/na по зонам + reg55 band_share в удобном месте.
4. Smoke на GCE + `recompute` имеющегося диапазона.
5. Тесты backend на multi-day trends и band counts.

### Волна 1 - UI колец и динамики на Сегодня (1 PR)

1. Разметка: strip KPI, host'ы `#yesterday-score-rings`, `#yesterday-score-dynamics`.
2. `renderScoreRings` (4 doughnut) + `renderScoreDynamics` (линии; granularity day/week).
3. Привязка окна B к `#period` / `query()`; подпись «в кольцах: …».
4. Клики сегментов → фильтры случаев.
5. Пустые/loading/error состояния на русском.
6. Статические + API contract tests; visual smoke dark/compact.

### Волна 2 - Период parity + МКБ (опционально следом)

1. Те же кольца на Период (вместо/рядом с текущими KPI №55 buttons).
2. Stacked area долей №55 по дням (из reg55 plan `trend_band_share`).
3. Мини-кольцо МКБ в «Подробнее».

Не в scope: возврат heatmap/Pareto/CRM на hero; AI-расходы; 7-й пункт меню.

---

## 7. Метрики успеха

| Метрика | Было | Цель |
|--|--|--|
| Точек в «тренде» на Сегодня при наличии 7+ дней данных | 1 | ≥7 (yesterday lookback 14) |
| Кольца зон на Сегодня | 0 | 3 |
| Кольцо №55 на Сегодня | 0 | 1 |
| Плитки очереди с числами | «-» | числа ≥0 |
| `#period` влияет на кольца/динамику | нет | да |
| Подпись графика = фактическое окно | нет | да |
| Coverage eligible на имеющихся днях | ? | ≥99% или явный banner |
| Регрессии зон-first меню (6 пунктов) | ok | ok |

---

## 8. Риски

| Риск | Митигация |
|--|--|
| Перегруз hero (кольца + линии + таблица) | кольца в один ряд 4×; динамика одна; table ниже fold на laptop |
| Конфликт с zones-first («№55 не на Сегодня») | это **осознанное** расширение по запросу владельца; №55 - одно кольцо, не чеклист |
| Путаница окно A vs B | явная подпись в page-head |
| Тяжёлый SQL на month | индексы `visit_date`; агрегаты по дням уже в warehouse; при необходимости materialize daily band_share в `fact_mo_daily` |
| Нет inbound за новые дни | UI честно показывает data_through; ops-трек extract отдельно |
| Параллельные PR по `mo-app.js` / `mo_backend.py` | отдельная task-ветка; волны 0→1 |

---

## 9. Решения владельца (зафиксировано 2026-08-09)

1. №55 кольцо на Сегодня hero: **да**.
2. Окно аналитики: **всегда = `#period`**.
3. Динамика при `period=month`: **дни** (не недели).
4. МКБ-кольцо: **волна 2**.
5. Волна 0 (API + recompute на GCE): **стартовать сразу**.

---

## 10. Шаги

1. [x] Инвентарь UI/API/оценок и баг тренда.
2. [x] План в `docs/plans/` + индекс README.
3. [x] Ответы на §9.
4. [x] Волна 0 API в коде (`score-dashboard`, lookback, bands, queue counts) - PR #104.
5. [ ] Волна 0 recompute на GCE после deploy.
6. [ ] Волна 1: кольца + динамика на Сегодня.
7. [ ] Волна 2: Период parity / МКБ / stacked №55.
8. [ ] Smoke `/methodist/mo` + GCE `/score-dashboard`.

