# Дашборды МО: зоны + новые оценки вместо старых BI (v1)

Дата: 2026-08-09
Статус: **active** (канон замены дашбордов)
Владелец волны: отдельная task-ветка после согласования с владельцем
Связанные (не дублировать код, только стыковать):

| План / PR | Роль относительно этого плана |
|--|--|
| `2026-08-08-mo-analytics-ui-target-v2.md` | канон меню и экранов (6 пунктов) - **не расширять** |
| `2026-08-08-mo-analytics-mz-sheet-layers-v2.md` | методика трёх зон |
| `2026-08-08-mo-analytics-implementation-blueprint-v1.md` | склад/API зон (уже в коде) |
| `2026-08-09-mo-case-review-quality-parity-v1.md` | `review_brief` / clinical gaps - **в разборе готово**; сюда - агрегаты |
| `docs/methodist/mo-evaluation-catalog.md` | словарь метрик для каждой колонки/KPI |
| **PR #87** `mo-reg55-section-pack` | новый scorer №55 (прил.1 разд.V) - **после merge**, UI только как secondary |
| Draft **PR #77** | старые docs layers; не тащить UI из него поверх ui-target-v2 |
| `2026-08-09-auth-accounts-unify-v1.md` | P0 SSO сделан; P1 login - **не трогать** в этой волне |
| `2026-08-08-mo-action-queue-precise-signals-v2.md` | очередь без №55-тикетов - сохранить |
| `2026-08-07-by-home-gcp-llm-split-v1.md` | LLM/narrative только GCE; не на hero |

Параллельно на 2026-08-09: dirty checkout'ы на других ПК, open PR #87 (scorer),
auth P0 уже в `main`. Этот план - **docs-first** согласование дашбордов, чтобы
агенты не переписали hero под dual №55/deep и не конфликтовали с #87.

---

## 1. Цель

Заменить «старый BI» (deep overall, binary №55, shadow-рубрика, heatmap МКБ,
Pareto, funnel, coverage/confidence как главные колонки) на дашборды, которые
отвечают на три вопроса методиста:

1. Что срочно разобрать сегодня?
2. Где провал: **Оформление / Диагноз / План / Риск**?
3. Что сказать врачу (из `review_brief` / gaps) - в разборе, не на overview.

Не цель: ещё один «магический %»; раздуть меню; сделать №55 главным KPI дня.

---

## 2. Карта оценок → поверхности UI

Источник истины по полям: `mo-evaluation-catalog.md`.
Правило: **hero = зоны**; остальное - secondary / «Подробнее» / разбор.

```text
L1  Зоны zone1 / zone2a / zone2b + attention     → hero всех экранов
L2  МКБ chip (icd_visit_status)                  → колонка/чип; не зона «Диагноз»
L3  Clinical gaps (shadow)                       → разбор + позже агрегат «Клиника»
L4  Protocol suggest / kp_status                 → фильтр + колонка План
L5  review_brief / doctor_feedback               → только разбор случая
L6  LLM narrative / action-judge                 → не hero; GCE; badge «ИИ»
R55 mo_reg55_section (PR #87)                    → secondary блок «Период»
Deep / binary №55 / rubric_mz / coverage         → «Подробнее» или удалить с hero
```

### 2.1 Что убираем с главных экранов

| Старое (сейчас ещё в `#page-overview` / `#page-yesterday`) | Куда |
|--|--|
| KPI «Итоговая оценка» / deep overall как главная | скрыть или secondary в «Подробнее» |
| Dual KPI «Рубрика МЗ» + `/rubric-summary` таблица 13 | убрать с hero; критерии - в разборе |
| Карточка «Соответствие №55» unavailable / binary % | заменить только после merge #87 → secondary |
| Heatmap «Специальность × глава МКБ» | убрать с Период (или deep-link в «Подробнее») |
| Pareto замечаний + воронка CRM | убрать с Период |
| Scatter врачей / boxplot специальностей | не возвращать в меню |
| Колонки Полнота / Надёжность / Итог % по умолчанию | только column picker |
| Отдельные страницы Safety / Специальности / Диагнозы | остаются `hidden` |

### 2.2 Что ставим вместо

| Поверхность | Primary widgets |
|--|--|
| **Сегодня** | 5 плиток внимания (очередь + zone*_bad) · таблица дня · тренд 3 зон 14д |
| **Период** | те же 5 плиток · тренд зон · «Куда смотреть» (врач/спец × zone bad %) · `<details>` №55 section (после #87) · `<details>` МКБ/клиника |
| **Очередь** | whitelist + колонки зон (band, не %) · причина · история chip |
| **Все случаи** | зоны + МКБ chip + kp_status + история; без deep по умолчанию |
| **Врачи** | zone*_bad_pct × 3 · в очереди · график полос по выбранной зоне |
| **Разбор** | уже: зоны → **Итог разбора** → findings → критерии (quality-parity) |

---

## 3. Спека экранов (замена старых)

Меню фиксировано ui-target-v2: Сегодня · Период · Очередь · Все случаи · Врачи · Отчёты.

### 3.1 Сегодня (замена «Вчера» + мусора)

**Оставить / добить:**

1. Строка фактов: записей · оценено · свежесть (`data_through`).
2. 5 плиток → drill в Очередь / Все случаи с `zone` + `zone_band=bad`.
3. Таблица дня (≤50): Приоритет · Раздел · Визит · Пациент · Дата · Врач · Dx · Причина · История.
4. Один график: 3 линии `zone1_avg` / `zone2a_avg` / `zone2b_avg` за 14 дней.

**Убрать с экрана:** pipeline completeness как hero, 4 deep-индекса, flow-chart,
P0/P1 action lists дублирующие очередь, findings Pareto дня.

**Новое (опц. v1.1):** chip-счётчик `clinical_gaps` за день (n случаев с ≥1 shadow gap)
- клик → Все случаи `finding_codes=B_complaint_exam_mismatch,...` (после калибровки precision).

### 3.2 Период (замена старого Overview BI)

**Hero (всегда открыт):**

1. Те же 5 плиток за период.
2. Тренд трёх зон.
3. Таблица «Куда смотреть»: спец/врач × zone*_bad_pct + кнопка «Открыть».

**`<details>` Подробнее - методика №55** (только после merge #87 + warehouse):

- один KPI `avg_reg55_section_pct` + доли градаций п.13 (`compliant_min` /
  `compliant_measures` / `noncompliant`);
- top-fail пункты pack;
- **не** dual с рубрикой МЗ; **не** на первом экране «Сегодня».

Согласование с PR #87: секции 4.x того плана переносим **сюда как secondary**,
а не как замену zone-hero (ui-target запрещает №55 как главный балл дня).

**`<details>` Подробнее - МКБ и клиника** (после агрегатов warehouse):

| Виджет | Поле | Drill |
|--|--|--|
| Доли chip МКБ | `icd_visit_status` counts | `icd_visit_status=` |
| Топ clinical gap codes | counts shadow findings | `finding_codes=` |
| План без КП | `zone2b_kp_status=unmatched` n/% | `kp_status=unmatched` |

**Не возвращать на Период:** heatmap МКБ×спец, funnel CRM, Pareto как обязательные.

### 3.3 Очередь

Уже: precise whitelist.
Добить колонки: Оформление / Диагноз / План = **band RU** (не %); chip истории;
причина на русском; фильтры `zone`, `zone_band`, `kp_status`, `attention_only`.

Запрет: тикеты из binary №55 / weak МКБ P3 / unmatched KP как «плохо план».

### 3.4 Все случаи

Default columns = ui-target §7.
Добавить (видимые): chip МКБ (`ok` / `нет Dx` / `не в МКБ` / `слабо`) с подписью
«не зона Диагноз».
Опц. колонка «Клиника» = n shadow gaps или «есть/нет» после калибровки.

### 3.5 Врачи

Таблица ui-target §8 без scatter.
Добавить сортировку по выбранной зоне; клик → documents с `zone_band=bad`.
Secondary (column picker): avg №55 section после #87.

### 3.6 Разбор

Уже закрыто quality-parity (R1-R5).
В этой волне дашбордов **не** менять drawer, кроме drill-ссылок из новых фильтров.

---

## 4. Зависимости и очередь с другими вкладками

### 4.1 Жёсткий порядок (не параллелить на одних файлах)

| Фаза | Содержание | Блокируется / ждёт |
|--|--|--|
| **D0** | Этот план + индекс README; согласование владельца | - |
| **D1** | Чистка Период/Сегодня: убрать heatmap/pareto/funnel/рубрику с hero; dual KPI | не трогать `mo_reg55_section*` из #87 |
| **D2** | Добить aggregations overview: zone trends + look-where уже есть - проверить empty/legacy KPI | warehouse zones уже есть |
| **D3** | Merge **PR #87** (scorer only) → GCE; warehouse поля `reg55_section_*` | не UI №55 до D3 |
| **D4** | Secondary блок №55 на Период (из §3.2) + фильтры `reg55_band/point` | D3; не менять action-queue whitelist |
| **D5** | Агрегаты МКБ chip + clinical_gaps на Период | quality-parity в проде; precision gaps ≥0.7 |
| **D6** | Column defaults / URL facets / smoke методиста | auth P1 не смешивать |

### 4.2 Что делают другие вкладки - не перехватывать

| Работа | Файлы / зона | Наш план |
|--|--|--|
| PR #87 №55 section scorer | `mo_reg55_section.py`, YAML packs, plan reg55 | ждём merge; UI только D4 |
| Auth unify P1 | `mo_expert_auth`, login pages | не трогать |
| GCP LLM / night | `deploy/gcp-llm/*`, grades | narrative остаётся opt-in; не KPI |
| ICD pipeline / name-match | `mo_icd_*` | только chip на таблицах |
| Action queue v2 | `mo_action_queue_select` | не расширять №55 |
| Dirty local Protocol (PDF KP, plans README) | чужой checkout | только clean worktree от `origin/main` |

### 4.3 Конфликт двух «канонов» №55 - решение владельца (зафиксировано здесь)

- **Операционный канон дня** = зоны (ui-target-v2).
- **Нормативный канон соответствия пост. №55** = `mo_reg55_section` (план #87).
- Оба живут; №55 **не** вытесняет зоны с «Сегодня».
- Shadow «Рубрика МЗ» как отдельный % на overview **снимаем** после D1/D4
  (критерии 0/0.5/1 остаются в разборе через zones.criteria / №55 pack points).

---

## 5. Склад / API (минимум для дашбордов)

Уже есть: `zone*_pct/band`, `attention_*`, filters в `/cases`, overview attention.

| Нужно | Когда | Примечание |
|--|--|--|
| `reg55_section_pct`, `reg55_band`, `reg55_pack`, top points | D3-D4 | из #87 + recompute дней |
| Агрегат `icd_visit_status` counts в overview | D5 | без LLM |
| Агрегат clinical_gap code counts (shadow) | D5 | flag `MO_CLINICAL_GAPS` |
| Не писать LLM % в warehouse zones | всегда | catalog §Z |

Recompute: после D3 минимум дни эталона + «вчера»; полный август - отдельным ops.

---

## 6. Метрики приёмки

| Метрика | Было (09.08) | Цель |
|--|--|--|
| Hero KPI на Период = deep / рубрика / empty №55 | да | нет; hero = зоны + очередь |
| Heatmap+Pareto+funnel на Период по умолчанию | да | нет (или только в свёрнутом legacy) |
| Плитки zone*_bad кликабельны | частично | 100% drill с URL-фильтрами |
| Разбор: `review_brief.available` | да (prod) | не регрессировать |
| Пользовательских «№55 %» на overview | 0 live / dual shadow | 1 secondary (`reg55_section`) после D4 |
| Очередь создаёт тикет из binary №55 | нет (v2) | сохранить нет |
| Время до «понял провал зоны» на Сегодня | путаница осей | ≤1 мин (опрос) |

---

## 7. Риски

| Риск | Митигация |
|--|--|
| #87 и D1 правят один `mo-app.js` / overview HTML | D1 сначала; #87 merge без UI; D4 отдельный PR |
| №55 section UI раздует Период как старый BI | только `<details>`; лимит виджетов §3.2 |
| Clinical gaps шумят в агрегатах | shadow; агрегат после precision ≥0.7 |
| Draft #77 снова тащит англ. Safety/Overview | не merge UI из #77; канон = ui-target-v2 |
| Грязный checkout на Mac | только `git_task_start` worktree |
| Auth P1 ломает smoke дашбордов | не трогать auth в D1-D5 |

---

## 8. Шаги статуса

- [x] Зафиксировать карту оценок → экраны и конфликт зон vs №55
- [x] Учесть open PR #87, draft #77, auth, quality-parity, queue, GCP
- [ ] D0: согласование владельца (меню 6 + №55 secondary)
- [ ] D1: PR чистки hero Период/Сегодня (без №55 UI)
- [ ] D2: smoke drill плиток зон
- [ ] D3: merge #87 + warehouse + GCE recompute
- [ ] D4: secondary №55 на Период
- [ ] D5: агрегаты МКБ + clinical gaps
- [ ] D6: опрос методиста / handoff

---

## 9. Файлы волны (ожидаемые владельцы)

| Фаза | Каталоги | Не пересекать с |
|--|--|--|
| D1-D2 | `frontend/web/methodist/mis-kz-quality.html`, `frontend/web/shared/mo-app.js`, `mo-ui.css` | #87 packs YAML |
| D3 | warehouse / `mo_backend` aggregates (после #87) | auth |
| D4-D5 | `mo_backend` overview payload + mo-app details | `mo_case_review_brief` (stable) |
| Docs | этот файл, catalog § UI screens, handoff | чужие active plans без archive |

---

## 10. Одна безопасная следующая команда

```bash
# После согласования D0 - отдельный clean worktree (не dirty Protocol):
scripts/ops/git_task_start.sh mo-dashboard-hero-cleanup --pc=pc1 \
  --branch=cursor/mo-dashboard-hero-cleanup-pc1
# D1: убрать heatmap/pareto/funnel/рубрику с hero Период; не трогать PR #87 файлы
```

Пока #87 open: **не** начинать D4. Пока auth P1 не стартовал в другой вкладке -
можно D1. Drawer / review_brief - freeze.
