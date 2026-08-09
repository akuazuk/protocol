# МО Аналитика: визуальный refresh + таблицы (фильтры/сортировка) v1

Дата: 2026-08-09  
Статус: active (решения §8 зафиксированы; волна A+B в коде)  
Связанные каноны (не ломать):

- `2026-08-08-mo-analytics-ui-target-v2.md` - 6 пунктов меню, зоны RU, лаконичность
- `2026-08-09-mo-dashboards-zones-first-v2.md` - hero = зоны + attention
- `2026-08-03-mo-filter-actions-ui-v1.md` - поиск и мультифильтры с явным «Применить»
- `2026-08-09-mo-settings-refresh-v1.md` - «Справка» внизу сайдбара

Инвентарь текущего UI (main @ 2026-08-09): только Очередь / Все случаи имеют column-sort;
Сегодня / Период / Врачи - таблицы без локальных фильтров и сортировки; палитра приглушённый sage/teal;
один стек шрифта Avenir Next на всё.

---

## 1. Контекст и проблема

Методист работает с **глобальной** панелью фильтров и с **разными** таблицами.
Сейчас:

| Поверхность | Проблема |
|--|--|
| Цвет | Всё в одной зелёно-серой гамме; зоны/статусы слабо отличаются от фона |
| Типографика | Почти всё 11-13px; кнопки, подписи форм, оси диаграмм и ячейки таблицы не разведены по ролям |
| Таблицы | Sort есть только у `#queue-rows` и `#document-rows`; у «Таблица дня», «Куда смотреть», врачи - нет |
| Фильтры таблицы | Нет локального поиска/колоночных фильтров; глобальная панель не заменяет «сузь эту таблицу» |
| Контролы | Primary/secondary есть, но визуальный вес одинаковый; filter-pop не «цветятся» по активному состоянию |

Цель: более **читаемый и цветной** рабочий BI без превращения в «дашборд ради дашборда» и без ломки 6 пунктов меню / языка зон.

---

## 2. Принципы (не обсуждаются без владельца)

1. **Глобальные фильтры** остаются источником истины периода/филиала/врача (explicit apply для multi-select и поиска).
2. **Каждая таблица** получает свой toolbar: локальный поиск + сортировка по колонкам (+ опционально колонный фильтр для 2-4 ключевых полей).
3. Локальные фильтры **не дублируют** глобальные (не второй «филиал»), а сужают уже загруженный/видимый набор: текст строки, статус зоны, «только плохо».
4. Цвет несёт **смысл зон и severity**, не декор. Акцент сильнее, фон чуть живее, но без purple-glow / cream-serif клише.
5. Шрифты: **не менять бренд-стек внезапно** на маркетинговый display; усилить **ролевую шкалу** внутри текущего семейства (+ отдельный tabular для цифр).
6. Экраны и тексты - только русский; зоны: Оформление / Диагноз / План / Риск.

---

## 3. Дизайн-система: роли контролов

### 3.1 Кнопки

| Роль | Класс (цель) | Когда | Визуал | Шрифт |
|--|--|--|--|--|
| Primary action | `.button` | Найти, Применить, Сохранить представление, Открыть разбор | заливка accent, тень soft | 13px / 750 / `--font-ui` |
| Secondary | `.button.secondary` | Очистить, Скачать, Назад, Справка-действия | контур + surface | 13px / 650 |
| Quiet / ghost | `.button.ghost` | иконки theme, закрыть drawer | без заливки | 13px |
| Danger | `.button.danger` | сброс сессии / редкие destructive | soft bad fill | 13px / 700 |
| Compact chip-btn | `.button.compact` | quick period, zone preset, table toolbar | height 32, pill | **12px / 700** |
| Nav | `.nav-button` | 6 пунктов | sidebar ink; current = accent soft | 13px / 650; current 700 |

Правило: в одном toolbar не больше **одной** primary.

### 3.2 Формы и фильтры (глобальная панель)

| Элемент | Поведение (канон) | Оформление |
|--|--|--|
| Поиск случаев | submit «Найти» / «Очистить» | поле 14px; label 11px uppercase muted; кнопка primary рядом |
| Период / compare / даты | apply on change | select/input `--control-height`; активный quick-period = accent soft fill |
| Multi filter-pop | draft + «Применить» | закрытый: нейтральный; **с выбранным** = цветной chip-badge (число) на accent/zone; открытый = raised card |
| Чипы активных фильтров | clear per chip | цвет по типу: период=neutral, врач=accent, статус=severity tint, зона=zone color |
| Zone presets | click applies | 4 кнопки с цветом зоны слева (3px bar) |

### 3.3 Табличный toolbar (новый стандарт на **каждую** таблицу)

Минимальный набор над таблицей:

```text
[ Поиск по таблице … ]   [ Только плохо ▾ ]   колонки↕   [ CSV ]
строк: N · сортировка: Колонка ↓
```

| Контрол | Функция |
|--|--|
| Локальный поиск | filter visible rows by text (врач, МКБ, id, finding) - debounce 200ms, **без** нового API roundtrip если данные уже на клиенте |
| Быстрый фильтр зоны/статуса | chips: все / плохо / слабо / риск |
| Сортировка колонок | click `th.sortable` ↔ asc/desc; индикатор ▲▼; aria-sort |
| Экспорт | CSV текущей **отфильтрованной** видимой таблицы (где уместно) |

Серверная пагинация (Очередь / Все случаи): локальный поиск **дополняет** глобальные фильтры; сортировка остаётся через `sort-by` / `sort-dir` API как сейчас.  
Клиентские таблицы (Сегодня, Период «Куда смотреть», Врачи): sort/filter только на клиенте.

### 3.4 Диаграммы

| Роль | Правило |
|--|--|
| Серии зон | жёстко: Оформление=`--zone-1`, Диагноз=`--zone-2a`, План=`--zone-2b`; Риск = `--bad` / discrete |
| Фон plot | лёгкий tint `--surface-2`, grid `--line-soft` |
| Подписи осей | 11px `--font-ui`, muted; легенда 12px / 600 |
| Tooltip | 13px ink, цветная точка серии |
| «Плохо» bars (Врачи) | насыщеннее текущего; bad/warn/ok не сливаются с teal-фоном |

Не возвращать heatmap/CRM/Pareto в hero.

### 3.5 Таблицы (визуал)

| Часть | Стиль |
|--|--|
| `th` | 11px / 700 uppercase, letter-spacing; фон soft zone-neutral; sticky optional для длинных |
| `td` | 13px / 400; числовые колонки `font-variant-numeric: tabular-nums` |
| Hover row | accent-soft 40% |
| Bad / weak cells | tint background + left border zone/severity color (как drawer zone-cards, но мягче) |
| Status pill | существующие `.status.*` усилить saturation +1 шаг |
| Empty | цветной empty-state (icon tint + 14px текст), не серая дыра |

---

## 4. Цветовая волна («более цветное»)

Без смены бренда (остаёмся в medical teal), но **увеличиваем контраст ролей**:

| Токен | Сейчас (смысл) | Цель |
|--|--|--|
| `--zone-1` Оформление | teal | чуть ярче, стабильный «бренд-зелёный» |
| `--zone-2a` Диагноз | blue muted | более чистый синий (читается отдельно от teal) |
| `--zone-2b` План | khaki | теплее amber-olive, не серо-коричневый |
| `--bad` / P0 | dusty rose | насыщеннее coral-rose для «плохо» |
| `--warn` / P1 | brown | ясный amber |
| `--good` / ok | = accent | оставить; chips ok получить soft green fill |
| Background | flat sage | сохранить glow, добавить **цветные полосы зон** в KPI / attention tiles (уже 3px → 4px + soft fill) |
| Charts | 5 близких hues | zone triad + warn/bad как 4-5я; не mauve ради mauve |
| Sidebar | dark green | ok; active nav = цветная полоска слева |

Dark theme: те же роли, +20% lightness на fills, не неоновый glow.

**Запрет:** purple-indigo градиенты, cream+#terracotta «AI landing», emoji в chrome.

---

## 5. Типографика (роли, не новый бренд)

Оставляем `"Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif` как `--font-ui` / `--font-display`.

Добавить роли (CSS vars), без подключения внешних CDN-шрифтов в этой волне:

| Роль | Размер / вес | Где |
|--|--|--|
| `--type-page` | clamp 26-32 / 700 display | H1 страницы |
| `--type-card` | 17 / 700 | H2 карточки |
| `--type-kpi` | clamp 24-30 / 750 tabular | числа KPI |
| `--type-body` | 14.5 / 400 | описания |
| `--type-control` | 13 / 650 | inputs, buttons |
| `--type-label` | 11 / 700 uppercase | labels фильтров, th |
| `--type-table` | 13 / 400 | td |
| `--type-chart` | 11-12 / 500 | оси, леген |
| `--type-meta` | 12 / 500 | card-sub, chips |

Опционально волна 2 (отдельное решение): `--font-mono-data` системный ui-monospace только для visit_id / МКБ кодов.

---

## 6. Карта таблиц: что именно получить

| Таблица | Страница | Sort | Local filter | Export |
|--|--|--|--|--|
| Таблица дня `#yesterday-action-rows` | Сегодня | колонки: приоритет, врач, зоны, статус | поиск + «только плохо» | CSV optional |
| Kind rows `#yesterday-kind-rows` | Сегодня / Подробнее | да | поиск | нет |
| Куда смотреть `#month-look-where` | Период | да (сейчас fixed) | поиск + зона | CSV optional |
| №55 / specialty tables в `#month-rubric-mz` | Период / Подробнее | да | поиск | нет |
| Очередь `#queue-rows` | Очередь | уже есть → унифицировать toolbar UI | + локальный поиск поверх page | уже CSV |
| Все случаи `#document-rows` | Все случаи | уже есть → тот же toolbar pattern | + локальный поиск page | уже CSV |
| Врачи `#doctor-rows` | Врачи | да | поиск + «есть плохо» | CSV optional |

Общий helper в `mo-app.js`: `enhanceTable(root, { columns, clientSide })` - один API для sort indicators + local filter input, чтобы не копипастить 7 раз.

---

## 7. Волны исполнения (после согласия)

### Волна A - фундамент (1 PR, низкий риск)

1. Токены цвета зон/severity + type roles в `mo-tokens.css`
2. Стили кнопок / filter-pop active / table toolbar / status pills в `mo-ui.css`
3. Без смены разметки страниц, кроме классов-обёрток

### Волна B - table chrome на всех таблицах (1 PR)

1. Helper `enhanceTable` + разметка toolbar
2. Клиентский sort/filter для Сегодня / Период / Врачи
3. Унификация UI sort для Очередь / Документы (поведение API сохранить)
4. Тесты маркеров в `test_mo_ui_phase2.py` (+ точечные unit на helper если вынесен)

### Волна C - диаграммы и KPI color (1 PR)

1. Zone trend / doctor bars: ярче series, легенда, tooltip
2. Attention tiles / KPI: цветные fills по зоне
3. Проверка dark + compact density

Не в scope: новый 7-й пункт меню; AI-расходы; возврат legacy BI в hero; смена auth.

---

## 8. Решение владельца (зафиксировано 2026-08-09)

1. Цвет: **умеренно+**.
2. Локальные фильтры таблиц: **поиск + chips + dropdown по колонкам на всех таблицах**.
3. Шрифты: **только шкала Avenir** (без нового display-файла).
4. Первая волна кода: **только sort/filter** (CSV на Сегодня/Врачи - позже).

---

## 9. Метрики

| Метрика | Было | Цель |
|--|--|--|
| Таблицы с column-sort | 2 | все основные (§6) ≥ 7 |
| Таблицы с local filter/search | 0 | все основные §6 |
| Единый table toolbar pattern | нет | да |
| Type roles в tokens | 0 | ≥ 8 vars |
| Zone/severity contrast step-up | muted | documented tokens + visual smoke |
| Регрессии `test_mo_ui_phase2` | baseline | 0 новых |
| Меню = 6 + Справка в foot | да | да (без изменений IA) |

---

## 10. Риски

| Риск | Митигация |
|--|--|
| Локальный filter vs серверная пагинация путает «почему мало строк» | подпись «из загруженной страницы N» / «по текущей выборке» |
| Яркий цвет = «игрушечность» | цвет только у смысла (зона/статус), chrome остаётся спокойным |
| Конфликт с параллельными PR по `mo-app.js` / HTML | отдельная task-ветка; мелкие волны A→B→C |
| `enhanceTable` на JS-built tables | вызывать после каждого re-render |
| Accessibility | aria-sort, focus rings, не полагаться только на цвет |

---

## 11. Шаги (статус)

1. [x] Инвентарь текущего UI (таблицы, фильтры, токены).
2. [x] План в `docs/plans/` + индекс README.
3. [x] Решения владельца §8.
4. [x] Волна A (tokens/CSS).
5. [x] Волна B (table toolbars + sort/filter + col dropdowns).
6. [ ] Волна C (charts/KPI color) - отдельный PR после A+B.
7. [ ] BUILD_VERSION / PR / smoke на `/methodist/mo`.
