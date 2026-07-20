# Search flow redesign v1

**Проект:** Protocol  
**Дата:** июль 2026  
**Сборка:** r54+  
**Статус:** внедрено в UI

## Цель

Сделать поиск протоколов удобным на приёме: крупные блоки выбора, явная навигация вперёд/назад, запоминание шагов, 4 макроэкрана вместо 8 мелких шагов без визуального веса.

## Макроэкраны (A-D)

| Фаза | UI id | Backend шаги | Содержание |
|------|-------|--------------|------------|
| A Запрос | `query` | 0 | Поле, быстрый старт, inline-аудитория |
| B Уточнение | `refine` | 1-3 | Аудитория, МКБ-10, рубрика |
| C Протокол | `protocols` | 4 | Список PDF |
| D На приёме | `clinical` | 5-7 | Сводка, нозология, раздел, цитата |

## Файлы

- `search-flow.css` - macro-stepper, context rail, choice cards, settings drawer
- `search-flow.js` - session restore, history.pushState, analytics events
- `index.html` - интеграция, крупные карточки выбора в wizard

## Компоненты UI

- **search-macro-stepper** - 4 фазы, кликабельные завершённые
- **search-context-rail** - «Ваш путь» с кнопками «изменить»
- **search-sub-stepper** - подшаги на фазе D
- **search-choice-card** - full-width карточки вместо pill-chip
- **search-settings-drawer** - S-tier и «только цитаты»
- **search-inline-population** - быстрый выбор аудитории на экране A
- **search-flow-restore-banner** - продолжить незавершённый поиск

## События (sessionStorage `protocol_search_flow_events`)

- `search_step_view`
- `search_step_back`
- `search_context_edit`
- `search_inline_population`
- `search_session_restore`

## KPI (целевые)

- Time to PDF < 40 с при явном МКБ
- Drop-off на шаге МКБ -30%
- Touch target ≥ 48px

## Связанные документы

- [search-funnel-v1.md](./search-funnel-v1.md)
