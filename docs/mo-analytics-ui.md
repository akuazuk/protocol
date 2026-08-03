# МО Аналитика: дизайн-система и словарь метрик

Источник истины по задачам: `docs/plans/2026-07-30-mo-analytics-bi-redesign-v1.md`.

## Экраны

- Обзор, Вчера, Месяц, Документы, Очередь, Отчёты, Кабинет врача, Настройки.
- Критические / «требует внимания»: строки с кнопками **МО** и **PDF**.
- Документ случая: `GET /api/methodist/mo/cases/{id}/document` (print-ready HTML).
- PDF: `GET /api/methodist/mo/cases/{id}/pdf` (Chrome headless или HTML fallback для печати).

## Словарь

| В UI | Не показывать |
|------|----------------|
| Требует внимания | bad / n_bad |
| Критические | P0-only jargon without label |
| Не оценивается: диагностика | Нет данных без причины |
| Визит / запись МИС | content_hash |

## Пустые состояния

- Нет файла отчёта, но есть витрина: карточка дня с KPI из `fact_mo_daily` и пометкой «витрина».
- Диагностика / неклиника в кабинете врача скрыты по умолчанию (`include_unscored=false`).
- Оценка `null` для scored kind: «Оценка ещё не рассчитана».

## Health

`GET /api/methodist/mo/health` - свежесть, сверка витрина↔report.json, флаги фич.

## Брифинг

`python3 scripts/ops/mo_morning_briefing.py [--date YYYY-MM-DD] [--dry-run]`
