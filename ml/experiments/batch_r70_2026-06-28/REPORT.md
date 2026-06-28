# Batch r70 на Render (`2026-06-28-r70-match-routing-filter`)

Дата: 2026-06-28

## L1 `--kz-only` (30 КЗ)

| Метрика | Значение |
|---------|----------|
| OK | 30/30 |
| avg overall | **84.2%** |
| &lt;70% | **2** |

### Слабые кейсы (не chunk QA)

| case | overall | status | причина |
|------|--------:|--------|---------|
| report_n_1 | 60% | manual_review_required | 2× НПВП, красный флаг без маршрутизации |
| report_n_2 | 61.9% | partially_compliant | короткое КЗ, sparse neurology caps |

r70: детский невро КП уходит в `not_applicable`, matches для scoring чище.

## Вывод

Средний балл **84.2%** - норма для пилота. Два слабых кейса - **клиническая логика scoring**, не качество чанков.

Следующий шаг: см. `GEMINI_FULL_QA_PLAN.md` - полный прогон для протоколов weak routing + retry низкого quality score.
