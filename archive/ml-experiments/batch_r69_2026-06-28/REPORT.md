# Batch r69 на Render (`2026-06-28-r69-fix-raw-text-length`)

Дата: 2026-06-28

## Smoke

| Проверка | Результат |
|----------|-----------|
| Render version | `r69-fix-raw-text-length` |
| B2C `A_2`, `a_1`, `a_3` | `upload_mismatch=true`, `overall=null` (шутка, без scoring) |
| KZ без 500 | OK |

## L1 `--kz-only` (30 КЗ)

- **avg overall: 84.2%**
- **&lt;70%:** 2 кейса — `report_n_1` (60%), `report_n_2` (61.9%)

### Причины слабых (не routing)

| case | overall | причина |
|------|--------:|---------|
| report_n_1 | 60% | `manual_review_required`: 2× НПВП (аэртал+дексалгин), красный флаг без маршрутизации |
| report_n_2 | 61.9% | короткое КЗ (1085 симв.): sparse neurology caps (structural 35%, diag 45%) |

Routing gates r67: детский невро КП **отклоняется** (`population_mismatch`), в target paths — взрослые КП.

## L2 weak (2 КЗ)

| case | L2 overall |
|------|----------:|
| report_n_1 | 35% |
| report_n_2 | 62% |

## Следующий шаг (r70)

Фильтр `filter_consult_protocol_matches` — детский/HIV КП не попадают в `matches` для scoring/UI.
