# МО: legacy `consultation` снова оценивается как clinical_visit

Статус: **active**  
Дата: 2026-08-08

## Контекст

После hard-gate «только clinical_visit» кейсы с `document_kind=consultation`
(legacy клинический приём в витрине до recompute) перестали получать:

- рубрику МЗ «Как оценивать» (`rubric_mz`);
- №55 / findings / KP / LLM.

В UI это выглядело как: «Shadow-оценка по методике МЗ пока недоступна».

## Решение

1. `consultation` ∈ `SCORED_DOCUMENT_KINDS` (алиас clinical_visit).
2. SQL helper `scored_kind_sql` включает оба вида.
3. UI показывает `rubric.reason` / `error`, а не общую фразу.

## Отличие рубрики МЗ и №55 (не смешивать в одном PR)

| | Рубрика МЗ «Как оценивать» | Пост. №55 |
|--|--|--|
| Источник | sheet/YAML, 13 критериев, шкала 0/0.5/1 | `mz_2021_55.json`, pass/fail пунктов |
| Регламент | в основном №127 (+ часть №55) | только №55 case-level |
| Роль | shadow полноты/глубины записи | ось regulatory + findings `D_reg55_*` |
| Объединение | позже: один «итог методики МЗ» после калибровки | сейчас отдельно в карточке |

## Шаги

1. [x] План
2. [x] SCORED kinds + SQL/API
3. [x] UI reason for rubric
4. [ ] Tests → PR → GCE
