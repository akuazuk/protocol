# МО: баннер «Данные неполные (scoring_coverage, llm_queue_pending)»

## Контекст

В МО Аналитике «Вчера» показывался баннер partial с `scoring_coverage` и
`llm_queue_pending`. На 2026-08-08: coverage 81.77% (166/203), pending 80.

## Причины

1. **scoring_coverage:** `split_kz_rows` доверял CSV `document_kind=diagnostic`,
   тогда как live `classify_document_kind` даёт `clinical_visit` (консультация+УЗИ).
   Completeness всегда классифицирует live → ложная дыра в покрытии.
2. **llm_queue_pending:** grades с Gemini spend cap (`429 spending cap`) не
   снимали pending; очередь выглядела «ещё доделывается».

## Изменения

1. L1 batch: eligibility только через live `classify_document_kind`.
2. Общий `count_llm_queue_pending`: успех или терминальная ошибка (spend/geo)
   снимает visit из pending.
3. Ops: resume-score дней 2026-08-04..08 + recompute на GCE.

## Метрики

| | Было (08.08) | Цель |
|--|--|--|
| coverage | 81.77% | ≥99% |
| partial reasons | scoring_coverage + llm_queue_pending | [] |
| llm_queue_pending при spend-cap grades | 80 | 0 |

## Статус

- [x] код + тесты
- [x] resume score + recompute на GCE (04..09: coverage 100%, partial=false)
- [x] PR https://github.com/akuazuk/protocol/pull/126
- [ ] merge / deploy image (hot-patch в контейнере до redeploy)
