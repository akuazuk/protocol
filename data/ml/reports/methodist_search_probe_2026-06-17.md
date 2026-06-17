# Methodist search probe report

- Generated: 2026-06-17 09:25 UTC
- Fixture: `/Users/pavel/CURSOR/Protocol/protocol/tests/fixtures/search_methodist_probe.jsonl`
- BUILD: `2026-06-01-r180-hybrid-probe-chunks`
- Probes: **111** (ok 111, errors 0)

- Avg AI rating (deterministic): **3.96** / 5
- Expected fragment in top-1: **76.6%** · top-3: **83.8%**
- Top-1 clinically irrelevant (AI): **0**
- Reject fragment in top-1: **0**

## Verdicts

- `mostly_correct`: 107
- `partially_wrong`: 4

## Tags

- `query_too_vague`: 4

## Recurring engine improvements

- (107×) Retrieve-only: ranking только по RAG; для аудита достаточно, для симптомов нужен шаг МКБ.
- (4×) Включить обязательный шаг МКБ-10 в воронке перед списком протоколов (symptom-only).

## Worst cases

| id | rating | verdict | top-1 | hit1 | reject |
|----|--------|---------|-------|------|--------|
| probe02 | 3 | partially_wrong | КП медицинской реабилитации пациентов, перенесши | False | False |
| probe03 | 3 | partially_wrong | КП_Диагностика_лечение_оториноларингологическими | True | False |
| probe55 | 3 | partially_wrong | КП_Диагностика_лечение_пациентов_с_заболеваниями | True | False |
| probe111 | 3 | partially_wrong | КП_Диагностика_лечение_оториноларингологическими | True | False |
| probe01 | 4 | mostly_correct | КП_Диагностика_и_лечение_взр_население_с_бронхиа | False | False |
| probe04 | 4 | mostly_correct | КП_Диагностика_и_лечение_взр_население_с_бронхиа | False | False |
| probe05 | 4 | mostly_correct | КП_Медицинская_реабилитация_пациентов_с_травмами | False | False |
| probe06 | 4 | mostly_correct | КП_Диагностика_и_лечение_взр_население_с_бронхиа | False | False |
| probe07 | 4 | mostly_correct | 05КП_Диагностика_лечение_эндоскопическими_метода | False | False |
| probe08 | 4 | mostly_correct | КП_Диагностика_и_лечение_взр_население_с_бронхиа | True | False |
