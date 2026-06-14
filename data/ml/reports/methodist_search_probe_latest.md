# Methodist search probe report

- Generated: 2026-06-14 16:49 UTC
- Fixture: `/Users/pavelkuzauka/Cursor_Folders/Protocol/tests/fixtures/search_methodist_probe.jsonl`
- BUILD: `2026-05-31-r155-search-ui-matrix-kz`
- Probes: **3** (ok 3, errors 0)

- Avg AI rating (deterministic): **4.0** / 5
- Expected fragment in top-1: **66.7%** · top-3: **66.7%**
- Top-1 clinically irrelevant (AI): **0**
- Reject fragment in top-1: **0**

## Verdicts

- `mostly_correct`: 3

## Recurring engine improvements

- (3×) Retrieve-only: ranking только по RAG; для аудита достаточно, для симптомов нужен шаг МКБ.

## Worst cases

| id | rating | verdict | top-1 | hit1 | reject |
|----|--------|---------|-------|------|--------|
| probe01 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe02 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe03 | 4 | mostly_correct | КП_Диагностика_лечение_оториноларингологическими | False | False |
