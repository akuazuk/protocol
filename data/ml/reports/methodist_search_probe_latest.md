# Methodist search probe report

- Generated: 2026-06-14 15:30 UTC
- Fixture: `/Users/pavelkuzauka/Cursor_Folders/Protocol/tests/fixtures/search_methodist_probe.jsonl`
- BUILD: `2026-05-31-r150-search-probe-fixes-kz-auto`
- Probes: **110** (ok 110, errors 0)

- Avg AI rating (deterministic): **4.0** / 5
- Expected fragment in top-1: **80.0%** · top-3: **91.8%**
- Top-1 clinically irrelevant (AI): **0**
- Reject fragment in top-1: **0**

## Verdicts

- `mostly_correct`: 110

## Recurring engine improvements

- (110×) Retrieve-only: ranking только по RAG; для аудита достаточно, для симптомов нужен шаг МКБ.

## Worst cases

| id | rating | verdict | top-1 | hit1 | reject |
|----|--------|---------|-------|------|--------|
| probe01 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe02 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe03 | 4 | mostly_correct | КП_Диагностика_лечение_оториноларингологическими | False | False |
| probe04 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe05 | 4 | mostly_correct | КП_Диагностика_лечение_острых_риносинуситов_вз_н | False | False |
| probe06 | 4 | mostly_correct | КП_Диагностика_лечение_оториноларингологическими | False | False |
| probe07 | 4 | mostly_correct | КП_Диагностика_лечение_внебольничной_пневмонии_д | True | False |
| probe08 | 4 | mostly_correct | КП_Диагностика_и_лечение_пациентов_д-нас_с_бронх | True | False |
| probe09 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | False | False |
| probe10 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | False | False |
