# Methodist search probe report

- Generated: 2026-06-14 17:42 UTC
- Fixture: `/Users/pavelkuzauka/Cursor_Folders/Protocol/tests/fixtures/search_methodist_probe.jsonl`
- BUILD: `2026-05-31-r159-probe-quality`
- Probes: **111** (ok 111, errors 0)

- Avg AI rating (deterministic): **4.0** / 5
- Expected fragment in top-1: **95.5%** · top-3: **100.0%**
- Top-1 clinically irrelevant (AI): **0**
- Reject fragment in top-1: **0**

## Verdicts

- `mostly_correct`: 111

## Recurring engine improvements

- (111×) Retrieve-only: ranking только по RAG; для аудита достаточно, для симптомов нужен шаг МКБ.

## Worst cases

| id | rating | verdict | top-1 | hit1 | reject |
|----|--------|---------|-------|------|--------|
| probe01 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe02 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe03 | 4 | mostly_correct | КП_Диагностика_лечение_оториноларингологическими | True | False |
| probe04 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe05 | 4 | mostly_correct | КП_Оказание_медпомощи_пациентам_с_аффективными_р | False | False |
| probe06 | 4 | mostly_correct | КП Ранняя диагностика и интенсивная терапия остр | True | False |
| probe07 | 4 | mostly_correct | КП диагностики и лечения пневмоний 05.07.2012 №  | True | False |
| probe08 | 4 | mostly_correct | КП_Диагностика_и_лечение_взр_население_с_бронхиа | True | False |
| probe09 | 4 | mostly_correct | КП_Диагностика_лечение_острых_респираторных_виру | True | False |
| probe10 | 4 | mostly_correct | КП_Диагностика_лечение_острых_респираторных_виру | True | False |
