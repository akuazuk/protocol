# Methodist search probe report

- Generated: 2026-06-16 07:13 UTC
- Fixture: `/Users/pavel/CURSOR/Protocol/protocol/tests/fixtures/search_methodist_probe.jsonl`
- BUILD: `2026-06-01-r168-icd-symptom-expand-reliable-search`
- Probes: **15** (ok 15, errors 0)

- Avg AI rating (deterministic): **3.73** / 5
- Expected fragment in top-1: **60.0%** · top-3: **60.0%**
- Top-1 clinically irrelevant (AI): **0**
- Reject fragment in top-1: **0**

## Verdicts

- `mostly_correct`: 11
- `partially_wrong`: 4

## Tags

- `query_too_vague`: 4

## Recurring engine improvements

- (11×) Retrieve-only: ranking только по RAG; для аудита достаточно, для симптомов нужен шаг МКБ.
- (4×) Включить обязательный шаг МКБ-10 в воронке перед списком протоколов (symptom-only).

## Worst cases

| id | rating | verdict | top-1 | hit1 | reject |
|----|--------|---------|-------|------|--------|
| probe02 | 3 | partially_wrong | КП диагностики и лечения острого и хронического  | True | False |
| probe03 | 3 | partially_wrong | КП_Диагностика_лечение_оториноларингологическими | True | False |
| probe04 | 3 | partially_wrong | КП диагностики и лечения острого и хронического  | True | False |
| probe10 | 3 | partially_wrong | КП диагностики и лечения острого и хронического  | False | False |
| probe01 | 4 | mostly_correct | КП_Диагностика_лечение_оториноларингологическими | False | False |
| probe05 | 4 | mostly_correct | КП_Диагностика_лечение_острых_риносинуситов_вз_н | False | False |
| probe06 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | False | False |
| probe07 | 4 | mostly_correct | КП_Диагностика_лечение_внебольничной_пневмонии_д | True | False |
| probe08 | 4 | mostly_correct | 05КП_Диагностика_и_лечение_пациентов_взр_и_дет_н | False | False |
| probe09 | 4 | mostly_correct | КП_Диагностика_лечение_острых_респираторных_виру | True | False |
