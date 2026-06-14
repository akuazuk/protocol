# Methodist search probe report

- Generated: 2026-06-14 15:02 UTC
- Fixture: `/Users/pavelkuzauka/Cursor_Folders/Protocol/tests/fixtures/search_methodist_probe.jsonl`
- BUILD: `2026-06-01-r148-clinical-search-routing-probe110`
- Probes: **110** (ok 110, errors 0)

- Avg AI rating (deterministic): **3.98** / 5
- Expected fragment in top-1: **78.2%** · top-3: **90.0%**
- Top-1 clinically irrelevant (AI): **1**
- Reject fragment in top-1: **0**

## Verdicts

- `mostly_correct`: 109
- `wrong`: 1

## Tags

- `wrong_population`: 1

## Recurring engine improvements

- (110×) Retrieve-only: ranking только по RAG; для аудита достаточно, для симптомов нужен шаг МКБ.
- (1×) При «взрослое население» в воронке отфильтровывать/опускать детские КП (дет нас, детск).
- (1×) Усилить doc_audience_hint для «дет нас» без подчёркивания в названии PDF.
- (1×) Проверить шаг 1 воронки: population=adult должен попадать в funnel_context feedback.

## Worst cases

| id | rating | verdict | top-1 | hit1 | reject |
|----|--------|---------|-------|------|--------|
| probe79 | 2 | wrong | КП_Диагностика_лечение_пациентов_заболеваниями_н | False | False |
| probe01 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe02 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe03 | 4 | mostly_correct | КП_Диагностика_лечение_оториноларингологическими | False | False |
| probe04 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | True | False |
| probe05 | 4 | mostly_correct | КП_Диагностика_лечение_острых_риносинуситов_вз_н | False | False |
| probe06 | 4 | mostly_correct | КП Диагностика и лечение пациентов (взрослое нас | False | False |
| probe07 | 4 | mostly_correct | КП_Диагностика_лечение_внебольничной_пневмонии_д | True | False |
| probe08 | 4 | mostly_correct | КП_Диагностика_и_лечение_пациентов_д-нас_с_бронх | True | False |
| probe09 | 4 | mostly_correct | КП диагностики и лечения острого и хронического  | False | False |
