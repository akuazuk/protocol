# Аудит knowledge-корпуса протоколов (scorer v3)

- Протоколов: **477**
- Требований (атомарных): **11259**
- Penalty-eligible правил: **0** (0.0%)
- Advisory правил: **11259**
- С подтверждённой цитатой: **11148** (99.0%)

## Ключевые метрики покрытия (§10.2)

| Метрика | Значение |
|---|---|
| protocol_structured_coverage_pct | 0.0% |
| penalty_eligible_coverage_pct | 0.0% |
| source_verified_coverage_pct | 99.0% |
| methodist_approved_coverage_pct | 0.0% |
| protocols_without_safe_penalty_rule | 477 |

## Trust levels

| Level | N |
|---|---|
| A | 0 |
| B | 0 |
| C | 477 |
| D | 0 |

## Review status

| Status | N |
|---|---|
| needs_review | 464 |
| not_reviewed | 13 |

## Покрытие структур (протоколов с ≥1 элементом)

| Поле | N |
|---|---|
| diagnosis_criteria | 372 |
| required_exams | 383 |
| conditional_exams | 356 |
| treatment | 413 |
| dose | 369 |
| route | 0 |
| frequency | 1 |
| duration | 0 |
| red_flags | 340 |
| monitoring | 0 |
| follow_up | 366 |

## Причины непригодности к штрафу

| Причина | N |
|---|---|
| trust_below_B | 11259 |
| no_applicability | 11259 |
| quote_not_verified | 111 |

> Наличие правила != пригодность к штрафу. Штраф допустим только для trust A/B
> с подтверждённой цитатой и применимостью (см. ТЗ §6, §10.2).
