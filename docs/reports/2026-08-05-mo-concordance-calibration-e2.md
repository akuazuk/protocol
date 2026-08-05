# Калибровка MO concordance (E2)

Дата: 2026-08-05  
Источник: `mis_protocol_2026-07_complete.csv` (локально, PHI не коммитится)  
Выборка: **5000** МО (seed=42)  
Ветка: `codex/mo-concordance-smirnova-agent1-pc1`  
Скрипт: `scripts/calibrate_mo_concordance.py`

## Audience

| audience | n | share |
|--|--|--|
| adult | 4177 | 83.5% |
| pediatric | 786 | 15.7% |
| unknown | 37 | 0.7% |

## Trigger rates (после тюнинга)

- any shadow finding: **26** (0.5%)
- any P0/P1: **5** (0.1%)

| code | n | rate | pediatric | adult |
|--|--|--|--|--|
| `plan_laterality_mismatch` | 18 | 0.4% | 1 | 17 |
| `icd_weakly_supported` | 5 | 0.1% | 1 | 4 |
| `finding_not_in_diagnosis` | 5 | 0.1% | 1 | 4 |
| `anamnesis_thin_for_duration` | 1 | 0.0% | 1 | 0 |
| `pediatric_limp_ddx_not_addressed` | 1 | 0.0% | 1 | 0 |
| `underworkup_chronic_red_flag` | 0 | 0% | 0 | 0 |

## Severity mix (finding instances)

| severity | n |
|--|--|
| P0 | 0 |
| P1 | 5 |
| P2 | 7 |
| P3 | 18 |

## До / после тюнинга (та же выборка)

| Метрика | До | После |
|--|--|--|
| any finding | 11.9% | 0.5% |
| `anamnesis_thin_for_duration` | 11.5% | 0.02% |
| P0/P1 | 0.1% | 0.1% |

Главный шум был в `anamnesis_thin` на любой хронике без MSK red-flag.

## Решения E2

1. **`anamnesis_thin_for_duration`**: только при хромоте/отёке сустава + duration≥28д + (`themes<2` **и** `anam_len<120`).
2. **Negation**: «Отеки: нет» рядом с суставом не считается `joint_edema`.
3. **ICD cover whitelist**: расширен (`m25`, `артропат`, `ГСС`, `коксарт`).
4. **`underworkup_chronic_red_flag`**: **P1 только pediatric**, adult → P2 (уже в коде). На июле 5k почти не срабатывает (редкий полный funnel: chronic+red+no imaging/labs+follow-up on worsening) - оставляем правило, эталон Смирнова покрыт unit-тестом.
5. **`MO_CONCORDANCE_PRIMARY`**: **не включать**. Shadow only.
6. Blocking ETL / publish - **не трогать**.

## Fixtures

Обезличенные 5 positive / 5 negative: `eval/mo_concordance/`.  
Unit-эталон Смирнова: `tests/test_mo_concordance_smirnova.py` (6/6 codes).

## Следующий шаг

E3 - русские labels, очередь «Вчера», связка находка↔диагноз в case detail (только после merge shadow в main и согласия методиста на 10 кейсах).
