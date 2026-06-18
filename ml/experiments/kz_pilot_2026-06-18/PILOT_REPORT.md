# Пилот batch КЗ (2026-06-18)

- **Папка:** `/Users/pavel/CURSOR/Protocol/protocol/clients_consult`
- **Файлов:** 10
- **Уровни:** L1
- **OK:** 10/10

## Сводка L1

| case_id | overall % | rules % | failed | analysis_id |
|---------|-----------|---------|--------|-------------|
| report_1 | 88.1% | None% | 0 | `20d172df…` |
| report_10 | 90.4% | None% | 0 | `e1712a12…` |
| report_2 | 59.3% | None% | 0 | `e87e87f0…` |
| report_3 | 86.9% | 50.0% | 1 | `30ce802f…` |
| report_4 | 61.9% | None% | 0 | `37017d56…` |
| report_5 | 87.4% | None% | 0 | `dc049ec2…` |
| report_6 | 82.4% | 0.0% | 2 | `a4c5fcc0…` |
| report_7 | 92.6% | 100.0% | 0 | `fe3499a9…` |
| report_8 | 95.9% | None% | 0 | `e104e73e…` |
| report_9 | 90.6% | None% | 0 | `144ab058…` |

## Дальше

1. UI: **Кабинет методиста** → **Очередь** → **Открыть** по `analysis_id`.
2. Проверьте overrides → **Одобрить — сохранить для обучения движка**.
3. `python3 scripts/export_training_feedback.py` → `ml/datasets/priority_cases.jsonl`.

Подробнее: `kz_pilot_2026-06-18/REVIEW_QUEUE.md`
