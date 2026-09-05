# Handoff: лаборатория в Endpoint C (shadow)

Дата: 2026-08-27
Репозиторий: `akuazuk/protocol`
Ветка: `cursor/mo-lab-dx-evidence-agent1-pc1`
Worktree: `/private/tmp/protocol-task-mo-lab-dx-evidence-pc1`
Base: `53d61e51` (`origin/main`)
PR: открыть после push этой ветки
Production: не менялся; live SHA остаётся `53d61e51`, `MO_LAB_IN_PRIMARY=0`

## Сделано

- Endpoint C получил слот `lab`: названия панелей и даты, без `value` / `unit` /
  `patient_id` / `patient_key` / `visit_id`.
- `dx_evidence_eligibility`: диагноз + лаборатория склада = `eligible`. Сырой
  UI-бандл со значениями сам по себе eligibility не открывает.
- Blind pack и GCE shadow runner кладут тот же snippet в промпт. Инструкция
  запрещает `poor`/`critical` за «отклонение показателя».
- Live/batch `result.lab.dx_evidence` показывает, что увидит Endpoint C.
- Finding `B_dx_lab_context` только shadow: диагноз есть, клинический текст
  пуст, панели на складе есть. В primary не поднимается.
- `exam_data` и формула оценки не менялись.

## Не сделано

- `MO_LAB_IN_PRIMARY=1` и оценка цифр без референса.
- Lab-aware Endpoint D / KP plan shadow.
- Live Gemini прогон Endpoint C на GCE с новым слотом.

## Тесты

- `pytest -q` узкий набор: lab dx evidence, calibration contracts/blind,
  lab shadow - passed.
- Не гонять night LLM / `grade_kz_llm` с Mac.

## Следующая команда

После merge: shadow Dx/Plan на GCE (`deploy/gcp-llm/run_on_gce.sh`), не с Mac.
Primary не включать.

Нельзя параллельно трогать: `clinical_knowledge/mo_dx_evidence_score.py`,
`clinical_knowledge/mo_lab_dx_evidence.py`,
`scripts/run_mo_calibration_blind_judge.py`.
