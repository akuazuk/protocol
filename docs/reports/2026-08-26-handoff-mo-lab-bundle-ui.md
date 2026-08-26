# Handoff: лаборатория в разборе случая (волна 1)

Дата: 2026-08-26  
Репозиторий: `akuazuk/protocol`  
Worktree: `/private/tmp/protocol-task-mo-lab-from-mis-tests-pc1`  
Ветка: `cursor/mo-lab-from-mis-tests-pc1`  
PR: https://github.com/akuazuk/protocol/pull/178  
Канон: не трогать грязный `/Users/pavelkuzauka/Cursor_Folders/Protocol` на `main`.

Primary: `https://protocol.kravira.by`  
Deploy этой волны: нет (ждём merge PR).

---

## Сделано

- Волна 0: `mo_lab.sqlite` на GCE, 427045 строк, 21.8% случаев МО с лаб в окне −14д…+1д.
- Волна 1: бандл `clinical_knowledge/mo_lab_bundle.py` читает склад по `patient_key` + окну дат.
- Case detail отдаёт `lab` (без `patient_id` / `patient_key` наружу).
- В разборе случая блок «Лаборатория» рядом с историей. Live SQL на клик нет.
- `MO_LAB_IN_PRIMARY` в v1 игнорируется. `exam_data` не трогали.

## Не сделано

- Волна 2: shadow «Рекомендации по обследованию» ↔ `type_name`.
- Волна 3: night append вчерашнего дня.
- Волна 4: primary / «плохой анализ» из `value` (нет референса).
- Deploy GCE после merge: не `deploy_to_gce.sh` с Mac SSH `pavelkuzauka` вслепую
  (chown `.env.gcp-public` ломает cron-user `pavel`). Обход: sudo assemble + docker.

## Тесты

`PYTHONPATH=. python3 -m pytest tests/test_mo_lab_bundle.py tests/test_ingest_mo_lab_from_mis_tests.py tests/test_mo_frontend_structure.py --noconftest`

## Запрет параллельно

- `clinical_knowledge/mo_lab_bundle.py`, `mo_backend.py` (build_case_detail),
  `frontend/web/shared/mo-app.js` (drawer), план lab-from-mis-tests.

## Следующая команда

Волна 2 после merge+GCE: shadow-сверка назначений и `type_name`. Не включать `MO_LAB_IN_PRIMARY`.
