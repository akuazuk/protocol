# Handoff: лаборатория в разборе случая (волны 0-2)

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
- Волна 1: бандл `clinical_knowledge/mo_lab_bundle.py`, блок «Лаборатория» в разборе.
- Волна 2: shadow-сверка `mo_lab_shadow.py` - рекомендации ↔ `type_name`.
  Finding P3 только shadow: назначено и уже есть; есть на складе, в МО не указано.
  «Назначено, на складе нет» - строка в блоке, не штраф.
  `MO_LAB_IN_PRIMARY` игнорируется. Значения в finding не кладём.

## Не сделано

- Волна 3: night append вчерашнего дня в `mo_lab.sqlite`.
- Волна 4: primary / «плохой анализ» из `value` (нет референса).
- Deploy GCE после merge: не `deploy_to_gce.sh` вслепую с Mac SSH `pavelkuzauka`.

## Тесты

`PYTHONPATH=. python3 -m pytest tests/test_mo_lab_bundle.py tests/test_mo_lab_shadow.py tests/test_ingest_mo_lab_from_mis_tests.py tests/test_mo_frontend_structure.py --noconftest`

## Запрет параллельно

- `clinical_knowledge/mo_lab_bundle.py`, `mo_lab_shadow.py`, `mo_backend.py`,
  `kz_deep_eval.py`, `frontend/web/shared/mo-app.js`, план lab-from-mis-tests.

## Следующая команда

Волна 3 после merge: night append вчерашнего дня. Не включать `MO_LAB_IN_PRIMARY`.
