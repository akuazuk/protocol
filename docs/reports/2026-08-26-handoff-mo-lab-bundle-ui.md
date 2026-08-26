# Handoff: лаборатория в разборе случая (волны 0-3)

Дата: 2026-08-26  
Репозиторий: `akuazuk/protocol`  
Worktree: `/private/tmp/protocol-task-mo-lab-from-mis-tests-pc1`  
Ветка: `cursor/mo-lab-from-mis-tests-pc1`  
PR: https://github.com/akuazuk/protocol/pull/178

Primary: `https://protocol.kravira.by`  
Deploy: нет (ждём merge PR). Ночной append заработает после выкладки `/opt/protocol`.

---

## Сделано

- Волна 0: `mo_lab.sqlite` на GCE, 427045 строк.
- Волна 1: блок «Лаборатория» в разборе (цифры там, не в findings).
- Волна 2: shadow-сверка план ↔ `type_name`.
- Волна 3: `night_mis_pipeline.sh` дописывает вчера+1д overlap в `mo_lab.sqlite`
  (host `venv-mis`, `--skip-coverage`, non-fatal).
- `MO_LAB_IN_PRIMARY` теперь читается. Default 0. При `=1` в оценку идёт только
  P3 «анализы есть, в МО не указаны». Не цифры и не «назначено повторно».

## Почему цифры не в замечаниях

В `mis_tests` нет референса и флага нормы. Finding с «гемоглобин 132» не говорит,
плохой он или нет, и уезжает в склад findings/отчёты. Методист видит значения
в блоке «Лаборатория». «Плохой анализ» из value - волна 4, после калибровки.

## Не сделано

- Выкладка GCE / merge.
- Включить `MO_LAB_IN_PRIMARY=1` в env GCE (намеренно default 0: лаб только у ~22% случаев).
- Оценка отклонения `value` без референса.

## Тесты

`pytest tests/test_ingest_mo_lab_from_mis_tests.py tests/test_mo_lab_bundle.py tests/test_mo_lab_shadow.py tests/test_mo_frontend_structure.py --noconftest`

## Следующая команда

Merge #178 + выкладка GCE. Затем при необходимости `MO_LAB_IN_PRIMARY=1` в public env.
