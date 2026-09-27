# Handoff: P6 экран «Пациент», P5 ещё накатывается

Дата: 2026-09-27

## Repo / прод

- Канон: `docs/plans/2026-09-27-mo-client-passport-score-ui-v2.md`
- Прод: `https://protocol.kravira.by`
- P4 в проде: `f46fb2b8`, version `2026-09-27-163848Z-case-steps-p4`
- P5 merge: `133e3d41` (#333), version `2026-09-27-165225Z-case-steps-p5`
- P5 deploy: идёт из `/private/tmp/protocol-release-133e3d41` (на [5/5] контейнер)
- P6 ветка: `cursor/mo-patient-page-p6-pc1`, worktree `/private/tmp/protocol-task-mo-patient-page-p6-1`
- `BUILD_VERSION` в ветке: `2026-09-27-170911Z-patient-page-p6`

## Сделано

- P4 #332 влито и проверено в UI: шаг 3, крошка на шаг 1, 14 визитов / 3 специальности.
- P5 #333 влито. Деплой с первого раза отказал (cwd был `f46fb2b8`). Повтор из release worktree `133e3d41`.
- P6 код: меню «Пациент», `GET /api/methodist/mo/patients/resolve`, вход на шаг 1. Запрос и PHI в ответ не кладутся.

## Проверки локально

`pytest` passport + steps + nav + frontend + route contract: все зелёные.
Снимок маршрутов: 220.

## Нужно

1. Дождаться P5 `PUBLIC_OK` и `/api/version` = `2026-09-27-165225Z-case-steps-p5` / `133e3d41`.
2. В браузере: `open=3737549` шаги 4 и 5, док решения только на 5.
3. Merge P6, деплой с `origin/main`, smoke: меню «Пациент», resolve visit_id, шаг 1, сентябрь fair 6341 / good 408.
4. S1: оценка дыр 2026 без LLM на GCE. Сентябрь не пересчитывать.
5. P7 после 40+40 меток.

Не трогать грязный checkout `/Users/pavelkuzauka/Cursor_Folders/Protocol`.
Не деплоить из task-worktree. Не переписывать сентябрь.
