# Handoff: вернуть разбор плохих МО

Дата: 2026-09-27

## Repo

- origin/main после R1: `138c2caf`
- Эта ветка: `cursor/mo-review-steps-flag-pc1` (сохранить `?steps=1` в URL)
- Worktree: `/private/tmp/protocol-task-mo-review-steps-flag-1`
- Прод после R1: `https://protocol.kravira.by`
  version `2026-09-27-184750Z-review-restore-r1`
  git_commit `138c2caf7da3b99275d4965b7afe2eb27c6cd205`

## Сделано

- R0 #336 `48ce5a0d` задеплоен: столбец «Проверка», зоны, «Почему так»,
  «Что не так», решение. Мастер только `?steps=1`.
- R1 #337 `138c2caf` задеплоен: «Срез» / «Основание», решение не прячется.
- Проверено на слабо-случае сентября: hero + 3 зоны + why + findings +
  decision, вкладка «Проверка» активна, мастера нет.
- R2 не делали: разбор снова годится, чтобы найти плохое МО.
- Сентябрь не пересчитывали.

## Делается

- Hotfix: `syncUrl` сохраняет `steps`, иначе `?steps=1` сразу пропадает.

## Нужно

- Merge и деплой этой ветки, если ещё не в main.
- S1 (дыры 2026) - отдельно, после extract января-марта. Не с Mac.
- P7 - после 40+40 меток.
- Не трогать dirty checkout `/Users/pavelkuzauka/Cursor_Folders/Protocol`.
