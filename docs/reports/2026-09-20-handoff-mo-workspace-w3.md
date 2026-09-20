# Handoff: W3 рабочий стол МО - честный Обзор

Дата: 2026-09-20
Ветка: `cursor/mo-workspace-w3-pc1`
Worktree: `/private/tmp/protocol-task-mo-workspace-w3-1`
Base: `origin/main` `804f87da` (W2 в проде)

## Сделано в коде (ещё не прод, пока нет merge)

- Плитки Обзора и «Критично в очереди» читают `score-dashboard.attention` / `queue` того же окна, что календарь.
- Заголовок: «Окно фильтров», не «Итоги за рабочий день» при зерне Месяц.
- Центр кольца: слово шкалы (в норме / слабо / плохо), не сырой %.
- Динамика: tooltip с n, `connectNulls: false`, клик дня уже ставил календарь.
- «Период» в Ещё ведёт на Обзор (`switchPage` alias).

`BUILD_VERSION` `2026-09-20-102745Z-mo-workspace-w3`.

## Прод сейчас

W2: SHA `804f87da`, version `2026-09-20-095731Z-mo-workspace-w2`. Разбор 1188 px, `?open=`, «К списку» возвращает таблицу.

## Дальше

Merge W3 → GCE. Затем W4 Поиск МИС (нужен GCE SQL, не Mac).
