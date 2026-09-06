# Handoff: защита МО от устаревшего ответа среза

Дата: 2026-09-06. Roadmap: A28.

Branch `cursor/mo-stale-response-guard-agent1-pc1`.
Worktree `/private/tmp/protocol-task-mo-stale-response-guard-pc1` (locked).
Base при старте: `e7121a695b41174a42413d2b937175a0c5fff570`.

## Требование и реализация

При новой загрузке страницы или смене фильтра увеличивается epoch среза и
прерывается предыдущий `AbortController`. Все GET/HEAD запросы страницы получают
signal этого epoch. Даже если транспорт не прервал старый ответ, wrapper
сравнивает epoch до передачи response рендереру. `AbortError` не показывается
пользователю как сбой нового среза. POST-действия не отменяются сменой фильтра.

Изменён один runtime-модуль: `frontend/web/shared/mo-app.js`. Клиническая
методика, данные, API и primary flags не меняются.

## Проверки

- `node --check frontend/web/shared/mo-app.js` - passed;
- 41 focused frontend pytest - passed;
- полный Playwright - 20 passed;
- отдельный synthetic race: первый lab response задержан и возвращает 11%,
  второй после смены периода возвращает 77%; после завершения обоих UI остаётся
  на 77%, global error скрыт;
- IDE lint - без ошибок.

Постоянный regression test с управляемой задержкой должен быть опубликован
отдельным test-only PR после code PR, чтобы не смешивать уровень 4 `tests/`
с frontend feature.

## Статусы

На момент первой записи: implemented и locally verified. CI, merge и deploy
ещё не выполнены. Clinical validation не требуется: изменение только устраняет
гонку отображения и не меняет медицинский вывод.

Следующая безопасная команда после merge текущего tracker PR:

```bash
scripts/ops/rebase_task_onto_main.sh
```
