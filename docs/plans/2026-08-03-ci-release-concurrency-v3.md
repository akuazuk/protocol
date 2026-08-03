# CI baseline и сериализованный production release (v3)

Дата: 2026-08-03
Статус: completed
Предшественник: `2026-08-03-multi-agent-release-guard-v2.md`

## Контекст

Hard guards уже не позволяют деплоить task HEAD, `main` защищён PR и обязательным
`manifest-mode`, production точно показывает Git SHA. Следующий незакрытый риск -
`lint-and-test` падает на 102 накопленных Ruff-ошибках, поэтому его нельзя сделать
обязательным. Render release запускается с рабочих компьютеров, а не единым GitHub Action.

## Что изменено в production

Production успешно обновлён GitHub Action на
`721e10e655199ecd52343b40700963155958e258`,
`2026-08-03-r23-ci-release-concurrency`. GitHub `main` требует PR и оба актуальных
check: `manifest-mode`, `lint-and-test`.

## Метрики

- Ruff errors: было 102, стало 0, цель 0.
- Full pytest: был скрыт красным lint, стало 100% passed (1 skip), цель green.
- Required CI checks: был 1 (`manifest-mode`), стало 2 (`manifest-mode`, `lint-and-test`), цель 2.
- Production release concurrency: было локальное соглашение, стала 1 GitHub concurrency group.
- Release source in CI: было отсутствует, стал только `github.sha` после push в `main`.
- Render API secret: сейчас отсутствует в GitHub, workflow должен безопасно работать через webhook.

## Шаги

- [x] Исправить безопасные Ruff autofix и вручную устранить остаток.
- [x] Запустить полный `ruff check .` и полный pytest.
- [x] Исправить скрытый pytest baseline: перенесённые frontend paths и 4 функциональных теста.
- [x] Добавить GitHub Actions workflow с `concurrency: production-render`.
- [x] В workflow проверять только точный `github.sha` через canonical release wrapper.
- [x] После зелёного PR сделать `lint-and-test` обязательным для `main`.
- [x] Поднять `BUILD_VERSION` до `2026-08-03-r23-ci-release-concurrency`.
- [x] Commit, push, PR #8, merge и production verification.
- [x] Подтвердить первый serialized workflow run: exact merge SHA, success за 2m1s.

## Риски

- Автоматическое удаление unused imports может выявить скрытые import-side effects. Все
  изменённые модули проверяются тестами и компиляцией.
- Полный pytest может быть долгим или зависеть от внешних данных. CI должен оставаться
  без внешних LLM-вызовов.
- Без `RENDER_API_KEY` workflow не создаёт deploy через API, а ждёт Render webhook. Secret
  можно добавить позже без изменения workflow.
