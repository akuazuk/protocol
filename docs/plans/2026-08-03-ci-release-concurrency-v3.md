# CI baseline и сериализованный production release (v3)

Дата: 2026-08-03
Статус: active
Предшественник: `2026-08-03-multi-agent-release-guard-v2.md`

## Контекст

Hard guards уже не позволяют деплоить task HEAD, `main` защищён PR и обязательным
`manifest-mode`, production точно показывает Git SHA. Следующий незакрытый риск -
`lint-and-test` падает на 102 накопленных Ruff-ошибках, поэтому его нельзя сделать
обязательным. Render release запускается с рабочих компьютеров, а не единым GitHub Action.

## Что изменено в production

Production работает на `883e43a2490b11aa071fc243872bbf63a41b94fb`,
`2026-08-03-r22-mo-filter-actions-ui`. GitHub `main` требует PR и актуальный
`manifest-mode`; release wrapper принимает только текущий `origin/main`.

## Метрики

- Ruff errors: было 102, стало 0, цель 0.
- Full pytest: был скрыт красным lint, стало 100% passed (1 skip), цель green.
- Required CI checks: был 1 (`manifest-mode`), пока 1, цель 2 (`manifest-mode`, `lint-and-test`).
- Production release concurrency: было локальное соглашение, стала 1 GitHub concurrency group.
- Release source in CI: было отсутствует, стал только `github.sha` после push в `main`.
- Render API secret: сейчас отсутствует в GitHub, workflow должен безопасно работать через webhook.

## Шаги

- [x] Исправить безопасные Ruff autofix и вручную устранить остаток.
- [x] Запустить полный `ruff check .` и полный pytest.
- [x] Исправить скрытый pytest baseline: перенесённые frontend paths и 4 функциональных теста.
- [x] Добавить GitHub Actions workflow с `concurrency: production-render`.
- [x] В workflow проверять только точный `github.sha` через canonical release wrapper.
- [ ] После зелёного PR сделать `lint-and-test` обязательным для `main`.
- [x] Поднять `BUILD_VERSION` до `2026-08-03-r23-ci-release-concurrency`.
- [ ] Commit, push, PR, merge и production verification.

## Риски

- Автоматическое удаление unused imports может выявить скрытые import-side effects. Все
  изменённые модули проверяются тестами и компиляцией.
- Полный pytest может быть долгим или зависеть от внешних данных. CI должен оставаться
  без внешних LLM-вызовов.
- Без `RENDER_API_KEY` workflow не создаёт deploy через API, а ждёт Render webhook. Secret
  можно добавить позже без изменения workflow.
