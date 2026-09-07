# Handoff: релиз remaining-work P0/P1-1 на GCE

Дата: 2026-09-07.
Repo: `akuazuk/protocol`.
Production SHA: `29b8a1746184527f94e5e66945166d2b6906a2cd`.
`BUILD_VERSION`: `2026-09-07-104339Z-history-same-day`.
Образ: `protocol-gcp-app:29b8a1746184`.
Предыдущий образ: `protocol-gcp-app:a6955ef2beac`.

## Merge

| PR | Merge SHA | Содержание |
|---|---|---|
| #243 | `46fd75c1` | concurrent save + training opt-in |
| #244 | закрыт | конфликт BUILD_VERSION после #243 |
| #245 | `29b8a174` | same-day history, rebase #244 на новый main |

CI #243 и #245 зелёные до merge. Deploy: `bash deploy/gcp-app/deploy_to_gce.sh`
из чистого worktree `/private/tmp/protocol-deploy-main` при `HEAD == origin/main`.
Rceth на VM не был запущен.

## Smoke на protocol.kravira.by

- `/health/live`: ok, version совпал.
- `/api/version`: `version` = `2026-09-07-104339Z-history-same-day`,
  `git_commit` = `29b8a1746184527f94e5e66945166d2b6906a2cd`.
- Скрипт деплоя: `PUBLIC_OK`.
- Verifier лаборатории внутри образа: `ranges=8 panels=17 shadow_findings=1`.
- GET review-packs без токена: 403. Prod mutation не выполнялась.
- `rag_ready` сразу после рестарта был false (асинхронная загрузка корпуса).
  Это не регресс оценки МО.

## Что в проде из remaining-work

- P0-1 защита конкурирующего сохранения и lineage.
- P0-2 opt-in / revoke допуска к обучению.
- P1-1 earlier same-day history по доказанному времени; query_failed не available.
- E20 поведенческие тесты сохранения. E06 query_failed.

## Что не закрыто

- P1-2 evaluated denominators и projection решения методиста.
- P1-3 E04 HTTP list/detail/export; E23 как постоянный CI (в этом релизе
  verifier отработал в образе, но это не отдельный required check).
- P1-4 UX leftover на кадре 07.09.
- P1-5 клиническая и нормативная приёмка.
- Пользовательская приёмка методистом на synthetic задачах не проводилась.

Весь план A01-A32 / CASE_REVIEW не объявлять закрытым.

Откат: предыдущий образ `protocol-gcp-app:a6955ef2beac` плюс снапшот
`protocol-data`. Render не прод.

## Следующая безопасная команда

Новый worktree от `29b8a174`:

```text
scripts/ops/git_task_start.sh mo-evaluated-denominators --pc=pc1 \
  --branch=cursor/mo-evaluated-denominators-agent1-pc1
```
