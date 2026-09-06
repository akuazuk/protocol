# Handoff: канонический план CASE Review и MO Analytics

Дата: 2026-09-06.
Repo: `akuazuk/protocol`.
Branch: `cursor/mo-case-review-plan-agent1-pc1`.
Worktree: `/private/tmp/protocol-task-mo-case-review-plan-pc1`.
Base: `495b2f439a07a2975c6e27d5115e86a2460401d1`.

## Сделано

- Локальный дополнительный `CASE_REVIEW_IMPLEMENTATION_PLAN.md` перенесен в
  versioned plan `docs/plans/2026-09-06-mo-case-review-analytics-v1.md`.
- Сохранены матрицы R01-R14, U01-U14, этапы A-E, 23 synthetic сценария,
  production baseline, метрики, риски и clinical/data/usability gates.
- В `docs/plans/README.md` добавлена append-only строка.
- Порядок общего файла согласован комментариями: этот план идет первым, PR #113
  и #186 перед merge должны перенестись на новый `origin/main` и сохранить строку.

## Production baseline

- SHA: `495b2f439a07a2975c6e27d5115e86a2460401d1`.
- `BUILD_VERSION`: `2026-09-06-155213Z-family-provenance`.
- `/health/live` успешен.
- Family provenance contract version 1 доступен.
- Lab assets и synthetic image evaluation проверены.

## Проверки

- `tests/test_plans_index.py`.
- UI dash normalization.
- `git diff --check`.
- Required CI после публикации PR.

## Не сделано

- Этапы A-E и U01-U14 этим документационным PR не реализуются.
- Clinical primary, новые веса и массовый backfill не включаются.
- Production не требует отдельного deploy только из-за plan publication.

## Следующая безопасная команда

После merge plan PR создать task-worktree от нового `origin/main` для persisted
assessment contract A1-A4.

## Не менять параллельно

- `docs/plans/README.md`;
- `docs/plans/2026-09-06-mo-case-review-analytics-v1.md`;
- `rag_server.py` до merge этого PR.
