# Handoff: разбор случая R6

Дата: 2026-09-21
Repo: `akuazuk/protocol`
Ветка: `cursor/mo-case-review-r6-pc1`
Worktree: `/private/tmp/protocol-task-mo-case-review-r6-pc1`
Прод: GCE `protocol.kravira.by` (не Render)

## Сделано

| Волна | PR | Merge SHA | BUILD_VERSION | Прод |
|--|--|--|--|--|
| R0-R5 | #271-#276 | `82ac1d64` | `2026-09-21-042344Z-mo-r5-meds-rceth` | да, до этой волны |
| R6 №55 читаемый | этот PR | нет до merge | см. `BUILD_VERSION` коммита | нет до GCE |

R6:

- аккордеон №55 ведёт градацией п.13 словами и подписью pack, не крупным средним %;
- fail/partial пункты сверху, полный чек-лист внутри, сначала невыполненные;
- «Почему так» добавляет fail-пункт, если band `noncompliant` / `compliant_measures`;
- зоны 1/2a/2b hero без балла №55; №127 только как «опора».

## Не сделано

- R7 полировка
- логика `mo_reg55_section.py` не менялась
- `MO_LAB_IN_PRIMARY` не включали
- `rag_server.py` занят #186/#113 (только BUILD_VERSION)

## Нужно

После CLEAN CI: squash-merge, затем с `origin/main`:

```bash
SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
```

Smoke: `/api/version` + разбор случая: hero = зоны, аккордеон №55 = pack-пункты, fail в «Почему так» при noncompliant.

Следующая волна: R7.

Запреты: не Render; не порт 8000; не dirty Cursor `main`; не `SYNC_PROTOCOL_CORPUS=1`; Gemini только GCE; PHI в чат не писать.
