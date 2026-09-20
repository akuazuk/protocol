# Handoff: разбор случая R0-R2

Дата: 2026-09-20
Repo: `akuazuk/protocol`
Прод: GCE `protocol.kravira.by` (не Render)

## Сделано

| Волна | PR | Merge SHA | BUILD_VERSION | Прод |
|--|--|--|--|--|
| R0 первый экран | #271 | `34488928` | `2026-09-20-163330Z-mo-r0-case-review` | да |
| план разбора | #270 | `c8c274c5` | - | docs |
| R1 unmatched план | #272 | `ec3acb5f` | `2026-09-20-172034Z-mo-r1-unmatched-plan` | да |
| R2 план ↔ КП | #273 | `1991c638` | `2026-09-20-182937Z-mo-r2-kp-plan` | да |

R2 smoke на проде:

- `/api/version` = `2026-09-20-182937Z-mo-r2-kp-plan`, `git_commit=1991c638fe01`
- `/health/live` ок
- случай без КП: секция «План ↔ КП» пустая, текст «протокол не подобран - сверка плана с КП недоступна», нет «не соответствует протоколу»
- случай с КП: таблица требований (обследование / лечение), клик «к рекомендациям» ставит focus на слот плана
- зоны не пересчитывались

## Не сделано

- R3-R7 runtime ещё нет
- Suggest по-прежнему async после первого кадра
- Карточки КП без exams/treatment дают честный empty, не fail
- `rag_server.py` занят #186/#113 (в R2 только BUILD_VERSION)

## Делается

R3: таймлайн эпизода вместо трёх чисел. Ветка `cursor/mo-case-review-r3-pc1`.

## Нужно

```bash
cd /private/tmp/protocol-task-mo-case-review-r3-pc1
# после тестов: commit, push, PR, squash-merge, SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
```

Запреты: не Render; не порт 8000; не dirty Cursor `main`; не `SYNC_PROTOCOL_CORPUS=1`; Gemini только GCE; PHI в чат не писать.
