# Handoff: разбор случая R5

Дата: 2026-09-21
Repo: `akuazuk/protocol`
Ветка: `cursor/mo-case-review-r5-pc1`
Worktree: `/private/tmp/protocol-task-mo-case-review-r5-pc1`
Прод: GCE `protocol.kravira.by` (не Render)

## Сделано

| Волна | PR | Merge SHA | BUILD_VERSION | Прод |
|--|--|--|--|--|
| R0-R4 | #271-#275 | `ef8771c5` | `2026-09-20-193442Z-mo-r4-lab-evidence` | да, до этой волны |
| R5 назначения Rceth + КП | этот PR | нет до merge | см. `BUILD_VERSION` коммита | нет до GCE |

R5:

- ложный бейдж «черновик» только если нет инструкции Rceth и нет схемы КП;
- 4.1 / 4.3 и дата редакции из уже скачанного `load_rceth_label_ctx`;
- сверка INN со схемой КП (drugs / drug_groups), иначе честно «в КП нет препаратов» / «протокол не подобран»;
- DDI и high-alert в колонке «Риск», не в зоне плана;
- `primary` карточек по-прежнему false.

## Не сделано

- R6 №55 читаемый, R7 полировка
- Suggest по-прежнему async после первого кадра
- Rceth не качали с Mac, PDF инструкций в git нет
- `MO_LAB_IN_PRIMARY` не включали
- `rag_server.py` занят #186/#113 (только BUILD_VERSION)

## Нужно

После CLEAN CI: squash-merge, затем с `origin/main`:

```bash
SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
```

Smoke: `/api/version` + секция «Назначения» на случае с КП и без КП.

Следующая волна: R6.

Запреты: не Render; не порт 8000; не dirty Cursor `main`; не `SYNC_PROTOCOL_CORPUS=1`; Gemini только GCE; PHI в чат не писать.
