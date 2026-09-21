# Handoff: разбор случая R7

Дата: 2026-09-21
Repo: `akuazuk/protocol`
Ветка задачи: `cursor/mo-case-review-r7-pc1` (merged)
Worktree: `/private/tmp/protocol-task-mo-case-review-r7-pc1`
Прод: GCE `protocol.kravira.by` (не Render)

## Сделано

| Волна | PR | Merge SHA | BUILD_VERSION | Прод |
|--|--|--|--|--|
| R0-R6 | #271-#277 | `1c0cd8c8` | `2026-09-21-054235Z-mo-r6-reg55-pack` | да |
| R7 полировка | #278 | `b180b849` | `2026-09-21-080622Z-mo-r7-polish` | да |

Прод проверен: `/health/live` ok, `/api/version` = `2026-09-21-080622Z-mo-r7-polish` / `b180b84954bdaf0a6f6a03066153c3f41d192c72`.

Smoke 3650914 (КП подобран) и 3788833 (протокол не подобран):

- док решения открыт и внизу колонки на 1440;
- клик карточки зоны открывает аккордеон (`evidence-kp-plan` для плана) и слот текста, вкладка документа не сбрасывается;
- «Итог разбора» нет; «Черновик сводки модели» и LLM только под закрытым «Черновик модели»;
- тёмная тема: клиническая колонка на токенах, не `#fbfcfd`;
- 720: `scrollWidth == clientWidth`, горизонтального overflow нет;
- 3788833: чип плана «протокол не подобран», hero без процента №55.

План точности R0-R7 закрыт на проде.

## Не сделано

- `MO_LAB_IN_PRIMARY` не включали
- `rag_server.py` занят #186/#113 (в R7 только BUILD_VERSION)
- computed `position` дока остаётся `static`: более специфичное `.case-workspace-decision .methodist-decision-panel { position: static }` перебивает `--dock { position: sticky }`. На 1440 док всё равно прижат flex-колонкой (проверено `dockInView`). Узкий sticky - отдельный CSS-hotfix при необходимости.
- статус R7 в `docs/plans/2026-09-20-mo-case-review-accuracy-v1.md` в main ещё «in progress» (этот PR писал до merge)

## Нужно

Новых волн этого плана нет. Смежный долг списка (не этот план): SQL-фильтр `critical` / `na` на «Найти МО».

Не включать `MO_LAB_IN_PRIMARY`. Не качать Rceth с Mac.

Запреты: не Render; не порт 8000; не dirty Cursor `main`; не `SYNC_PROTOCOL_CORPUS=1`; Gemini только GCE; PHI в чат не писать.
