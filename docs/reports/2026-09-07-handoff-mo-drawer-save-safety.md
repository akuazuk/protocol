# Handoff: CASE Review save safety D6-D8

Дата: 2026-09-07

## Состояние

- Branch: `cursor/mo-case-drawer-save-safety-agent1-pc1`
- Worktree: `/private/tmp/protocol-task-mo-drawer-safety-pc1`
- Base: `6911db7cbc5af14b46f39312d95339258ce2763f`
- BUILD_VERSION: `2026-09-07-043518Z-drawer-save-safety`
- PR: будет указан после публикации

## Реализация

- D6: изменение решения переводит форму в `Черновик`; Close, Esc, Previous, Next и
  переход из истории требуют явного подтверждения при несохранённых данных.
- После загрузки сохранённого pack или нового protocol-suggest baseline формы
  пересчитывается без ложного dirty state.
- D7: save передаёт `expected_document_revision`, `evaluation_run_id` и
  `Idempotency-Key`.
- Сервер отклоняет устаревшую document revision с HTTP 409.
- Повтор POST с тем же ключом для actor/case возвращает тот же pack и не создаёт
  вторую CRM запись.
- Review pack хранит document revision и evaluation run в колонках и system snapshot.
- D8: существующий focus trap, возврат фокуса и keyboard navigation дополнены
  безопасным Esc; sticky panel визуально отмечает несохранённый черновик.

Клинические пороги, веса и primary/shadow флаги не менялись.

## Проверки

- Focused Python suite: 49 passed.
- Полный browser smoke: 15 passed.
- Synthetic backend: idempotent replay создаёт одну строку; stale revision отклонена.
- Synthetic browser: dirty guard, cancel/confirm navigation, HTTP 409 UX,
  idempotency-key reuse и revision lineage - успешно.
- Python compile, JS syntax, `git diff --check` и IDE diagnostics - успешно.

## Production baseline

- Production SHA: `6911db7cbc5af14b46f39312d95339258ce2763f`.
- Production version: `2026-09-07-041400Z-case-drawer-d1-d5`.
- Exact version/commit, live drawer at 390 px и reports smoke успешны.
- D6-D8 ещё не merged и не deployed.

## Следующий безопасный шаг

Required CI → merge через GitHub → exact-main GCE deploy → production save-safety smoke
без записи клинического решения. Затем перейти к clinical context B/C.

## Не трогать параллельно

- `clinical_knowledge/mo_review_pack.py`
- `frontend/web/shared/mo-app.js`
- `frontend/web/shared/mo-ui.css`
- `rag_server.py`
