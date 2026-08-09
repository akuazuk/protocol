# Handoff: MO protocol Study reader (v2)

Дата: 2026-08-09

## Git

| | |
|--|--|
| repo | `akuazuk/protocol` |
| branch | `cursor/mo-protocol-nav-reader-pc1` |
| worktree | `/private/tmp/protocol-task-mo-protocol-nav-reader-pc1` |
| base | `403b9438` (+ plan cherry-pick) |
| HEAD | `2dce015c` |
| PR | (создаётся в той же сессии) |
| BUILD_VERSION | `2026-08-09-094029Z-mo-protocol-reader` |

## Сделано

- Backend: `clinical_knowledge/protocol_reader.py`, `GET /api/protocol-reader`
- UI: full-width Study (абзацы) + Visit (brief); поиск, deep-link `section`/`page`, PDF page links
- Тесты: `tests/test_protocol_reader.py`, обновлён `test_mo_protocol_nav_viewer.py`
- План: `docs/plans/2026-08-09-mo-protocol-nav-reader-v2.md` R0-R4

## Не сделано

- Merge PR + GCE deploy (`SYNC_PROTOCOL_CORPUS=0`)
- Smoke 20 КП на проде
- R6 PDF.js split
- Закрыть docs-only PR #101 после merge

## Тесты

```bash
pytest tests/test_protocol_reader.py tests/test_mo_protocol_nav_viewer.py -q
```

## Следующая команда

```bash
# после merge origin/main:
SYNC_PROTOCOL_CORPUS=0 COPYFILE_DISABLE=1 bash deploy/gcp-app/deploy_to_gce.sh
```

## Не трогать параллельно

- `frontend/web/shared/proto-viewer.html`, `mo-protocol-viewer.css`
- `clinical_knowledge/protocol_reader.py`, `/api/protocol-reader` в `rag_server.py`
