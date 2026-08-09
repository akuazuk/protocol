# Handoff: protocol navigator rebuild (P1–P4)

Дата: 2026-08-09

| | |
|--|--|
| PR | [#96](https://github.com/akuazuk/protocol/pull/96) merged |
| merge SHA | `31fc189f` |
| `BUILD_VERSION` | `2026-08-09-085019Z-mo-protocol-nav` |
| production | GCE `protocol.kravira.by` deployed |
| plan | `docs/plans/2026-08-09-mo-protocol-nav-viewer-v1.md` |

## Сделано

- Разбор: кнопки **Открыть протокол** / **Поиск в каталоге**; `section`/`page` в URL.
- Viewer на `mo-tokens` + `mo-protocol-viewer.css`.
- Never-empty: brief → source_view; **Оригинал · стр. N** на пунктах.
- Deep-link `?section=` / `?page=`.

## Smoke

- `/api/version` = `2026-08-09-085019Z-mo-protocol-nav`
- `mo-protocol-viewer.css` в образе; маркеры `briefFromSourceDoc` / `Открыть протокол` на месте

## Дальше (не блокер)

- Если `#page=` в каком-то браузере не прыгает - PDF.js (вне v1).
- Массовый re-extract тонких Summary - отдельно.
