# Handoff: Ilex-паспорта КП (шаг 3, к merge)

Дата: 2026-09-18

## Repo

- worktree: `/private/tmp/protocol-task-ilex-kp-passports-pc1`
- branch: `cursor/ilex-kp-passports-agent1-pc1`
- PR: https://github.com/akuazuk/protocol/pull/250
- `BUILD_VERSION`: `2026-09-18-190626Z-ilex-kp-2026-wave3`

## Сделано

- Паспорта Ilex + overlay (шаг 1).
- АГ N 38 и аритмии N 34 (шаг 2).
- Шаг 3: 56 PDF 2026 с сайта МЗ (копии по рубрикам). Карты 5193 → 5552.
  Ключи Ilex 2026: 28/30. Нет на сайте как 2026-PDF: ревматология N 3, N 77.
- Suggest: recency на тексте; ОКС-алиас; `(дети)` = child; текстовый путь
  не ранжирует по МКБ случая. Главы Ilex → `ilex_chapters` в карте и suggest.
- Eval overlay on: 4/4, golden 40/40, чужих 0. ОКС/ТЭЛА/стенокардия/инсульт/
  бронхит 2026 в top-1.
- HTML Ilex в git нет. Усечённый `chunks.jsonl` не коммитить.

## Не сделано в этом PR

- Nightly Ilex API на GCE (`scripts/ops/` - отдельный PR).
- Полный plan-score по главам (поле `ilex_chapters` уже на карте).
- Ревматология N 3 и N 77, пока МЗ не выложит 2026-PDF.
- Deploy. После merge: `bash deploy/gcp-app/deploy_to_gce.sh`.

## Не трогать параллельно

`clinical_knowledge/ilex_protocol_passports.py`, `protocol_match.py`,
`case_protocol_suggest.py`, `dx_query_expand.py`, `applicability.py`,
`output/registry/protocol_cards.jsonl`, `minzdrav_protocols/`.
`rag_server.py` - только `BUILD_VERSION`.
