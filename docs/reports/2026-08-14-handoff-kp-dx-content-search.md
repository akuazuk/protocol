# Handoff: КП по диагнозу и содержанию

- repo: `akuazuk/protocol`
- branch: `cursor/suggest-no-specialty-filler-pc1`
- worktree: `/private/tmp/protocol-task-suggest-no-specialty-filler-pc1`
- PR: https://github.com/akuazuk/protocol/pull/149
- `BUILD_VERSION`: `2026-08-14-083323Z-kp-dx-content-search`

## Сделано

Поиск КП: диагноз → иначе МКБ (RU title) → иначе жалобы/анамнез.
Матч по названию и по содержанию summary (индекс `data/catalog/protocol_content_index.json`).
Нет clinical hit → «Нет клинического протокола МЗ…», без specialty-filler.
Кейс I84.9 / геморрой → КП №22 прямой кишки, не аневризма.

## Не сделано

Merge и GCE deploy. Не трогать PR #146 (Rceth).

## Тесты

`PYTHONPATH=. python3 -m pytest tests/test_icd_mkb.py tests/test_case_protocol_suggest.py tests/test_protocol_content_index.py tests/test_mo_icd_directory_eval.py --noconftest`

## Следующая команда

После merge #146/#149: `GCE_OPS_USER=pavel SYNC_PROTOCOL_CORPUS=0 COPYFILE_DISABLE=1 bash deploy/gcp-app/deploy_to_gce.sh`

Не параллелить: `clinical_knowledge/case_protocol_suggest.py`, `protocol_match.py`, `data/catalog/protocol_content_index.json`.
