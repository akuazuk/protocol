# Handoff: МО ICD name_only match v2 (фазы A+B)

Дата: 2026-08-08  
Repo: akuazuk/protocol  
Branch: `cursor/mo-icd-name-match-v2-pc1`  
Worktree: `/private/tmp/protocol-task-mo-icd-name-match-v2-pc1`  
Base: `origin/main` @ `46c3a7b8`  
BUILD_VERSION: `2026-08-08-064315Z-icd-name-match-v2`

## Сделано

- План: `docs/plans/2026-08-08-mo-icd-name-match-v2.md` (v1 в индексе archived)
- A: re-export idempotent (15616 rows, sha совпал); `icd10_ru_mkb10su.meta.json`; export пишет meta
- B: `clinical_text_similarity.py` (normalize/strip codes/combined + `score_against_sections` stub для фазы D)
- B: `mo_icd_name_match.py` shadow findings `B_icd_name_no_match` / `B_icd_name_weak_match`
- Wire: `rag_server` case detail, `kz_deep_eval`, RU labels
- Тесты: `tests/test_clinical_text_similarity.py`, `tests/test_mo_icd_name_match.py` (8 passed)

## Не сделано

- C: калибровка порогов на gold / день GCE; primary flag
- D: findings по Dx ↔ жалобы/анамнез/обследования/план (есть только stub API)
- Deploy GCE после merge

## Deploy

Не запускался. После merge: `bash deploy/gcp-app/deploy_to_gce.sh`, smoke `/api/version`.

## Следующая команда

После merge PR - deploy на GCE и выборка name_fit на дне склада.

## Не трогать параллельно

- `clinical_knowledge/mo_icd_directory_eval.py` (v1)
- `data/icd_reference/icd10_ru_mkb10su.json` без согласованного обновления xlsx
