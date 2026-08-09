# Handoff: C6B LLM-proxy labels + C7

Дата: 2026-08-09

## Repo

- repo: `akuazuk/protocol`
- branch: `cursor/mo-calibration-confirmatory-proxy-c9a-pc1`
- worktree: `/private/tmp/protocol-task-mo-calibration-confirmatory-proxy-c9a-pc1`
- PR: https://github.com/akuazuk/protocol/pull/113
- BUILD_VERSION: `2026-08-09-171239Z-c6b-blocked-score-sanitize`

## Сделано

- Owner waived human methodist gate (вариант 1).
- GCE mode `--calibration-llm-methodist-labels`.
- C6 pack: `22/22` labels, `passed=true`, `comparison_unsealed=true`,
  reviewer `llm_proxy_c6b_not_human_gold`, model `gemini-3.1-pro-preview`.
- C7 against proxy-gold: `production_decision_allowed=false`.
  - Dx: labeled 7 (non-blocked/na), bad 1; top `blind.mean_2`.
  - Plan: labeled 7, bad 2; top `snapshot.overall_v3`.
- PHI-safe: `eval/mo_score_calibration/c6b_c7_public_summary.json`.

## Не сделано

- Formal human C6 labels всё ещё 0/22.
- C8 formal methodology / production thresholds не выбирались.
- C9A July confirmatory на GCE ещё бежал (blind ~88/100 на момент C6B).
- Commit `88480a10` содержит нежелательный `Co-authored-by: Cursor` trailer
  (hook не снял; уже push). Чинить только через amend + force-with-lease по
  явной просьбе владельца.

## Production

- Deploy не запускался; scoring / SSOT / queue не менялись.

## Safe next

```bash
# после owner review ranking:
# либо formal C8 draft thresholds (shadow only),
# либо дождаться C9A confirmatory и сверить.
gcloud compute ssh protocol-app --zone=europe-central2-a --command \
  "sudo wc -l /var/data/medical_exams/calibration/mo-score-v3-confirmatory-2026-07-26-2026-07-31/secret/blind_confirmatory.jsonl"
```

## Не трогать параллельно

- GCE pilot dir `calibration/mo-score-v3-2026-08-01-2026-08-08/secret/`
- `methodist_labels.jsonl` (уже proxy-filled)
- production scoring paths
