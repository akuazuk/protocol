# Chunk classifier v1

- Dataset: `/Users/pavelkuzauka/Cursor_Folders/Protocol/ml/datasets/chunk_qa_classifier.jsonl`
- Train/test: 22027 / 3957 (split by doc_id)
- Train time: **2.6s** (CPU, HashingVectorizer + OvR LogReg)

## Metrics

- Issues F1 micro: **0.771**
- Issues F1 macro: **0.654**
- needs_action F1: **0.997**

## P0 gate (F1 >= 0.85)

- `preamble_leak`: F1=0.631 support=169 → **FAIL**
- `icd_inflation`: F1=0.000 support=6 → **SKIP**

**Overall gate:** FAIL (do not enable skip-Gemini yet)
