# Embedder experiment 001

- Base model: `intfloat/multilingual-e5-small`
- Train pairs (resolved): **313**
- Golden hold-out queries: **13**
- Epochs: 2

## Golden hold-out (baseline e5-small)

- MRR@10: **0.1673**
- Recall@1: **0.1538**

## Golden hold-out (fine-tuned final)

- MRR@10: **0.3294**
- Recall@1: **0.2308**

## 5-fold CV (by protocol path)

- Baseline MRR mean: **0.1246**
- Fine-tuned MRR mean: **0.4137**
- Delta: **0.2891**

- Fold 0: baseline MRR=0.0917 → finetuned=0.4211 (Δ=0.3294)
- Fold 1: baseline MRR=0.1622 → finetuned=0.4725 (Δ=0.3103)
- Fold 2: baseline MRR=0.0876 → finetuned=0.3588 (Δ=0.2712)
- Fold 3: baseline MRR=0.1066 → finetuned=0.3955 (Δ=0.2889)
- Fold 4: baseline MRR=0.1749 → finetuned=0.4208 (Δ=0.2459)

Elapsed: 4999.4s