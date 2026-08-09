# МО: C6B LLM-proxy labels как substitute gold + formal C7

Дата: 2026-08-09
Статус: **completed** (C6B/C7/C8B done; superseded for methodology by C9A)

Связанные планы:
- `2026-08-09-mo-score-ssot-llm-recompute-v3.md`
- `2026-08-09-mo-calibration-agent-proxy-v1.md` (C6A/C7A exploratory)
- `2026-08-09-mo-calibration-confirmatory-proxy-v1.md` (C8A/C9A)

## Контекст

Formal C6 ждал 22 human labels (0/22). Владелец выбрал вариант 1:
LLM заполняет C6 pack на GCE, дальше C7-C9 идут против этого proxy-gold.
Production scoring / SSOT / queue не меняются.

Отличие от C6A/C7A: labels пишутся в `methodist_labels.jsonl` через тот же
`save_label` / unseal path, с `reviewer_id=llm_proxy_c6b_not_human_gold`.
Это не human gold и не закрывает «методистскую» часть, но разблокирует
сравнение семейств scores на том же seal/unseal контракте.

## Что изменено в production

- Ничего. Thresholds, action queue, warehouse SSOT и UI scores не трогаем.

## Дизайн

1. `scripts/run_mo_calibration_llm_methodist_labels.py` - blind Pro labels для
   pending C6 endpoints; GCE-only live contour.
2. После `audit.passed` сервер unseal'ит comparison как для human path.
3. `scripts/eval_mo_score_calibration_c7.py` - C7 vs labels,
   `gold_kind=llm_proxy_c6b`, `production_decision_allowed=false`.
4. GCE mode: `bash deploy/gcp-llm/run_on_gce.sh 2026-08-01 2026-08-08 --calibration-llm-methodist-labels`.

## Метрики

- Было: `complete_label_n=0/22`, `comparison_unsealed=false`.
- Стало C6B: `22/22` complete, `passed=true`, `comparison_unsealed=true`,
  reviewer `llm_proxy_c6b_not_human_gold`, model `gemini-3.1-pro-preview`.
- C7 против scored (non-blocked/na) gold: Dx labeled 7 / bad 1;
  Plan labeled 7 / bad 2; `production_decision_allowed=false`.
- Top Dx: `blind.mean_2` (PR-AUC 1.0 на n=7, нестабильно).
- Top Plan: `snapshot.overall_v3` (PR-AUC 0.75).
- PHI-safe: `eval/mo_score_calibration/c6b_c7_public_summary.json`.

## Шаги

- [x] L0 Owner waiver: human methodist gate не блокирует proxy path.
- [x] L1 Скрипт LLM labels + C7 eval + unit tests.
- [x] L2 GCE mode `--calibration-llm-methodist-labels`.
- [x] L3 GCE: заполнить 22 labels, unseal, audit passed.
- [x] L4 C7 aggregate против LLM-proxy gold; PHI-safe отчёт.
- [x] L5 C8B provisional из C7/C6B: оба endpoint `no_stable_provisional`
  (bad-n 1/2 < gate); артефакт
  `eval/mo_score_calibration/provisional-methodology-c8b.json`.
  Production decision запрещён; дальше - C9A confirmatory.

## Риски

- Proxy ≠ human gold; CIs и ranking provisional.
- Параллельный C9A confirmatory может делить Gemini quota.
- Нельзя писать эти labels как methodist review в UI/отчётах для людей.
