# C6 methodist labeling guide

## Scope

Label all 22 disagreement endpoints from the frozen C5 pilot:

- 18 unique cases;
- 9 diagnosis-evidence labels;
- 13 diagnosis-conditioned plan labels;
- 4 cases require both endpoint labels.

This meets the pre-registered minimum of 15 cases only after every row is complete
and the validator passes.

## PHI boundary

Work only on GCE `protocol-app` inside:

```text
/var/data/medical_exams/calibration/mo-score-v3-2026-08-01-2026-08-08/secret/methodist/
```

Do not copy `methodist_cases.jsonl` or `methodist_labels.jsonl` to git, chat,
PRs, email, or a local Mac.

Files:

- `methodist_cases.jsonl` - blinded clinical cases and allowed protocol context;
- `methodist_labels.jsonl` - the only file the reviewer edits;

The pack does not show engine, LLM pass, or LLM adjudication outputs during human
review. The comparison file is not created until label validation passes. This
avoids anchoring.

## Required fields

For each row in `methodist_labels.jsonl`, fill:

- `verdict`: `good`, `partial`, `poor`, `critical`, `blocked`, or `na`;
- `score_pct`: integer or decimal from 0 to 100;
- `potential_harm`: JSON boolean `true` or `false`;
- `icd_fit`: for Dx, `fit`, `partial`, `mismatch`, `unknown`, or `na`; for plan
  keep `na`;
- `confidence`: 0 to 1;
- `rationale`: at least 10 characters with the clinical reason;
- `reviewer_id`: stable methodist identifier without patient information;
- `reviewed_at`: ISO timestamp.

Do not add patient identifiers or copy clinical text into the rationale.

## Web form

After deployment, sign in through the methodist cabinet and open:

```text
/methodist/calibration
```

The form:

- is available only to methodist, lead, and admin roles;
- never shows engine scores, LLM passes, or LLM adjudication;
- writes reviewer identity and timestamp on the server;
- does not keep clinical cases in browser storage;
- records open/save actions without clinical text;
- unlocks the comparison only after all 22 labels pass validation.

## Validate

On GCE, after all labels are saved:

```bash
sudo docker exec protocol-web python scripts/build_mo_calibration_methodist_pack.py \
  --cases /var/data/medical_exams/calibration/mo-score-v3-2026-08-01-2026-08-08/secret/secret_cases.jsonl \
  --pilot /var/data/medical_exams/calibration/mo-score-v3-2026-08-01-2026-08-08/secret/blind_pilot.jsonl \
  --secret-out-dir /var/data/medical_exams/calibration/mo-score-v3-2026-08-01-2026-08-08/secret/methodist \
  --labels /var/data/medical_exams/calibration/mo-score-v3-2026-08-01-2026-08-08/secret/methodist/methodist_labels.jsonl \
  --public-status /var/data/medical_exams/calibration/mo-score-v3-2026-08-01-2026-08-08/methodist_status.json
```

C6 is complete only when public status reports:

```text
expected_label_n=22
complete_label_n=22
case_n=18
missing_n=0
extra_n=0
passed=true
```

Only after that may C7 unseal comparisons and calculate candidate/ensemble metrics.
The validator then creates `methodist_llm_comparison_unsealed.jsonl` outside the
review directory for C7.
