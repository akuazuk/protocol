# Handoff: MO calibration AI-proxy C6A/C7A

Дата: 2026-08-09

## Git

- Repo: `akuazuk/protocol`
- Branch: `cursor/mo-calibration-agent-proxy-c6a-pc1`
- Worktree: `/private/tmp/protocol-task-mo-calibration-agent-proxy-c6a-pc1`
- Base: `1e6a3773`
- Published branch head at PR creation: `63f0f2cd`
- PR: [#112](https://github.com/akuazuk/protocol/pull/112)
- BUILD_VERSION: `2026-08-09-154205Z-mo-calibration-proxy`

## Что сделано

1. На GCE выполнен отдельный blind proxy arm:
   `gemini-3.1-pro-preview`, 30 frozen cases, по одному проходу Endpoint C/D.
2. Proxy не видел deterministic scores, предыдущие LLM passes, adjudication или
   labels методиста. Human label file не изменялся.
3. Добавлен aggregate evaluator:
   - MAE и Spearman;
   - ROC-AUC и PR-AUC для proxy-bad;
   - confusion metrics на threshold 55;
   - bootstrap 95% CI, 2,000 итераций;
   - endpoint-specific primary candidates и отдельные control scores.
4. Secret proxy rows оставлены только на GCE. В git добавлен только PHI-safe
   aggregate без source/sample IDs и clinical text.
5. Добавлен воспроизводимый GCE mode `--calibration-agent-proxy`.

## Результат proxy run

- Rows: `30`.
- Dx payload: `30/30`; scored non-abstain: `25`; proxy-bad: `2`; abstain: `5`.
- Plan payload: `29/30`; scored non-abstain: `21`; proxy-bad: `8`; abstain: `8`.
- Один grounded plan дважды нарушил contract nullability
  (`blocked/na` вместе со score). Валидный Dx этого row сохранён, plan исключён.
- Leakage failures: `0`; geo errors: `0`; route coverage: passed.
- Judge fingerprint:
  `147c5db8a35f7c2c294a6d322e4a2997303f8d1108d4b147abe73aff3298c760`.

## Exploratory C7A

### Diagnosis evidence

- Proxy-bad всего `2`, поэтому ranking статистически неустойчив.
- Лучший из primary candidates по PR-AUC:
  blind adjudicated-or-mean `0.233` (95% CI `0.048-1.000`);
  ROC-AUC `0.607`.
- Snapshot zone2a: PR-AUC `0.133` (95% CI `0.056-0.324`);
  ROC-AUC `0.375`.
- Arm D clinical concordance: PR-AUC `0.127`; ROC-AUC `0.457`.
- Ни один Dx score нельзя выбирать primary по этому proxy cohort.

### Plan concordance

- Blind pass 1: PR-AUC `0.594` (95% CI `0.264-0.865`);
  ROC-AUC `0.635`; threshold-55 sensitivity `0.75`, specificity `0.58`.
- Blind adjudicated-or-mean: PR-AUC `0.558` (95% CI `0.239-0.823`);
  ROC-AUC `0.591`.
- Arm D + blind ensemble: PR-AUC `0.496` (95% CI `0.254-0.838`);
  ensemble не улучшил blind pass.
- Snapshot clinical concordance: PR-AUC `0.432`.
- Snapshot zone2b не имел достаточного покрытия для primary comparison. Это
  отдельный gap score availability, а не доказательство качества zone2b.
- CI сильно перекрываются. Blind plan выглядит перспективнее deterministic
  alternatives, но provisional methodology не выбрана.

## Self-check по этапам

- A0 boundary: proxy не записан в methodist labels; formal C6 остаётся `0/22`.
- A1 implementation: synthetic perfect/inverse, partial-endpoint, abstention и
  PHI-safe tests пройдены.
- A2 runtime: только GCE, model/inputs/evaluator захешированы, leakage `0`.
- A3 metrics: endpoint-specific candidates, controls отдельно, bootstrap CI есть.
- A4 interpretation: human gold не заявлен, production decision запрещён.

## Проверки

- `pytest`: 25 calibration regression tests passed.
- Proxy evaluator tests: 3 passed.
- `ruff`: passed.
- `bash -n deploy/gcp-llm/run_on_gce.sh`: passed.
- `git diff --check`: passed.

## Что не сделано

- Formal C6: human labels `0/22`.
- Formal C7-C9 не начаты.
- Thresholds, production scoring, action queue и SSOT не изменены.
- Merge/deploy отсутствуют; GCE primary application code не перезапускался.

## Следующий безопасный шаг

Реальный методист открывает `https://protocol.kravira.by/methodist/calibration`
и заполняет 22 blind labels. После gate `22/22` повторить evaluator уже против
human gold.

## Не трогать параллельно

- `scripts/eval_mo_score_agent_proxy.py`
- `deploy/gcp-llm/run_on_gce.sh`
- `eval/mo_score_calibration/agent-proxy-summary.json`
- `docs/plans/2026-08-09-mo-calibration-agent-proxy-v1.md`
- `docs/reports/2026-08-09-handoff-mo-calibration-agent-proxy-c6a.md`
