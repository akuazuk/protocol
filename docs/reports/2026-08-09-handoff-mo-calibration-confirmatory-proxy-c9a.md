# Handoff: MO calibration confirmatory proxy C8A/C9A

Дата: 2026-08-09

## Git

- Repo: `akuazuk/protocol`
- Branch: `cursor/mo-calibration-confirmatory-proxy-c9a-pc1`
- Worktree: `/private/tmp/protocol-task-mo-calibration-confirmatory-proxy-c9a-pc1`
- Base: `a76f93f3` (PR #112 merge)
- BUILD_VERSION: `2026-08-09-161306Z-mo-calibration-c9a`
- HEAD / PR: заполняются после push.

## Что сделано

1. C8A provisional methodology из pilot proxy aggregate:
   - Dx: `no_stable_provisional` (только 2 proxy-bad).
   - Plan: `provisional_shadow:blind.pass_1`.
   - Production rollout: запрещён.
2. Sampler:
   - `--no-sentinel` для независимого периода;
   - `--exclude-manifest` против overlap с pilot 30;
   - умеренно scaled floors для target=100;
   - KP-pool scaling; large-N skips O(n^2) local repair.
3. GCE mode `--calibration-confirmatory-proxy`:
   - directory `mo-score-v3-confirmatory-${FIRST}-${LAST}`;
   - July `2026-07-26..2026-07-31`, seed 43, n=100;
   - blind flash + proxy pro + aggregate + provisional.
4. GCE run started for the July confirmatory cohort.

## Что не сделано / не закрыто

- Formal C6 human labels остаются `0/22`.
- Formal C7-C9 не закрыты.
- Production scoring/SSOT не менялись.
- Полный July LLM прогон ещё выполняется на GCE.

## Проверки

- Sampler/provisional/proxy tests: passed.
- `bash -n deploy/gcp-llm/run_on_gce.sh`: passed.
- `ruff`: passed.

## Следующий безопасный шаг

Дождаться GCE confirmatory completion, скачать PHI-safe aggregates, обновить
отчёт и открыть/дополнить PR. Human labeling остаётся отдельным gate.
