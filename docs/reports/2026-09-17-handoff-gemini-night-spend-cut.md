# Handoff: срез расхода ночного Gemini

Дата: 2026-09-17
Repo: `akuazuk/protocol`
Branch: `cursor/gemini-night-spend-cut-agent1-pc1`
Worktree: `/private/tmp/protocol-task-gemini-night-spend-cut-pc1`
PR: https://github.com/akuazuk/protocol/pull/249
HEAD: `6b1f730d`

## Сделано

Ночной контур МО больше не гоняет shadow по всем визитам и не эскалирует
очередь 80 на Pro из-за `needs_human`. Thinking выключен, JSON урезан,
spend-cap не ретраится, action-judge читает локальный jsonl.

Код в task-ветке. В проде нет, пока нет merge + `deploy_to_gce.sh`.

## Не сделано

- Merge / GCE deploy
- Budget alert: `bash deploy/gcp-llm/ensure_gemini_spend_budget.sh`
  (проект `gen-lang-client-0274478609`, биллинг `01D5C2-ECFF77-88FFEC`)

## Следующая команда

После merge координатором:

```bash
bash deploy/gcp-app/deploy_to_gce.sh
bash deploy/gcp-llm/ensure_gemini_spend_budget.sh
```

Smoke после первой ночи: `shadow.jsonl` ≤ 30 строк, в grades нет массового
`_grader_model=gemini-3.1-pro-preview`, `/api/version` совпал с BUILD_VERSION.

## Не трогать параллельно

- `scripts/grade_kz_llm.py`
- `scripts/mo_llm_range_runner.sh`
- `scripts/run_mo_shadow_dx_plan.py`
- `scripts/run_mo_action_queue_llm_judge.py`
- `clinical_knowledge/gemini_lite.py`
- `clinical_knowledge/gemini_batch.py`
- `config/llm_pricing.yaml`
