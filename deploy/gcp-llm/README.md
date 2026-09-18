# deploy/gcp-llm

Night LLM + action-judge на GCP (эпоха E1: та же VM `protocol-app`).

## Ops (канон B3)

```bash
# smoke Gemini с GCE (1 кейс)
bash deploy/gcp-llm/run_on_gce.sh 2026-08-06 --smoke

# день / диапазон в фоне (внутри container protocol-web)
bash deploy/gcp-llm/run_on_gce.sh 2026-08-06
bash deploy/gcp-llm/run_on_gce.sh 2026-08-01 2026-08-06

# foreground
bash deploy/gcp-llm/run_on_gce.sh 2026-08-06 --foreground
```

Логи: `/var/data/medical_exams/logs/mo_llm_backfill_<first>_<last>.log`

## Важно

- **Не** запускать `grade_kz_llm` на Mac (geo-block).
- Render SSH (`run_mo_render_llm_backfill.sh`) - legacy, пока Render primary writer.
- Пока LLM job крутится в образе `protocol-gcp-app` / container `protocol-web` на том же PD (общие deps + ключи). Отдельный thin `protocol-gcp-llm` образ - следующий harden.
- Двойной старт: runner проверяет `pgrep grade_kz_llm` в контейнере.

## Расход Gemini (с 2026-09-17)

Ночь по умолчанию:

- `thinking_budget=0`, JSON output ≤ 2048 токенов
- Pro-эскалация только parse / confidence < 0.6 / harm disagreement, не `needs_human`
- shadow ≤ 30 кейсов/день (`MO_SHADOW_DX_PLAN_LIMIT=0` вернёт «все»)
- action-judge ≤ 20 и только если локальный jsonl читается
- bulk Batch API best-effort (`GEMINI_NIGHT_USE_BATCH=0` выключает)
- spend-cap не ретраится после смены ключа

Алерты: `bash deploy/gcp-llm/ensure_gemini_spend_budget.sh` (проект `gen-lang-client-0274478609`).

Контракт inbox/outbox: [job-contract.md](job-contract.md).
