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

Контракт inbox/outbox: [job-contract.md](job-contract.md).
