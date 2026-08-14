# Handoff: непрерывность эпизода, backfill ключей, deep run

Дата: 2026-08-14

## Repo

- merged: https://github.com/akuazuk/protocol/pull/145
- production SHA: `e24ee662325dec95269a644e74ed7d6a8d6f17d4`
- `BUILD_VERSION` на GCP: `2026-08-14-065244Z-history-deep`
- hotfix branch: `cursor/history-backfill-bulk-pc1` (bulk UPDATE + без pydantic на старте)

## Сделано

- Слои A/B/C в коде, официальный `overall_pct` не переписывается.
- Deploy на `https://protocol.kravira.by`: `/health/live` ok, version совпал.
- Backfill `patient_key` из MIS identity (без `result`): склад 100129/100129 с ключом, январь-август 100%.
- Refresh `history_prior_n` / `history_tier` за июль-август: июль 6552 с prior, август 2842.
- Слой B за 2026-08-13: 20 случаев, 16 `known_episode_same_doctor`, 13 с загруженными слотами prior.
- Слой C `--llm` за 15 случаев: раннер отработал, все 15 `ResourceExhausted` (квота Gemini).

## Не сделано

- Повтор слоя C, когда квота Gemini снова жива.
- Полный refresh `history_prior_n` за январь-июнь (очередь дня уже видит июль-август).

## Тесты

`tests/test_mo_history_continuity.py`, `tests/test_mo_history_deep.py`, `tests/test_mo_patient_key_backfill.py`.

## Следующая команда

Когда Gemini ответит:

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a --project=protocol-home-e1 \
  --command='bash /opt/protocol/deploy/gcp-app/run_history_deep_on_gce.sh yesterday 15 --llm'
```

## Не трогать параллельно

- `scripts/backfill_mo_patient_keys_from_mis.py`
- `scripts/run_mo_history_deep.py`
- `clinical_knowledge/mo_history_continuity.py`
- `clinical_knowledge/mo_history_deep.py`
