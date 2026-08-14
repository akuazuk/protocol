# Handoff: непрерывность эпизода, backfill ключей, deep run

Дата: 2026-08-14

## Repo

- merged: https://github.com/akuazuk/protocol/pull/145
- production SHA: `2d2243ebdd40a81b812e05ce68ac34a161cc34b3`
- `BUILD_VERSION` на GCP: `2026-08-14-072703Z-history-backfill-bulk`
- follow-up branch: `cursor/history-deep-billed-key-pc1` (слой C сам берёт billed key)

## Сделано

- Слои A/B/C в коде, официальный `overall_pct` не переписывается.
- Deploy на `https://protocol.kravira.by`: `/health/live` ok, version совпал.
- Backfill `patient_key` из MIS identity (без `result`): склад 100129/100129 с ключом, январь-август 100%.
- Refresh `history_prior_n` / `history_tier` за июль-август: июль 6552 с prior, август 2842.
- Слой B за 2026-08-13: 20 случаев, 16 `known_episode_same_doctor`, 13 с загруженными слотами prior.
- Слой C `--llm` за 13.08: 15/15 через `GENERATIVE_LANGUAGE_API_KEY` + `gemini-3.6-flash`.
  Вердикты: review 4, good 3, poor 4, acceptable 4; history_explains_gap 3.
- Почему первый прогон упал: оба AI Studio ключа (`GOOGLE_API_KEY`, `_2`) на monthly spend cap
  с 09.08. Раннер брал только первый ключ и не крутил billed `AQ…`.

## Не сделано

- Поднять spend cap в AI Studio (`https://ai.studio/spend`), иначе night/lite путь снова 429.
- Полный refresh `history_prior_n` за январь-июнь (очередь дня уже видит июль-август).
- Другая вкладка: PR #146 rceth labels - не пересекаться.

## Тесты

`tests/test_mo_history_continuity.py`, `tests/test_mo_history_deep.py`, `tests/test_mo_patient_key_backfill.py`.

## Следующая команда

Не гонять Gemini с Mac. Не трогать PR #146 (rceth). После merge этого follow-up:

```bash
GCE_OPS_USER=pavel SYNC_PROTOCOL_CORPUS=0 COPYFILE_DISABLE=1 bash deploy/gcp-app/deploy_to_gce.sh
```

## Не трогать параллельно

- `scripts/backfill_mo_patient_keys_from_mis.py`
- `scripts/run_mo_history_deep.py`
- `clinical_knowledge/mo_history_continuity.py`
- `clinical_knowledge/mo_history_deep.py`
