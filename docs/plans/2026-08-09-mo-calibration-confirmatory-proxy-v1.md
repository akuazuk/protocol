# МО: exploratory confirmatory proxy C8A/C9A

Дата: 2026-08-09
Статус: **active** (blind 100/100 done; Pro-proxy in progress; C8B unstable)

Связанные планы:
- `2026-08-09-mo-score-ssot-llm-recompute-v3.md`
- `2026-08-09-mo-calibration-agent-proxy-v1.md`

## Контекст

Formal C6 human labels остаются 0/22; owner waived gate. C6B уже заполнил
22 LLM-proxy labels и C7; C8B по этому gold - `no_stable_provisional`
(слишком мало bad). C8A остаётся exploratory по C7A proxy aggregate.
C9A: July cohort 100, blind flash готов (parse 99/100), Pro-proxy в работе.

## Что изменено в production

- Production scoring, thresholds, action queue и SSOT не меняются.
- Human label file и methodist UI не затрагиваются.
- Confirmatory artifacts живут в отдельной GCE directory
  `calibration/mo-score-v3-confirmatory-YYYY-MM-DD-YYYY-MM-DD/`.

## Дизайн

1. C8A: `select_mo_calibration_provisional.py` по PHI-safe C7A aggregate.
2. C9A period: `2026-07-26..2026-07-31`, seed `43`, `--no-sentinel`.
3. Exclude pilot keys from
   `mo-score-v3-2026-08-01-2026-08-08/secret/secret_manifest.jsonl`.
4. Blind arm: `gemini-3.6-flash`, 100×1.
5. Proxy arm: `gemini-3.1-pro-preview`, 100×1.
6. Aggregate + provisional selector; production rollout всегда `false`.

## Метрики

- C8A Dx: `no_stable_provisional` (proxy-bad=2).
- C8A Plan: `provisional_shadow:blind.pass_1`.
- C9A: selected_n=100; blind parse 99/100, leakage/geo=0; proxy in progress.
- C8B (C6B gold): Dx/Plan `no_stable_provisional`.
- Formal C6 human: остаётся `0/22`.

## Шаги

- [x] B0 Зафиксировать exploratory C8A/C9A без подмены human gold.
- [x] B1 Sampler: `--no-sentinel`, `--exclude-manifest`, scaled floors.
- [x] B2 GCE mode `--calibration-confirmatory-proxy`.
- [x] B3 C8A provisional из pilot proxy aggregate.
- [ ] B4 GCE July sample 100 + blind + proxy + eval (blind done; proxy running).
- [ ] B5 PHI-safe report/PR; production decision запрещён.

## Риски

- Proxy ≠ methodist gold.
- July pool может не закрыть редкие strata; тогда уменьшить floors или расширить период.
- 200 LLM case-judges дороги по времени; resume обязателен.
- Arm D fingerprint confirmatory не обязан совпадать с pilot.
