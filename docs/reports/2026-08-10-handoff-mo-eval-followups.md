# Handoff: MO eval quality followups (A–D)

Дата: 2026-08-10  
Repo: `akuazuk/protocol`  
Branch: `cursor/mo-eval-followups-impl-pc1`  
Worktree: `/private/tmp/protocol-task-mo-eval-followups-impl-pc1`  
Base: `origin/main` @ `2e20ce50` (severity labels #128)

## Сделано

- **A:** fix `мелоксикам`→`diclofenac` в `_overrides` (zip ru/inn); бренды SSRI/триптанов/мелоксикама; DDI title surface+INN; unit-регрессия 3600047.
- **B:** справка: приоритет ≠ статус «Критично»; потолки ≠ таксономия.
- **C:** job mode `deep_rescore` (settings + `rescore_mo_deep_days --update-primary-score` + warehouse recompute).
- **D:** `worst_severity_cases` + фильтр `worst_severity`; кольцо на Обзоре; `MO.moDonut` (толстые кольца); chip фильтра.
- План `docs/plans/2026-08-10-mo-eval-quality-followups-v2.md` обновлён (GCE re-score / smoke ещё open).
- `BUILD_VERSION`: `2026-08-10-165444Z-mo-eval-followups`

## Не сделано / после merge

- Re-score визита `3600047` на GCE + сверка findings (0 diclofenac).
- Smoke смены потолка Важно 60→55 (опционально).
- Фаза E (клинические сигналы) - бэклог.

## Тесты

- `test_drug_normalizer`, `test_worst_severity_filter`, `test_mo_ui_phase2`, `test_mo_dashboard_hero_cleanup` - OK.

## Deploy

- Production path: GCE `protocol.kravira.by` via `SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh`.
- PR #129 (docs-only plan) - закрыть как superseded этим PR.

## Следующая команда

```bash
# после merge в main, из чистого worktree или этого:
SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
# затем deep-rescore дня 2026-08-05 на GCE и проверка visit 3600047
```

## Не трогать параллельно

- `clinical_knowledge/drug_normalizer.py`, `kz_deep_eval.py`, `mo_scoring_profile.py`, `frontend/web/shared/mo-app.js`, `mo-charts.js`
