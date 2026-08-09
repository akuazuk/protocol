# Handoff: empty dashboards + muted visual refresh (D7)

Дата: 2026-08-09

## Repo / deploy

| | |
|--|--|
| repo | `akuazuk/protocol` |
| PR | [#94](https://github.com/akuazuk/protocol/pull/94) merged |
| merge SHA | `fe3d72fdc5f7c6a7bb63c9626634998457787c49` |
| production | GCE `protocol.kravira.by` via `deploy/gcp-app/deploy_to_gce.sh` |
| `BUILD_VERSION` | `2026-08-09-081413Z-mo-dashboard-visual` |
| plan | `docs/plans/2026-08-09-mo-dashboards-zones-first-v2.md` (D7 done) |

## Сделано

1. Merged #93 (Справка) earlier; then #94 visual/data fixes.
2. Warehouse recompute 2026-08-01..06 already on GCE (data was never empty).
3. API: overview attention resolves `month=` → date range.
4. UI: Сегодня falls back to `data_through` with banner; zone trend = muted ECharts.
5. Palette: soft teal/blue/amber tokens; no neon purple/crimson.

## Smoke (GCE localhost + token)

- freshness `data_through=2026-08-06`
- overview `month=2026-08`: `n_evaluated=2242`, `zone1_bad=474`, `zone_trends=6`
- daily `2026-08-06`: `n_evaluated=441`

## Не сделано / нельзя без inbound

- Дней **2026-08-07..09** нет в extract (`gs://…/inbound/extract` / `/var/data/.../inbound`). Без новой выгрузки из МИС нечего пересчитывать.

## Параллельно

- Draft #77 (docs MZ layers) - не трогать без нужды.
- Грязные dash-normalize файлы в старых worktree не коммитить.

## Одна следующая команда

```bash
# когда появится inbound за 07+ :
gcloud compute ssh protocol-app --zone=europe-central2-a --command \
  'docker exec protocol-web python3 scripts/recompute_mo_days.py --from 2026-08-07 --to 2026-08-09'
```

## Не трогать параллельно

`frontend/web/shared/mo-{app,charts,tokens,ui}.*`, `clinical_knowledge/mo_backend.py` (attention helper).
