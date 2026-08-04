## scripts/mo

Methodist / МО daily ETL, warehouse, publish-to-Render, validation.

### Canonical entrypoints (preferred)

- `scripts/mo/py/run_mo_daily_pipeline.py` → `scripts/run_mo_daily_pipeline.py`
- `scripts/mo/py/run_mo_daily_report.py` → `scripts/run_mo_daily_report.py`
- `scripts/mo/py/export_mo_daily.py` → `scripts/export_mo_daily.py`
- `scripts/mo/py/publish_mo_to_render.py` → `scripts/publish_mo_to_render.py`
- `scripts/mo/py/backfill_mo_warehouse.py` → `scripts/backfill_mo_warehouse.py`
- `scripts/mo/py/recompute_mo_v4.py` → `scripts/recompute_mo_v4.py`
- `scripts/mo/py/build_mo_v4_validation.py` → `scripts/build_mo_v4_validation.py`

Legacy flat paths under `scripts/*.py` remain supported. Physical moves are deferred
until call-sites and launchd/CI are migrated (plan phase 2b+).

### Domain inventory (still flat in `scripts/`)

- `scripts/backfill_mo_warehouse.py`
- `scripts/benchmark_mo_summary.py`
- `scripts/build_mo_v4_validation.py`
- `scripts/export_mo_daily.py`
- `scripts/manage_mo_daily_launchd.py`
- `scripts/migrate_mo_crm_to_warehouse.py`
- `scripts/publish_mo_to_render.py`
- `scripts/recompute_mo_days.py`
- `scripts/recompute_mo_v4.py`
- `scripts/run_methodist_ai_smoke.py`
- `scripts/run_methodist_batch.py`
- `scripts/run_methodist_search_probe.py`
- `scripts/run_mo_daily_pipeline.py`
- `scripts/run_mo_daily_report.py`
