## scripts/mis

MIS MariaDB exports and L1 batch over mis_protocol.

### Canonical entrypoints (preferred)

- `scripts/mis/py/export_mis_protocol_month.py` → `scripts/export_mis_protocol_month.py`
- `scripts/mis/py/merge_mis_protocol_export.py` → `scripts/merge_mis_protocol_export.py`
- `scripts/mis/py/run_mis_protocol_l1_batch.py` → `scripts/run_mis_protocol_l1_batch.py`
- `scripts/mis/py/validate_mis_export.py` → `scripts/validate_mis_export.py`

Legacy flat paths under `scripts/*.py` remain supported. Physical moves are deferred
until call-sites and launchd/CI are migrated (plan phase 2b+).

### Domain inventory (still flat in `scripts/`)

- `scripts/export_mis_protocol_month.py`
- `scripts/merge_mis_protocol_export.py`
- `scripts/run_mis_protocol_l1_batch.py`
- `scripts/validate_mis_export.py`
