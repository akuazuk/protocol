#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/backfill_mo_warehouse.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "backfill_mo_warehouse.py"), run_name="__main__")
