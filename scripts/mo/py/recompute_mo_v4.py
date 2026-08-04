#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/recompute_mo_v4.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "recompute_mo_v4.py"), run_name="__main__")
