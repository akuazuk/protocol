#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/validate_mis_export.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "validate_mis_export.py"), run_name="__main__")
