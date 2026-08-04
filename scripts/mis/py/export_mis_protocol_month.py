#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/export_mis_protocol_month.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "export_mis_protocol_month.py"), run_name="__main__")
