#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/merge_mis_protocol_export.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "merge_mis_protocol_export.py"), run_name="__main__")
