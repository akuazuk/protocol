#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/run_mis_protocol_l1_batch.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "run_mis_protocol_l1_batch.py"), run_name="__main__")
