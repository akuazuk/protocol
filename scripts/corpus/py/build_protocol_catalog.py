#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/build_protocol_catalog.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "build_protocol_catalog.py"), run_name="__main__")
