#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/build_rich_chunks.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "build_rich_chunks.py"), run_name="__main__")
