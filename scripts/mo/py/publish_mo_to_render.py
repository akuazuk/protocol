#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/publish_mo_to_render.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "publish_mo_to_render.py"), run_name="__main__")
