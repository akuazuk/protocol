#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/enrich_rich_chunk_tags.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "enrich_rich_chunk_tags.py"), run_name="__main__")
