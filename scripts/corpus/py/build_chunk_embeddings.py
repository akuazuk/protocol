#!/usr/bin/env python3
"""Canonical entrypoint shim. Legacy path: scripts/build_chunk_embeddings.py."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "scripts" / "build_chunk_embeddings.py"), run_name="__main__")
