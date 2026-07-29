#!/usr/bin/env python3
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "check_gemini_key.py"), run_name="__main__")
