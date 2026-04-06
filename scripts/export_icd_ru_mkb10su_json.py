#!/usr/bin/env python3
"""Совместимость: делегирует scripts/export_icd_ru_from_xlsx.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_icd_export", _here / "export_icd_ru_from_xlsx.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
raise SystemExit(_mod.main())
