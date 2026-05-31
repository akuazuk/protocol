#!/usr/bin/env python3
"""CLI check-kz / check-kz-folder (ТЗ §5).

    python -m scripts.check_kz --file path/to/kz.pdf
    python -m scripts.check_kz --folder data/examples/consultations
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_consultation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
