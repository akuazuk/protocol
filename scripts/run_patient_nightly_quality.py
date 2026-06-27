#!/usr/bin/env python3
"""Ночной отчёт качества B2C + email + черновики snippet для методиста.

Пример:
  python3 scripts/run_patient_nightly_quality.py
  python3 scripts/run_patient_nightly_quality.py --dry-run
  ML_FEEDBACK_DIR=/var/data/ml/feedback python3 scripts/run_patient_nightly_quality.py --no-email
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import env_load

    env_load.load_project_env(ROOT)
except ImportError:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Protocol B2C nightly quality report")
    ap.add_argument("--dry-run", action="store_true", help="Не писать файлы и не слать email")
    ap.add_argument("--no-email", action="store_true", help="Только файлы отчёта")
    ap.add_argument(
        "--feedback-dir",
        type=Path,
        default=None,
        help="Каталог JSONL (иначе ML_FEEDBACK_DIR или data/ml/feedback)",
    )
    args = ap.parse_args()

    from clinical_knowledge.patient_nightly_quality import run_patient_nightly_quality

    out = run_patient_nightly_quality(
        fb_root=args.feedback_dir,
        send_email=not args.no_email,
        dry_run=args.dry_run,
    )
    print(out.get("report_md") or "(dry-run)")
    email = out.get("email") or {}
    if email.get("ok"):
        print("Email sent to:", email.get("to"))
    elif email.get("skipped"):
        print("Email skipped:", email.get("reason", "smtp_not_configured"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
