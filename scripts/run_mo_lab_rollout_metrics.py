#!/usr/bin/env python3
"""Build PHI-safe MO lab rollout metrics or enforce the primary guard."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "mo_lab_rollout_standalone",
    ROOT / "clinical_knowledge" / "mo_lab_rollout.py",
)
assert _SPEC and _SPEC.loader
_ROLLOUT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ROLLOUT)
build_rollout_report = _ROLLOUT.build_rollout_report
ensure_shadow_state = _ROLLOUT.ensure_shadow_state
lab_primary_guard = _ROLLOUT.lab_primary_guard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("MO_DATA_ROOT") or "/var/data/medical_exams")
    parser.add_argument("--analytics-db", default="")
    parser.add_argument("--lab-db", default="")
    parser.add_argument("--git-sha", default=os.environ.get("GIT_COMMIT_SHA") or "")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--init-shadow-state-only", action="store_true")
    parser.add_argument("--check-primary", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    state = ensure_shadow_state(
        git_commit_sha=args.git_sha,
        data_root=data_root,
    )
    if args.init_shadow_state_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "engine": state.get("engine"),
                    "shadow_since": state.get("shadow_since"),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.check_primary:
        guard = lab_primary_guard(data_root=data_root)
        print(json.dumps(guard, ensure_ascii=False))
        return 0 if not guard["requested"] or guard["allowed"] else 3

    analytics_db = Path(args.analytics_db or data_root / "warehouse" / "mo_analytics.sqlite")
    lab_db = Path(args.lab_db or data_root / "warehouse" / "mo_lab.sqlite")
    report = build_rollout_report(
        analytics_db=analytics_db,
        lab_db=lab_db,
        data_root=data_root,
        end_date=(
            date.fromisoformat(args.end_date)
            if args.end_date
            else date.today() - timedelta(days=1)
        ),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "engine": report.get("engine"),
                "generated_date": report.get("generated_date"),
                "successful_lab_nights": (report.get("guard_inputs") or {}).get(
                    "successful_lab_nights"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
