#!/usr/bin/env python3
"""Применить кабинетный профиль жёсткости после подгрузки / night LLM.

Если в mo_scoring_profile.json стоит apply_on_next_load или pending_recompute -
запускает warehouse recompute (зоны/полосы) и снимает флаги.
Иначе только штампует last_applied_version, если штатный recompute уже прошёл.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_scoring_profile import (  # noqa: E402
    consume_next_load_recompute,
    get_recompute_job,
    load_scoring_profile,
    mark_profile_applied,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="MO_DATA_ROOT; по умолчанию из env /var/data/medical_exams",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="дождаться завершения фонового job (для shell-пайплайна)",
    )
    parser.add_argument("--timeout-sec", type=int, default=3600)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser() if args.data_root else None
    profile_before = load_scoring_profile(root=root)
    job = consume_next_load_recompute(root=root, actor="pipeline_next_load")
    if job is None:
        # штатный recompute уже был; зафиксируем версию профиля
        if profile_before.get("last_applied_version") != profile_before.get("profile_version"):
            mark_profile_applied(root=root)
        print(
            json.dumps(
                {
                    "ok": True,
                    "action": "stamp_only",
                    "profile_version": profile_before.get("profile_version"),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.wait:
        deadline = time.time() + max(30, int(args.timeout_sec))
        while time.time() < deadline:
            current = get_recompute_job(root=root) or {}
            if current.get("status") in {"done", "error"}:
                print(json.dumps({"ok": True, "action": "recompute", "job": current}, ensure_ascii=False))
                return 0 if current.get("status") == "done" else 1
            time.sleep(2)
        print(json.dumps({"ok": False, "action": "timeout", "job": get_recompute_job(root=root)}, ensure_ascii=False))
        return 1

    print(json.dumps({"ok": True, "action": "recompute_started", "job": job}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
