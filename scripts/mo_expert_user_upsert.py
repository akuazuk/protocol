#!/usr/bin/env python3
"""Создать / обновить логин врача-эксперта в SQLite warehouse.

Пример:
  python3 scripts/mo_expert_user_upsert.py --login expert --password '...' --name 'Эксперт'
  MO_ANALYTICS_DB=/var/data/medical_exams/warehouse/mo_analytics.sqlite \\
    python3 scripts/mo_expert_user_upsert.py --login expert --password '...'
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_expert_auth import upsert_expert_user  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Upsert MO expert user")
    ap.add_argument("--login", required=True)
    ap.add_argument("--password", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--inactive", action="store_true")
    args = ap.parse_args()
    password = args.password.strip()
    generated = False
    if not password:
        password = secrets.token_urlsafe(12)
        generated = True
    result = upsert_expert_user(
        login=args.login,
        password=password,
        display_name=args.name,
        active=not args.inactive,
    )
    print(f"ok login={result['login']} expert_id={result['expert_id']} active={result['active']}")
    if generated:
        print(f"generated_password={password}")
        print("Сохраните пароль вне git (Render env / password manager).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
