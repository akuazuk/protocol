#!/usr/bin/env python3
"""
Проверка ключа из .env: один короткий запрос к Gemini.

  python3 check_gemini_key.py

Код выхода: 0 — OK, 1 — ошибка.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local", override=True)
except ImportError:
    pass

from gemini_verify import verify_gemini_key


def main() -> int:
    ok, msg = verify_gemini_key()
    if ok:
        print("OK — модель ответила:", msg)
        return 0
    print("FAIL:", msg)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
