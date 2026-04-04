"""
Загрузка .env и .env.local в os.environ.
Сначала python-dotenv (если есть), иначе простой разбор строк — чтобы работало с любым python3.
"""
from __future__ import annotations

import os
from pathlib import Path


def _parse_and_apply(path: Path, override: bool) -> None:
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = val


def load_project_env(root: Path) -> None:
    """root — каталог с .env (рядом с rag_server.py)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
        load_dotenv(root / ".env.local", override=True)
        return
    except ImportError:
        pass
    _parse_and_apply(root / ".env", override=False)
    _parse_and_apply(root / ".env.local", override=True)
