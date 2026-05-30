"""Загрузчик протоколов: вспомогательные функции манифеста (без сети)."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "download_minzdrav_protocols", ROOT / "download_minzdrav_protocols.py"
)
_dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dl)


def test_sha256_file_matches(tmp_path) -> None:
    p = tmp_path / "a.bin"
    data = b"hello protocol"
    p.write_bytes(data)
    assert _dl._sha256_file(p) == hashlib.sha256(data).hexdigest()


def test_now_utc_iso_format() -> None:
    ts = _dl._now_utc_iso()
    assert ts.endswith("Z")
    assert "T" in ts and len(ts) == 20


def test_safe_filename_from_url() -> None:
    name = _dl.safe_filename_from_url("https://minzdrav.gov.by/upload/КП №123.pdf")
    assert name.endswith(".pdf")
    assert "/" not in name
