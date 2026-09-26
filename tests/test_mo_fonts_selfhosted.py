"""Волна E редизайна МО: webfont с кириллицей самохостингом, без CDN."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
TOKENS = (ROOT / "frontend/web/shared/mo-tokens.css").read_text(encoding="utf-8")
FONT_DIR = ROOT / "frontend/web/shared/vendor/fonts"


def test_font_files_and_licenses_present() -> None:
    for family in ("onest", "golos-text", "jetbrains-mono"):
        for subset in ("cyrillic", "latin"):
            path = FONT_DIR / f"{family}-var-{subset}.woff2"
            assert path.is_file(), path
            assert path.read_bytes()[:4] == b"wOF2"
            assert path.stat().st_size < 120_000  # subset, не полный шрифт
    for lic in ("LICENSE-Onest.txt", "LICENSE-GolosText.txt", "LICENSE-JetBrainsMono.txt"):
        assert "SIL OPEN FONT LICENSE" in (FONT_DIR / lic).read_text(encoding="utf-8")


def test_tokens_declare_faces_and_families() -> None:
    faces = re.findall(r"@font-face \{(.*?)\}", TOKENS, re.S)
    assert len(faces) == 6
    for face in faces:
        assert 'url("/vendor/fonts/' in face
        assert "font-display: swap" in face
        assert "unicode-range" in face
        assert "googleapis" not in face and "gstatic" not in face
    assert '--font-display: "Onest"' in TOKENS
    assert '--font-ui: "Golos Text"' in TOKENS
    assert '--font-mono: "JetBrains Mono"' in TOKENS
    assert "font-variant-numeric: tabular-nums" in TOKENS
    # кириллический диапазон объявлен для каждой семьи
    assert TOKENS.count("U+0400-045F") == 3


def test_font_route_serves_woff2_and_rejects_traversal() -> None:
    import rag_server

    client = TestClient(rag_server.app)
    ok = client.get("/vendor/fonts/onest-var-cyrillic.woff2")
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("font/woff2")
    assert "immutable" in ok.headers.get("cache-control", "")
    assert ok.content[:4] == b"wOF2"
    assert client.get("/vendor/fonts/missing-font.woff2").status_code == 404
    assert client.get("/vendor/fonts/..%2F..%2Frag_server.py").status_code in (404, 400)
    assert client.get("/vendor/fonts/LICENSE-Onest.txt").status_code == 404
