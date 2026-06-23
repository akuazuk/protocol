"""Тесты кэша и индекса protocol-summary-nav."""
from __future__ import annotations

from clinical_knowledge.protocol_nav_cache import (
    clear_protocol_nav_cache,
    resolve_protocol_nav_cached,
)
from clinical_knowledge.protocol_summary.nav import (
    _catalog_path_index,
    rebuild_catalog_path_index,
)


def test_catalog_path_index_is_cached():
    rebuild_catalog_path_index()
    a = _catalog_path_index()
    b = _catalog_path_index()
    assert a is b


def test_nav_cache_hit(monkeypatch):
    clear_protocol_nav_cache()
    calls = {"n": 0}

    def fake_resolve(path, *, query, icd_codes, allow_rich_fallback=True):
        calls["n"] += 1
        return {
            "available": True,
            "path": path,
            "conditions": [],
            "source": "summary",
        }

    monkeypatch.setattr(
        "clinical_knowledge.search_funnel.resolve_protocol_nav",
        fake_resolve,
    )
    r1 = resolve_protocol_nav_cached("minzdrav_protocols/x/a.pdf", query="test", icd_codes=["J20"])
    r2 = resolve_protocol_nav_cached("minzdrav_protocols/x/a.pdf", query="test", icd_codes=["J20"])
    assert calls["n"] == 1
    assert r1.get("cache_hit") is False
    assert r2.get("cache_hit") is True
    assert r2.get("nav_ms") is not None
