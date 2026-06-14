"""Tests for ICD → protocol fast lookup index."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_catalog_has_all_index_pdfs():
    pc = _load_module("protocol_catalog", "clinical_knowledge/protocol_catalog.py")
    cat = pc.load_protocol_catalog()
    assert len(cat) >= 470


def test_lookup_l60_adult_prefers_appendage_not_pediatric():
    pii = _load_module("protocol_icd_index", "clinical_knowledge/protocol_icd_index.py")
    pii._inverted_index.cache_clear()
    result = pii.lookup_protocols_by_icd(
        icd_codes=["L60.0"],
        query="вросший ноготь\nКонтекст подбора: взрослое население",
        population="adult",
        limit=5,
    )
    protos = result.get("protocols") or []
    assert protos, "expected at least one protocol for L60.0"
    top_path = (protos[0].get("path") or "").lower()
    assert "дет" not in top_path or "взросл" in top_path or "придат" in top_path


def test_lookup_returns_quickly():
    pii = _load_module("protocol_icd_index", "clinical_knowledge/protocol_icd_index.py")
    result = pii.lookup_protocols_by_icd(
        icd_codes=["J06.9"],
        query="ОРВИ кашель",
        population="adult",
        limit=6,
    )
    assert result.get("lookup_ms", 9999) < 500
    assert isinstance(result.get("protocols"), list)


def test_format_assist_payload_shape():
    pii = _load_module("protocol_icd_index", "clinical_knowledge/protocol_icd_index.py")
    lookup = pii.lookup_protocols_by_icd(icd_codes=["J06.9"], query="орви", limit=3)
    payload = pii.format_assist_payload(query="орви", lookup_result=lookup, icd_analysis={})
    assert payload.get("icd_fast_lookup") is True
    assert isinstance((payload.get("llm_json") or {}).get("protocols"), list)


def test_lookup_rubric_fallback_when_icd_missing_from_catalog():
    """C50.9 may be absent in catalog ICD lists — rubric novoobrazovaniya must still match."""
    pii = _load_module("protocol_icd_index", "clinical_knowledge/protocol_icd_index.py")
    pii._inverted_index.cache_clear()
    result = pii.lookup_protocols_by_icd(
        icd_codes=["C50.9"],
        query="рак молочной железы",
        population="adult",
        limit=5,
    )
    protos = result.get("protocols") or []
    assert protos, "expected oncology protocols via rubric fallback for C50.9"
    paths = " ".join(p.get("path", "") for p in protos).lower()
    assert "novoobrazovaniya" in paths


def test_lookup_sore_throat_r07_excludes_hiv_and_pediatric():
    """R07.0 + adult: не подмешивать J02 из жалоб; ВИЧ и детские КП не должны быть top."""
    pii = _load_module("protocol_icd_index", "clinical_knowledge/protocol_icd_index.py")
    pii._inverted_index.cache_clear()
    q = (
        "болит горло и температура 38\n"
        "Контекст подбора: взрослое население\n"
        "МКБ-10: R07.0"
    )
    result = pii.lookup_protocols_by_icd(
        icd_codes=["R07.0"],
        query=q,
        population="adult",
        limit=6,
    )
    protos = result.get("protocols") or []
    assert protos, "expected ENT/respiratory protocols for sore throat"
    titles = " ".join(p.get("title", "") for p in protos).lower()
    assert "вич" not in titles
    assert all(p.get("audience") != "pediatric" for p in protos)
    top = (protos[0].get("title") or "").lower()
    assert any(k in top for k in ("оториноларинг", "отоларинг", "орви", "респиратор", "фаринг", "ангин"))
