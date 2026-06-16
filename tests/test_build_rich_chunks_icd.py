"""Тесты обогащения icd10_codes в build_rich_chunks."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_rich_chunks",
    ROOT / "scripts" / "build_rich_chunks.py",
)
assert _SPEC and _SPEC.loader
_brc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_brc)


def test_merge_icd10_preserves_priority() -> None:
    out = _brc.merge_icd10_code_lists(["K64.9", "K62.5"], ["K92.2", "K64.9"], max_codes=4)
    assert out == ["K64.9", "K62.5", "K92.2"]


def test_build_chunk_icd_includes_catalog_and_text_for_criteria() -> None:
    icd = _brc.build_chunk_icd10_codes(
        chunk_text="Показания при геморрое K64.1 и кровотечении K62.5",
        chunk_type="criteria_block",
        protocol_primary=["K64.9"],
        protocol_all=["K64.0", "K64.9"],
        catalog_primary=["K64.9", "K62.5"],
        catalog_all=["K64.9", "K62.5", "K92.2"],
    )
    assert "K64.1" in icd
    assert "K62.5" in icd
    assert "K64.9" in icd
    assert icd[0] in ("K64.1", "K62.5", "K64.9")


def test_embedding_ready_text_duplicates_text_icd_for_diagnostics() -> None:
    emb = _brc.build_embedding_ready_text(
        section_title="Диагностика",
        chunk_text="Обследование при K92.2",
        icd_codes=["K92.2", "K64.9"],
        populations=[],
        chunk_type="diagnostics",
    )
    assert emb.count("K92.2") >= 2
    assert emb.startswith("МКБ-10:")


def test_catalog_lookup_loads_primary_codes() -> None:
    rows = _brc._protocol_catalog_by_path()
    assert len(rows) >= 100
    sample = next(v for v in rows.values() if v.get("icd10_primary"))
    assert sample["icd10_primary"]
