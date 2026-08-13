"""КП sync: HTML crawl, diff, jsonl merge, metadata, recency."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from clinical_knowledge.kp_sync.diff import diff_catalog, parse_post_from_name
from clinical_knowledge.kp_sync.jsonl_merge import merge_jsonl_by_path
from clinical_knowledge.kp_sync.metadata import extract_protocol_metadata
from clinical_knowledge.kp_sync.parse import category_pages, site_docs_from_pages
from clinical_knowledge.kp_sync.recency import recency_multiplier
from clinical_knowledge.protocol_match import compute_match_score, match_protocol_cards
from clinical_knowledge.protocol_pick_filters import is_administrative_protocol

FIX = Path(__file__).resolve().parent / "fixtures" / "kp_sync"


def test_parse_categories_and_docs():
    index = (FIX / "index.html").read_text(encoding="utf-8")
    cats = category_pages(index)
    slugs = {s for s, _ in cats}
    assert slugs == {"gastroenterologiya", "bolezni-sistemy-krovoobrashcheniya"}
    pages = []
    for slug, url in cats:
        html = (FIX / f"{slug}.html").read_text(encoding="utf-8")
        pages.append((slug, url, html))
    docs = site_docs_from_pages(pages)
    names = {d["filename"] for d in docs}
    assert any("№36" in n or "№36" in n.replace("№", "№") for n in names)
    assert any("ОКС" in n for n in names)
    assert len(docs) == 3


def test_diff_added_updated_superseded_rename():
    site = [
        {
            "slug": "cardio",
            "filename": "пост. МЗ РБ от 28.04.2026 №44_КП_ОКС.pdf",
            "relative_path": "minzdrav_protocols/cardio/пост. МЗ РБ от 28.04.2026 №44_КП_ОКС.pdf",
            "url": "https://minzdrav.gov.by/a.pdf",
            "sha256": "aaa",
        },
        {
            "slug": "gastro",
            "filename": "пост. МЗ РБ от 25.04.2026 №36_КП_ЗНО пищевода.pdf",
            "relative_path": "minzdrav_protocols/gastro/пост. МЗ РБ от 25.04.2026 №36_КП_ЗНО пищевода.pdf",
            "url": "https://minzdrav.gov.by/b.pdf",
            "sha256": "bbb2",
        },
    ]
    local = [
        {
            "filename": "КП диагностики и лечения инфаркта миокарда 06.06.2017 № 59.pdf",
            "relative_path": "minzdrav_protocols/cardio/КП диагностики и лечения инфаркта миокарда 06.06.2017 № 59.pdf",
            "sha256": "old",
        },
        {
            "filename": "пост. МЗ РБ от 25.04.2026 №36_КП_ЗНО пищевода.pdf",
            "relative_path": "minzdrav_protocols/gastro/пост. МЗ РБ от 25.04.2026 №36_КП_ЗНО пищевода.pdf",
            "sha256": "bbb1",
        },
    ]
    d = diff_catalog(site, local)
    assert len(d["added"]) == 1
    assert "ОКС" in d["added"][0]["filename"]
    assert len(d["updated"]) == 1
    assert d["updated"][0]["sha256"] == "bbb2"
    assert len(d["superseded"]) == 1
    assert "2017" in d["superseded"][0]["filename"]
    assert d["changed_paths"]


def test_diff_rename_same_post():
    site = [
        {
            "slug": "neo",
            "filename": "пост._МЗ_РБ от 18.04.2022 №34 (в ред. пост. МЗ РБ от 18.05.2026 №54)_КП_неонатология.pdf",
            "relative_path": "minzdrav_protocols/neo/new.pdf",
            "url": "https://minzdrav.gov.by/n.pdf",
            "sha256": "n2",
        }
    ]
    local = [
        {
            "filename": "КП_Оказание_медицинской_помощи_в_неонатологии_постановление_МЗ_18042022_34.pdf",
            "relative_path": "minzdrav_protocols/neo/old.pdf",
            "sha256": "n1",
            "display_title": "неонатология от 18.04.2022 №34",
        }
    ]
    d = diff_catalog(site, local)
    # same post 18.04.2022 №34 → updated/renamed, not added+superseded pair
    assert d["added"] == []
    assert len(d["updated"]) == 1
    assert d["superseded"] == []


def test_merge_keeps_untouched_paths():
    old = [
        {"source_path": "minzdrav_protocols/a/old.pdf", "text": "keep"},
        {"source_path": "minzdrav_protocols/b/chg.pdf", "text": "stale"},
    ]
    new = [{"source_path": "minzdrav_protocols/b/chg.pdf", "text": "fresh"}]
    merged = merge_jsonl_by_path(old, new)
    by = {r["source_path"]: r["text"] for r in merged}
    assert by["minzdrav_protocols/a/old.pdf"] == "keep"
    assert by["minzdrav_protocols/b/chg.pdf"] == "fresh"
    assert len(merged) == 2


def test_metadata_title_kind_audience():
    text = (
        "МИНИСТЕРСТВО ЗДРАВООХРАНЕНИЯ\n"
        "Клинический протокол «Диагностика и лечение острых форм "
        "ишемической болезни сердца (взрослое население)»\n"
        "утвержден постановлением от 28.04.2026 № 44\n"
    )
    meta = extract_protocol_metadata(
        text=text,
        filename="пост. МЗ РБ от 28.04.2026 №44_КП_ОКС.pdf",
    )
    assert "ишемической" in meta["title"].lower()
    assert meta["protocol_kind"] == "clinical"
    assert meta["audience"] == "adult"
    assert meta["approval_number"] == "44"
    assert meta["clinical_for_score"] is True

    rehab = extract_protocol_metadata(
        text="Клинический протокол «Медицинская реабилитация пациентов с инсультом (взрослое население)»",
        filename="КП_МР_инсульт.pdf",
    )
    assert rehab["protocol_kind"] == "rehab"
    assert rehab["clinical_for_score"] is False


def test_recency_prefers_2026_over_2017_same_icd():
    today = date(2026, 8, 13)
    new = {"approval": {"date": "2026-04-28"}, "status": "active"}
    old = {"approval": {"date": "2017-06-06"}, "status": "active"}
    assert recency_multiplier(new, today=today) > recency_multiplier(old, today=today)

    card_new = {
        "title": "ОКС 2026",
        "source_path": "minzdrav_protocols/cardio/ocs2026.pdf",
        "icd10_all": ["I21.0", "I20"],
        "icd10_primary": ["I21.0"],
        "specialty_slug": "bolezni-sistemy-krovoobrashcheniya",
        "population": "adult",
        "approval": {"date": "2026-04-28", "number": "44"},
        "status": "active",
        "protocol_kind": "clinical",
    }
    card_old = {
        "title": "Инфаркт 2017",
        "source_path": "minzdrav_protocols/cardio/mi2017.pdf",
        "icd10_all": ["I21.0", "I20"],
        "icd10_primary": ["I21.0"],
        "specialty_slug": "bolezni-sistemy-krovoobrashcheniya",
        "population": "adult",
        "approval": {"date": "2017-06-06", "number": "59"},
        "status": "active",
        "protocol_kind": "clinical",
    }
    s_new = compute_match_score(
        card_new,
        icd_list=["I21.0"],
        audience="adult",
        hints=set(),
        specialty_slug="bolezni-sistemy-krovoobrashcheniya",
        diag_text="острый коронарный синдром",
        complaints=[],
        performed_exams=[],
    )
    s_old = compute_match_score(
        card_old,
        icd_list=["I21.0"],
        audience="adult",
        hints=set(),
        specialty_slug="bolezni-sistemy-krovoobrashcheniya",
        diag_text="острый коронарный синдром",
        complaints=[],
        performed_exams=[],
    )
    assert s_new > s_old


def test_superseded_and_rehab_not_primary():
    assert is_administrative_protocol({"title": "МР ЧМТ", "protocol_kind": "rehab"})
    cards = [
        {
            "protocol_id": "old",
            "title": "Инфаркт 2017",
            "source_path": "minzdrav_protocols/cardio/old.pdf",
            "icd10_all": ["I21"],
            "icd10_primary": ["I21"],
            "specialty_slug": "bolezni-sistemy-krovoobrashcheniya",
            "population": "adult",
            "status": "superseded",
            "superseded_by": "minzdrav_protocols/cardio/new.pdf",
            "protocol_kind": "clinical",
        }
    ]
    # match_protocol_cards loads registry; inject via monkeypatch in caller if needed.
    # Filter is inside the function after scoring - use compute path by calling
    # with a patched loader.
    from unittest.mock import patch

    with patch(
        "clinical_knowledge.protocol_match.load_protocol_cards_registry",
        return_value=cards,
    ):
        out = match_protocol_cards(
            {
                "consultation": {
                    "icd10": ["I21"],
                    "diagnosis_text": "инфаркт",
                },
                "patient_context": {"adult_or_child": "adult"},
            },
            specialty_slug="bolezni-sistemy-krovoobrashcheniya",
            limit=3,
        )
    assert out == []


def test_parse_post():
    assert parse_post_from_name("пост. МЗ РБ от 28.04.2026 №44_КП_ОКС.pdf") == (
        "28.04.2026",
        "44",
    )


def test_canonical_alias_same_filename():
    from clinical_knowledge.kp_sync.parse import mark_canonical_aliases

    docs = mark_canonical_aliases(
        [
            {"filename": "same.pdf", "relative_path": "minzdrav_protocols/a/same.pdf", "slug": "a"},
            {"filename": "same.pdf", "relative_path": "minzdrav_protocols/b/same.pdf", "slug": "b"},
        ]
    )
    assert docs[0]["canonical"] == "1"
    assert docs[1]["alias_of"] == docs[0]["relative_path"]


def test_merge_indexes_keeps_other_paths(tmp_path: Path):
    from clinical_knowledge.kp_sync.indexes import merge_catalog_for_paths, merge_icd_profiles_for_paths
    from clinical_knowledge.kp_sync.jsonl_merge import write_jsonl

    cat = tmp_path / "catalog.jsonl"
    write_jsonl(
        cat,
        [{"path": "minzdrav_protocols/x/old.pdf", "title": "keep"}],
    )
    chunks = {
        "minzdrav_protocols/y/new.pdf": [
            {"source_path": "minzdrav_protocols/y/new.pdf", "icd10_codes": ["I21.0"], "text": "диагностика ЭКГ"}
        ]
    }
    stats = merge_catalog_for_paths(
        ["minzdrav_protocols/y/new.pdf"],
        chunks_by_path=chunks,
        out_path=cat,
    )
    assert stats["after"] == 2
    icd_path = tmp_path / "icd.jsonl"
    write_jsonl(icd_path, [{"path": "minzdrav_protocols/x/old.pdf", "diagnostics": [{"text": "keep"}]}])
    icd_stats = merge_icd_profiles_for_paths(
        ["minzdrav_protocols/y/new.pdf"],
        chunks_by_path=chunks,
        out_path=icd_path,
    )
    assert icd_stats["after"] == 2
    empty_stats = merge_icd_profiles_for_paths(
        ["minzdrav_protocols/z/missing.pdf"],
        chunks_by_path={},
        out_path=icd_path,
    )
    assert empty_stats["incoming"] == 0
    assert empty_stats["after"] == 2


def test_public_kp_sync_payload(tmp_path: Path):
    from clinical_knowledge.kp_sync.status import load_latest_kp_sync, public_kp_sync_payload

    p = tmp_path / "kp_sync_2026-08-13.json"
    p.write_text(
        '{"status":"success","site_count":10,"local_count":9,"added":[{"filename":"a.pdf","slug":"cardio","relative_path":"minzdrav_protocols/cardio/a.pdf","action":"downloaded"}],"updated":[],"superseded":[],"changed_paths":["minzdrav_protocols/cardio/a.pdf"]}',
        encoding="utf-8",
    )
    raw = load_latest_kp_sync(tmp_path)
    pub = public_kp_sync_payload(raw)
    assert pub["site_count"] == 10
    assert pub["changed_n"] == 1
    assert pub["added"][0]["filename"] == "a.pdf"


def test_scan_local_from_pdf_root(tmp_path: Path):
    import runpy

    mod = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "kp_sync_run.py"))
    slug = tmp_path / "revmatologiya"
    slug.mkdir()
    (slug / "KP1.pdf").write_bytes(b"%PDF")
    rows = mod["local_docs_from_pdf_root"](tmp_path)
    assert rows[0]["filename"] == "KP1.pdf"
    assert rows[0]["relative_path"] == "minzdrav_protocols/revmatologiya/KP1.pdf"
