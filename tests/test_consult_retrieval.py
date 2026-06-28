"""Tests for strict consult protocol path selection."""
from __future__ import annotations

from clinical_knowledge.consult_retrieval import (
    consult_target_protocol_paths,
    filter_retrieval_by_category_slugs,
    filter_retrieval_rows_by_paths,
)


def test_filter_retrieval_rows_by_paths():
    rows = [
        {"path": "minzdrav_protocols/pulmonologiya/a.pdf", "score": 1.0},
        {"path": "minzdrav_protocols/stomatologiya/b.pdf", "score": 0.9},
    ]
    out = filter_retrieval_rows_by_paths(
        rows, ["minzdrav_protocols/pulmonologiya/a.pdf"]
    )
    assert len(out) == 1
    assert "pulmonologiya" in out[0]["path"]


def test_consult_target_from_matched_rules():
    rules = {
        "matched_protocols": [
            {"source_path": "minzdrav_protocols/gastroenterologiya/gerd.pdf", "match_score": 80}
        ]
    }
    paths, meta = consult_target_protocol_paths(
        merged_icd=["K21.9"],
        diag_icd=["K21.9"],
        clinical_rules=rules,
        specialty_slugs=["gastroenterologiya"],
    )
    assert paths
    assert "gerd.pdf" in paths[0]
    assert meta.get("strict")


def test_filter_retrieval_by_category_slugs():
    rows = [
        {"path": "minzdrav_protocols/nevrologiya-neyrokhirurgiya/a.pdf", "category": "nevrologiya-neyrokhirurgiya"},
        {"path": "minzdrav_protocols/akusherstvo-ginekologiya/b.pdf", "category": "akusherstvo-ginekologiya"},
    ]
    out = filter_retrieval_by_category_slugs(
        rows,
        ["nevrologiya-neyrokhirurgiya"],
        strict=True,
    )
    assert len(out) == 1
    assert out[0]["category"] == "nevrologiya-neyrokhirurgiya"


def test_m54_always_has_protocol_pick():
    facts = {
        "consultation": {
            "complaints": ["боль в пояснице с иррадиацией в ногу"],
            "diagnosis_text": "M54.3 ишиас",
            "conditions_hint": ["боль в пояснице"],
            "performed_exams": [],
        },
        "patient_context": {"adult_or_child": "adult"},
    }
    paths, meta = consult_target_protocol_paths(
        merged_icd=["M54.3"],
        diag_icd=["M54.3"],
        clinical_rules=None,
        specialty_slugs=["nevrologiya"],
        consult_facts=facts,
        primary_specialty="nevrologiya",
        min_match_score=22.0,
    )
    assert paths, meta
    top = (meta.get("protocol_matches") or [{}])[0]
    assert float(top.get("match_score") or 0) >= 12.0
    assert "nevrologiya-neyrokhirurgiya" in str(paths[0]) or meta.get("icd_coverage_fallback")


def test_hiv_protocol_rejected_without_consult_markers():
    hiv_path = (
        "minzdrav_protocols/infektsionnye-zabolevaniya/"
        "КП_Оказание_медпомощи_пациентам_с_ВИЧ-инфекцией_пост_МЗ_25.07.2022_73.pdf"
    )
    rules = {
        "matched_protocols": [
            {"source_path": hiv_path, "match_score": 80, "title": "ВИЧ-инфекция"},
        ]
    }
    paths, meta = consult_target_protocol_paths(
        merged_icd=["G43.9"],
        diag_icd=["G43.9"],
        clinical_rules=rules,
        specialty_slugs=["infektsionnye-zabolevaniya"],
        consult_text="Жалобы: головная боль. Диагноз G43.9 мигрень.",
        consult_facts={"patient_context": {"adult_or_child": "adult"}},
    )
    assert hiv_path not in paths
    rejected = meta.get("rejected_protocols") or []
    assert any("wrong_nosology_hiv_without_markers" in (r.get("pick_risk_flags") or []) for r in rejected)


def test_adult_consult_rejects_pediatric_protocol():
    ped_path = (
        "minzdrav_protocols/nevrologiya-neyrokhirurgiya/"
        "КП_Диагностика_лечение_пациентов_заболеваниями_нервной_системы_детс_нас_пост_МЗ_12.04.2023_53.pdf"
    )
    adult_path = (
        "minzdrav_protocols/nevrologiya-neyrokhirurgiya/"
        "КП_Диагностика_лечение_пациентов_с_заболеваниями_нервной_системы_взр_нас_пост_МЗ_2018_8.pdf"
    )
    rules = {
        "matched_protocols": [
            {"source_path": ped_path, "match_score": 85, "title": "Нервная система дети"},
            {"source_path": adult_path, "match_score": 70, "title": "Нервная система взрослые"},
        ]
    }
    paths, meta = consult_target_protocol_paths(
        merged_icd=["G43.9"],
        diag_icd=["G43.9"],
        clinical_rules=rules,
        specialty_slugs=["nevrologiya-neyrokhirurgiya"],
        consult_text="Пациент взрослый. Жалобы на головную боль.",
        consult_facts={"patient_context": {"adult_or_child": "adult"}},
    )
    assert ped_path not in paths
    assert adult_path in paths
