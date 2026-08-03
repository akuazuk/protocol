"""Tests for catalog-wide rules build."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.catalog_build import (
    catalog_source_paths,
    extract_rules_all_catalog_pdfs,
    merge_rules_into_catalog,
    write_coverage_report,
)
from clinical_knowledge.loader import clear_clinical_knowledge_cache, load_rules_by_condition
from clinical_knowledge.rule_checker import run_rule_checker


def test_catalog_source_paths_non_empty():
    paths = catalog_source_paths()
    if not Path("output/registry/protocol_cards.jsonl").is_file():
        return
    assert len(paths) >= 100
    rubrics = {p.split("/")[1] for p in paths if p.startswith("minzdrav_protocols/")}
    assert len(rubrics) >= 10


def test_extract_rules_sample_pdf():
    paths = catalog_source_paths()
    if not paths:
        return
    extracted, meta = extract_rules_all_catalog_pdfs(
        Path("output/chunks/chunks.jsonl"),
        chunks_index={},
    )
    assert meta["pdfs_total"] == len(paths)
    assert isinstance(meta.get("by_rubric"), dict)


def test_merge_rules_and_coverage_report(tmp_path: Path, monkeypatch):
    import clinical_knowledge.catalog_build as cb

    monkeypatch.setattr(cb, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(cb, "COVERAGE_PATH", tmp_path / "rules_coverage_report.json")

    extracted = {
        "acute_appendicitis": [
            {
                "rule_id": "test_path_append",
                "rule_type": "diagnosis_formula",
                "required_components": ["нозология", "форма"],
                "auto_extracted": True,
                "extraction_method": "path_template",
                "source": {"source_path": "minzdrav_protocols/khirurgiya/x.pdf"},
            }
        ]
    }
    meta = {
        "pdfs_total": 1,
        "pdfs": {"minzdrav_protocols/khirurgiya/x.pdf": {"rules": 1, "rubric": "khirurgiya"}},
        "by_rubric": {"khirurgiya": {"pdfs": 1, "with_rules": 1, "coverage_pct": 100.0}},
    }
    out_dir = tmp_path / "rules"
    counts = merge_rules_into_catalog(extracted, out_dir=out_dir)
    assert counts.get("acute_appendicitis") == 1
    assert (out_dir / "path_acute_appendicitis.json").is_file()
    report = write_coverage_report(meta, extracted)
    assert report["pdfs_with_rules"] == 1
    assert report["by_rubric"]["khirurgiya"]["coverage_pct"] == 100.0
    assert (tmp_path / "rules_coverage_report.json").is_file()


def test_loader_merges_catalog_rules(tmp_path: Path, monkeypatch):
    catalog_rules = tmp_path / "catalog" / "rules"
    catalog_rules.mkdir(parents=True)
    payload = {
        "condition_id": "pneumonia",
        "rules": [
            {
                "rule_id": "unit_pneumonia_diag",
                "rule_type": "diagnosis_formula",
                "required_components": ["нозология"],
                "severity": "warning",
            }
        ],
    }
    (catalog_rules / "path_pneumonia.json").write_text(json.dumps(payload), encoding="utf-8")

    import clinical_knowledge.loader as loader_mod

    monkeypatch.setattr(loader_mod, "CATALOG_DIR", tmp_path / "catalog")
    clear_clinical_knowledge_cache()
    rules = load_rules_by_condition()
    assert "pneumonia" in rules
    assert any(r.get("rule_id") == "unit_pneumonia_diag" for r in rules["pneumonia"])


def test_rule_checker_uses_registry_hints():
    facts = {
        "consultation": {
            "diagnosis_text": "Острая пневмония нижней доли",
            "icd10": ["J18.9"],
            "conditions_hint": [],
            "complaints": [],
            "text_sample": "",
        },
        "patient_context": {"adult_or_child": "adult"},
    }
    matched = [
        {
            "protocol_id": "proto_test",
            "source_path": "minzdrav_protocols/pulmonologiya-ftiziatriya/pneumonia_adult.pdf",
            "title": "Пневмония",
        }
    ]
    result = run_rule_checker(facts, matched_protocols=matched)
    assert "pneumonia" in result.get("checked_conditions", [])
    assert result.get("findings")
