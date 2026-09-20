"""R2: plan vs Protocol Summary concordance without new score thresholds."""
from __future__ import annotations

from pathlib import Path

import yaml

from clinical_knowledge.mo_case_kp_concordance import (
    ENGINE,
    attach_kp_concordance,
    build_kp_concordance,
)
from clinical_knowledge.protocol_summary.schema import ProtocolSummary

FIX = Path(__file__).resolve().parent / "fixtures" / "protocol_summaries" / "yaml"


def _summary(name: str) -> ProtocolSummary:
    data = yaml.safe_load((FIX / name).read_text(encoding="utf-8"))
    return ProtocolSummary.model_validate(data)


def _matched_suggest(*, path: str, title: str, protocol_id: str, icd: str) -> dict:
    return {
        "ok": True,
        "available": True,
        "items": [
            {
                "protocol_id": protocol_id,
                "title": title,
                "source_path": path,
                "match_kind": "clinical",
                "score": 88,
                "trust": "B",
                "icd_fit": [{"code": icd, "weight": 0.9}],
            }
        ],
    }


def test_unmatched_is_empty_not_fail() -> None:
    payload = build_kp_concordance(
        protocol_suggest={"ok": True, "available": False, "items": []},
        clinical={"exam_recommendations": "ЭГДС"},
        summary=_summary("test_gastro_k30.yaml"),
    )
    assert payload["available"] is False
    assert payload["rows"] == []
    assert payload["kp_status"] == "unmatched"
    assert "не подобран" in (payload["reason"] or "")
    assert "не соответствует протоколу" not in (payload["reason"] or "")


def test_required_exam_present_in_plan() -> None:
    summary = _summary("test_gastro_k30.yaml")
    payload = build_kp_concordance(
        protocol_suggest=_matched_suggest(
            path=summary.source.local_path or "minzdrav_protocols/gastroenterologiya/test.pdf",
            title=summary.source.title,
            protocol_id=summary.protocol_id,
            icd="K30",
        ),
        clinical={"exam_recommendations": "Рекомендована ЭГДС в плановом порядке."},
        summary=summary,
    )
    assert payload["available"] is True
    exams = [row for row in payload["rows"] if row["kind"] == "exam"]
    assert exams
    assert exams[0]["requirement"] == "ЭГДС"
    assert exams[0]["status"] == "present"
    assert "ЭГДС" in exams[0]["mo_quote"]
    assert exams[0]["slot"] == "exam_recommendations"


def test_required_exam_missing_from_plan() -> None:
    summary = _summary("test_gastro_k30.yaml")
    payload = build_kp_concordance(
        protocol_suggest=_matched_suggest(
            path=summary.source.local_path or "x.pdf",
            title=summary.source.title,
            protocol_id=summary.protocol_id,
            icd="K30",
        ),
        clinical={"exam_recommendations": "ОАК, биохимия.", "treatment_recommendations": "диета"},
        summary=summary,
    )
    exams = [row for row in payload["rows"] if row["kind"] == "exam" and row["requirement"] == "ЭГДС"]
    assert exams and exams[0]["status"] == "missing"
    assert payload["counts"]["missing"] >= 1


def test_treatment_drug_present() -> None:
    summary = _summary("test_phleb_i801.yaml")
    payload = build_kp_concordance(
        protocol_suggest=_matched_suggest(
            path=summary.source.local_path or "minzdrav_protocols/test/phleb.pdf",
            title=summary.source.title,
            protocol_id=summary.protocol_id,
            icd="I80.1",
        ),
        clinical={"treatment_recommendations": "ривароксабан 20 мг 1 раз в сутки"},
        summary=summary,
    )
    drugs = [row for row in payload["rows"] if row["kind"] == "treatment"]
    assert drugs and drugs[0]["status"] == "present"
    assert drugs[0]["slot"] == "treatment_recommendations"


def test_night_off_protocol_reused_without_new_threshold() -> None:
    summary = _summary("test_gastro_k30.yaml")
    payload = build_kp_concordance(
        protocol_suggest=_matched_suggest(
            path=summary.source.local_path or "x.pdf",
            title=summary.source.title,
            protocol_id=summary.protocol_id,
            icd="K30",
        ),
        clinical={"exam_recommendations": "ЭГДС", "treatment_recommendations": "антибиотик вне схемы"},
        night_plan={
            "verdict": "partial",
            "provenance": "kp_grounded",
            "kp_status": "matched",
            "missing_required": [],
            "off_protocol": ["антибиотик вне схемы"],
            "source_refs": ["summary"],
            "summary_ru": "план частично по протоколу",
            "exam_pct": 80,
            "treatment_pct": 40,
        },
        summary=summary,
    )
    off = [row for row in payload["rows"] if row["status"] == "off_protocol"]
    assert off
    assert off[0]["source"] == "night"
    assert payload["engine"] == ENGINE


def test_low_trust_does_not_invent_fail() -> None:
    summary = _summary("test_gastro_k30.yaml")
    payload = build_kp_concordance(
        protocol_suggest={
            "ok": True,
            "available": True,
            "items": [
                {
                    "protocol_id": summary.protocol_id,
                    "title": summary.source.title,
                    "source_path": summary.source.local_path,
                    "match_kind": "clinical",
                    "score": 88,
                    "trust": "D",
                }
            ],
        },
        clinical={"exam_recommendations": ""},
        summary=summary,
    )
    assert payload["available"] is False
    assert payload["rows"] == []
    assert "A/B" in (payload["reason"] or "")


def test_attach_does_not_dump_clinical_text() -> None:
    summary = _summary("test_gastro_k30.yaml")
    suggest = _matched_suggest(
        path=summary.source.local_path or "x.pdf",
        title=summary.source.title,
        protocol_id=summary.protocol_id,
        icd="K30",
    )
    clinical = {
        "exam_recommendations": "ЭГДС",
        "complaints": "секретный текст жалоб пациента",
    }
    attach_kp_concordance(suggest, clinical=clinical, summary=summary)
    blob = str(suggest["kp_concordance"])
    assert "секретный текст жалоб пациента" not in blob
    assert "kp_concordance" in suggest
