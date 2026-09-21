"""R5: Rceth label snippets and KP scheme on medication cards, no primary score."""
from __future__ import annotations

from pathlib import Path

import yaml

from clinical_knowledge.medication_normative_cards import (
    build_medication_normative_cards,
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


def _metformin_label() -> dict:
    return {
        "inn": "metformin",
        "reg_id": "test_metformin",
        "status": "active",
        "nd_changes": ["2024-03-01"],
        "sections": {
            "indications_4_1": ["Сахарный диабет 2 типа у взрослых."],
            "contraindications_4_3": ["Тяжёлая почечная недостаточность."],
        },
    }


def test_draft_when_no_label_and_no_kp_scheme() -> None:
    payload = build_medication_normative_cards(
        {"treatment_recommendations": "Метформин 500 мг"},
        label_ctx={"by_inn": {}},
        protocol_suggest={"ok": True, "available": False, "items": []},
    )
    assert payload["primary"] is False
    assert payload["cards"]
    card = payload["cards"][0]
    assert card["draft"] is True
    assert card["rceth"]["available"] is False
    assert card["kp_scheme"]["status"] == "unmatched"


def test_label_removes_draft_and_exposes_41_43() -> None:
    payload = build_medication_normative_cards(
        {"treatment_recommendations": "Метформин 500 мг"},
        label_ctx={"by_inn": {"metformin": [_metformin_label()]}},
        protocol_suggest={"ok": True, "available": False, "items": []},
    )
    card = payload["cards"][0]
    assert card["draft"] is False
    assert card["rceth"]["available"] is True
    assert card["rceth"]["revision"] == "2024-03-01"
    assert "диабет" in card["rceth"]["indications_4_1"].lower()
    assert "почечн" in card["rceth"]["contraindications_4_3"].lower()
    assert card["primary"] is False


def test_kp_scheme_match_without_label() -> None:
    summary = _summary("test_phleb_i801.yaml")
    payload = build_medication_normative_cards(
        {"treatment_recommendations": "ривароксабан 20 мг 1 раз в сутки"},
        label_ctx={"by_inn": {}},
        protocol_suggest=_matched_suggest(
            path=summary.source.local_path or "minzdrav_protocols/test/phleb.pdf",
            title=summary.source.title,
            protocol_id=summary.protocol_id,
            icd="I80.1",
        ),
        summary=summary,
    )
    card = payload["cards"][0]
    assert payload["kp_scheme"]["has_scheme"] is True
    assert card["draft"] is False
    assert card["kp_scheme"]["status"] == "in_scheme"
    assert "ривароксабан" in (card["kp_scheme"]["match_name"] or "").lower()


def test_kp_scheme_absent_drug_is_not_plan_zone() -> None:
    summary = _summary("test_phleb_i801.yaml")
    payload = build_medication_normative_cards(
        {"treatment_recommendations": "Метформин 500 мг"},
        label_ctx={"by_inn": {}},
        protocol_suggest=_matched_suggest(
            path=summary.source.local_path or "x.pdf",
            title=summary.source.title,
            protocol_id=summary.protocol_id,
            icd="I80.1",
        ),
        summary=summary,
    )
    card = payload["cards"][0]
    assert card["draft"] is False
    assert card["kp_scheme"]["status"] == "not_in_scheme"
    assert payload["primary"] is False
    assert payload["methodology"]["safety_role"] == "risk_not_plan_zone"


def test_matched_protocol_without_drugs_stays_honest() -> None:
    summary = _summary("test_gastro_k30.yaml")
    payload = build_medication_normative_cards(
        {"treatment_recommendations": "Метформин 500 мг"},
        label_ctx={"by_inn": {}},
        protocol_suggest=_matched_suggest(
            path=summary.source.local_path or "x.pdf",
            title=summary.source.title,
            protocol_id=summary.protocol_id,
            icd="K30",
        ),
        summary=summary,
    )
    card = payload["cards"][0]
    assert payload["kp_scheme"]["status"] == "no_drugs_in_kp"
    assert card["kp_scheme"]["status"] == "no_drugs_in_kp"
    assert card["draft"] is True
