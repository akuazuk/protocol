"""Shadow medication/normative cards for MO case review."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from clinical_knowledge.medication_parser import medication_assignments_from_case
from clinical_knowledge.rceth_sync.label_ctx import load_rceth_label_ctx, lookup_label

ENGINE = "mo_medication_normative_cards_v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _protocol_context(
    assessment: Mapping[str, Any],
    zones: Mapping[str, Any],
    protocol_suggest: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _mapping(assessment.get("protocol"))
    hit = _mapping(protocol_suggest.get("clinical_kp_hit"))
    if not hit and protocol_suggest.get("available") is True:
        items = protocol_suggest.get("items")
        if isinstance(items, list) and items and isinstance(items[0], Mapping):
            hit = dict(items[0])
    zone2b = _mapping(zones.get("zone2b"))
    matched = bool(hit) or str(zone2b.get("kp_status") or "") == "matched"
    protocol_id = (
        hit.get("id")
        or hit.get("protocol_id")
        or protocol.get("id")
    )
    protocol_version = (
        hit.get("version")
        or hit.get("protocol_version")
        or protocol.get("version")
    )
    protocol_check = "evaluated" if matched and protocol_id else "not_evaluated"
    return {
        "protocol_check": protocol_check,
        "id": protocol_id or None,
        "version": protocol_version or None,
        "title": hit.get("title") or hit.get("name") or None,
        "applicability_status": (
            "applicable" if protocol_check == "evaluated" else "not_evaluated"
        ),
    }


def _label_revision(label: Mapping[str, Any]) -> str | None:
    for key in (
        "edition",
        "revision",
        "document_date",
        "updated_at",
        "effective_date",
    ):
        value = str(label.get(key) or "").strip()
        if value:
            return value[:80]
    parse = _mapping(label.get("parse"))
    value = str(parse.get("generated_at") or "").strip()
    return value[:80] or None


def build_medication_normative_cards(
    case: Mapping[str, Any] | None,
    *,
    assessment: Mapping[str, Any] | None = None,
    zones: Mapping[str, Any] | None = None,
    protocol_suggest: Mapping[str, Any] | None = None,
    reg55: Mapping[str, Any] | None = None,
    label_ctx: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build text-minimal cards; no normative source changes primary scoring."""
    current = case if isinstance(case, Mapping) else {}
    assessment_map = _mapping(assessment)
    protocol = _protocol_context(
        assessment_map,
        _mapping(zones),
        _mapping(protocol_suggest),
    )
    reg = _mapping(reg55)
    ctx = dict(label_ctx) if isinstance(label_ctx, Mapping) else load_rceth_label_ctx()
    cards: list[dict[str, Any]] = []

    for item in medication_assignments_from_case(current):
        inn = str(item.get("inn") or "").strip().lower()
        form = str(item.get("form") or "").strip() or None
        label = lookup_label(ctx, inn, form) if inn else None
        active = (
            item.get("activity_status") == "active"
            and item.get("assertion") == "confirmed"
            and item.get("subject") == "patient"
        )
        uncertainty: list[str] = []
        if not active:
            uncertainty.append("assignment_not_active_patient_fact")
        if not inn:
            uncertainty.append("drug_identity_unresolved")
        if label is None:
            uncertainty.append("rceth_instruction_unavailable")
        if protocol["protocol_check"] == "not_evaluated":
            uncertainty.append("protocol_not_evaluated")

        instructions: list[dict[str, Any]] = []
        if label is not None:
            instructions.append(
                {
                    "source": "rceth_label",
                    "source_id": label.get("reg_id") or label.get("id"),
                    "revision": _label_revision(label),
                    "applicability": "candidate" if active else "not_applicable",
                    "normative": True,
                }
            )
        if protocol["protocol_check"] == "evaluated":
            instructions.append(
                {
                    "source": "national_protocol",
                    "source_id": protocol["id"],
                    "revision": protocol["version"],
                    "applicability": "candidate",
                    "normative": True,
                }
            )
        if reg:
            instructions.append(
                {
                    "source": "local_reg55_pack",
                    "source_id": reg.get("pack_id"),
                    "revision": reg.get("scorer_version"),
                    "applicability": "methodology_only",
                    "normative": False,
                }
            )

        cards.append(
            {
                "assignment": {
                    "drug_name": item.get("drug_name"),
                    "inn": inn or None,
                    "form": form,
                    "route": item.get("route"),
                    "dose_value": item.get("dose_value"),
                    "dose_unit": item.get("dose_unit"),
                    "frequency": item.get("frequency"),
                    "duration": item.get("duration"),
                    "activity_status": item.get("activity_status"),
                },
                "patient_fact": {
                    "assertion": item.get("assertion"),
                    "subject": item.get("subject"),
                    "fact_time": item.get("fact_time"),
                },
                "protocol_check": protocol["protocol_check"],
                "instructions": instructions,
                "applicability": "candidate" if active else "not_applicable",
                "uncertainty_reason_codes": uncertainty,
                "assessment_status": (
                    "not_evaluated"
                    if protocol["protocol_check"] == "not_evaluated"
                    else "partial"
                ),
                "shadow": True,
                "primary": False,
            }
        )

    return {
        "engine": ENGINE,
        "contract_version": 1,
        "status": "completed" if cards else "empty",
        "protocol": protocol,
        "cards": cards,
        "methodology": {
            "national_scope": "Постановление №55, раздел V",
            "n127_role": "evidence_helper",
            "local_pack_id": reg.get("pack_id") or None,
            "local_pack_is_normative": False,
            "local_pack_label": reg.get("pack_label_ru") or None,
        },
        "primary": False,
        "shadow": True,
    }

