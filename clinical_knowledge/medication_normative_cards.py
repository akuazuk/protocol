"""Shadow medication/normative cards for MO case review."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from clinical_knowledge.medication_parser import medication_assignments_from_case
from clinical_knowledge.rceth_sync.label_ctx import load_rceth_label_ctx, lookup_label

ENGINE = "mo_medication_normative_cards_v1"
CONTRACT_VERSION = 2
SECTION_CLIP = 220
MAX_KP_NAMES = 24


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clip(value: Any, limit: int = SECTION_CLIP) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _section_text(label: Mapping[str, Any], key: str) -> str:
    sections = label.get("sections") if isinstance(label.get("sections"), Mapping) else {}
    chunks = sections.get(key) if isinstance(sections, Mapping) else None
    if isinstance(chunks, list):
        return _clip(" ".join(str(item).strip() for item in chunks if str(item).strip()))
    return _clip(chunks)


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
    changes = label.get("nd_changes")
    if isinstance(changes, list) and changes:
        last = str(changes[-1] or "").strip()
        if last:
            return last[:80]
    for key in (
        "edition",
        "revision",
        "document_date",
        "updated_at",
        "effective_date",
        "term_from",
    ):
        value = str(label.get(key) or "").strip()
        if value:
            return value[:80]
    parse = _mapping(label.get("parse"))
    value = str(parse.get("generated_at") or "").strip()
    return value[:80] or None


def _unique_names(*groups: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, (list, tuple)):
            continue
        for raw in group:
            name = str(raw or "").strip()
            key = name.lower()
            if len(name) < 3 or key in seen:
                continue
            seen.add(key)
            out.append(name)
            if len(out) >= MAX_KP_NAMES:
                return out
    return out


def _summary_drug_names(summary: Any) -> list[str]:
    names: list[str] = []
    for cond in list(getattr(summary, "conditions", None) or []):
        block = getattr(cond, "treatment", None)
        if block is None:
            continue
        for drug in list(getattr(block, "drugs", None) or []):
            names.extend(
                [
                    getattr(drug, "drug_name", None),
                    getattr(drug, "active_substance", None),
                    getattr(drug, "drug_group", None),
                ]
            )
        for group in list(getattr(block, "drug_groups", None) or []):
            names.append(getattr(group, "drug_group", None))
    return _unique_names(names)


def _concordance_drug_names(protocol_suggest: Mapping[str, Any]) -> list[str]:
    conc = _mapping(protocol_suggest.get("kp_concordance"))
    names: list[str] = []
    aliases: list[str] = []
    for row in conc.get("rows") or []:
        if not isinstance(row, Mapping) or str(row.get("kind") or "") != "treatment":
            continue
        names.append(row.get("requirement"))
        if isinstance(row.get("aliases"), list):
            aliases.extend(row.get("aliases") or [])
    return _unique_names(names, aliases)


def _load_kp_summary(
    protocol_suggest: Mapping[str, Any],
    *,
    summary: Any = None,
    summary_loader: Callable[[str], Any] | None = None,
) -> Any:
    if summary is not None:
        return summary
    from clinical_knowledge.mo_plan_protocol_score import resolve_plan_route

    route = resolve_plan_route(protocol_suggest)
    hit = route.get("hit") if isinstance(route.get("hit"), Mapping) else {}
    source_path = str(hit.get("source_path") or hit.get("local_path") or "")
    if not source_path:
        return None
    loader = summary_loader
    if loader is None:
        try:
            from clinical_knowledge.protocol_summary.nav import find_summary_by_catalog_path

            loader = find_summary_by_catalog_path
        except Exception:
            return None
    try:
        return loader(source_path)
    except Exception:
        return None


def kp_scheme_for_suggest(
    protocol_suggest: Mapping[str, Any] | None,
    *,
    summary: Any = None,
    summary_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Drug names from the matched protocol card. Empty is not a plan-zone fail."""
    suggest = _mapping(protocol_suggest)
    from clinical_knowledge.mo_plan_protocol_score import resolve_plan_route

    route = resolve_plan_route(suggest)
    hit = route.get("hit") if isinstance(route.get("hit"), Mapping) else {}
    protocol_id = hit.get("protocol_id") or hit.get("id")
    if str(route.get("kp_status") or "") != "matched":
        return {
            "status": "unmatched",
            "has_scheme": False,
            "names": [],
            "protocol_id": protocol_id,
        }
    loaded = _load_kp_summary(suggest, summary=summary, summary_loader=summary_loader)
    names = _summary_drug_names(loaded) if loaded is not None else []
    if not names:
        names = _concordance_drug_names(suggest)
    if not names:
        return {
            "status": "no_drugs_in_kp",
            "has_scheme": False,
            "names": [],
            "protocol_id": protocol_id,
        }
    return {
        "status": "has_scheme",
        "has_scheme": True,
        "names": names,
        "protocol_id": protocol_id,
    }


def _in_kp_scheme(inn: str, drug_name: str, names: list[str]) -> tuple[bool, str]:
    needles = [item for item in (inn, drug_name) if item and len(item.strip()) >= 3]
    for name in names:
        blob = name.lower()
        for needle in needles:
            key = needle.strip().lower()
            if key in blob or blob in key:
                return True, name
    return False, ""


def _card_kp_status(scheme: Mapping[str, Any], inn: str, drug_name: str) -> dict[str, Any]:
    status = str(scheme.get("status") or "unmatched")
    if status != "has_scheme":
        return {"status": status, "match_name": None, "has_scheme": False}
    found, match_name = _in_kp_scheme(inn, drug_name, list(scheme.get("names") or []))
    return {
        "status": "in_scheme" if found else "not_in_scheme",
        "match_name": match_name or None,
        "has_scheme": True,
    }


def _rceth_payload(label: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(label, Mapping):
        return {
            "available": False,
            "revision": None,
            "reg_id": None,
            "indications_4_1": "",
            "contraindications_4_3": "",
        }
    revision = _label_revision(label)
    return {
        "available": True,
        "revision": revision,
        "reg_id": label.get("reg_id") or label.get("id"),
        "indications_4_1": _section_text(label, "indications_4_1"),
        "contraindications_4_3": _section_text(label, "contraindications_4_3"),
    }


def build_medication_normative_cards(
    case: Mapping[str, Any] | None,
    *,
    assessment: Mapping[str, Any] | None = None,
    zones: Mapping[str, Any] | None = None,
    protocol_suggest: Mapping[str, Any] | None = None,
    reg55: Mapping[str, Any] | None = None,
    label_ctx: Mapping[str, Any] | None = None,
    summary: Any = None,
    summary_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Build text-minimal cards; no normative source changes primary scoring."""
    current = case if isinstance(case, Mapping) else {}
    assessment_map = _mapping(assessment)
    suggest = _mapping(protocol_suggest)
    protocol = _protocol_context(
        assessment_map,
        _mapping(zones),
        suggest,
    )
    scheme = kp_scheme_for_suggest(
        suggest,
        summary=summary,
        summary_loader=summary_loader,
    )
    reg = _mapping(reg55)
    ctx = dict(label_ctx) if isinstance(label_ctx, Mapping) else load_rceth_label_ctx()
    cards: list[dict[str, Any]] = []

    for item in medication_assignments_from_case(current):
        inn = str(item.get("inn") or "").strip().lower()
        drug_name = str(item.get("drug_name") or "").strip()
        form = str(item.get("form") or "").strip() or None
        label = lookup_label(ctx, inn, form) if inn else None
        rceth = _rceth_payload(label)
        kp = _card_kp_status(scheme, inn, drug_name)
        draft = (not rceth["available"]) and (not kp["has_scheme"])
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
                    "revision": rceth["revision"],
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
                "rceth": rceth,
                "kp_scheme": kp,
                "draft": draft,
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
        "contract_version": CONTRACT_VERSION,
        "status": "completed" if cards else "empty",
        "protocol": protocol,
        "kp_scheme": {
            "status": scheme["status"],
            "has_scheme": scheme["has_scheme"],
            "names": list(scheme.get("names") or []),
            "protocol_id": scheme.get("protocol_id"),
        },
        "cards": cards,
        "methodology": {
            "national_scope": "Постановление №55, раздел V",
            "n127_role": "evidence_helper",
            "local_pack_id": reg.get("pack_id") or None,
            "local_pack_is_normative": False,
            "local_pack_label": reg.get("pack_label_ru") or None,
            "evidence_role": "context",
            "safety_role": "risk_not_plan_zone",
        },
        "primary": False,
        "shadow": True,
    }
