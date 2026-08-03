"""Построение evidence map для проверенных правил (ТЗ improve_kz §12)."""
from __future__ import annotations

import re
from typing import Any

from .condition_registry import infer_conditions_hints
from .consult_schema import ConsultationDocument, EvidenceMapItem
from .rule_labels_ru import (
    decision_ru,
    extract_condition_id,
    found_status_ru,
    localize_message_ru,
    rule_title_ru,
    rule_type_ru,
)
from .rule_model import ProtocolRule, legacy_rule_to_protocol_rule, rule_applicable_to_patient

_EXAM_BLOB_MARKERS = re.compile(
    r"оак|общий\s+анализ\s+кров|узи|фгдс|эгдс|кт|мрт|колоноскоп|экг|биопси|ana|anti-dna",
    re.I,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _consult_blobs(doc: ConsultationDocument) -> dict[str, str]:
    s = doc.sections
    performed = " ".join(e.exam_name for e in doc.performed_exams)
    recommended = " ".join(
        e.exam_name for e in doc.planned_exams
    ) + " " + (s.recommendations_exams or "")
    return {
        "complaints": s.complaints or "",
        "anamnesis": s.anamnesis or "",
        "objective_status": s.objective_status or "",
        "local_status": s.local_status or "",
        "performed_exams": performed + " " + (s.exam_results or ""),
        "recommended_exams": recommended,
        "diagnosis": (s.diagnosis_text or "") + " " + " ".join(d.raw_text for d in doc.diagnoses),
        "treatment": (s.recommendations_treatment or "") + " " + " ".join(m.raw_text for m in doc.medications),
        "medications": " ".join(m.raw_text for m in doc.medications),
        "follow_up": (s.follow_up_text or "") + " " + " ".join(
            d.raw_text or "" for d in doc.follow_up
        ),
        "routing": s.routing or "",
    }


def _exam_status(doc: ConsultationDocument, item: str) -> tuple[bool, str, list[str]]:
    low = _norm(item)
    if len(low) < 3:
        return False, "unknown", []
    blobs = _consult_blobs(doc)
    perf = _norm(blobs["performed_exams"])
    rec = _norm(blobs["recommended_exams"])
    if low in perf or any(tok in perf for tok in low.split() if len(tok) > 4):
        snippet = next((e.exam_name for e in doc.performed_exams if low[:6] in _norm(e.exam_name)), item)
        return True, "performed", [snippet]
    if low in rec or any(tok in rec for tok in low.split() if len(tok) > 4):
        return True, "recommended", [item]
    return False, "not_found", []


def _relevant_rule(raw: dict[str, Any], hints: set[str]) -> bool:
    if raw.get("skipped"):
        return False
    if raw.get("rule_source") == "summary":
        return True
    rid = str(raw.get("rule_id") or "")
    cid = extract_condition_id(rid)
    if not cid:
        return True
    if cid in hints:
        return True
    rt = str(raw.get("rule_type") or "")
    if rt in ("population_mismatch",) or rid.endswith("_population_guard"):
        return False
    # Авто-правила чужих нозологий не показываем в карте доказательств.
    if rid.startswith(("llm_", "tbl_")) or "_auto_" in rid or re.match(r"^[a-f0-9]{8}_auto_", rid):
        return False
    return bool(raw.get("passed"))


def _make_item(
    *,
    rule: ProtocolRule,
    raw: dict[str, Any],
    decision: str,
    expl: str,
    found: bool,
    fstatus: str,
    ev: list[str],
    required_item: str | None,
    proto_ev: list[str],
) -> EvidenceMapItem:
    expl_ru = localize_message_ru(expl)
    return EvidenceMapItem(
        rule_id=rule.rule_id,
        rule_source=str(raw.get("rule_source") or rule.rule_source or "legacy"),  # type: ignore[arg-type]
        protocol_id=rule.protocol_id or (rule.source.protocol_id if rule.source else None),
        condition_id=rule.condition_id or extract_condition_id(rule.rule_id),
        title_ru=rule_title_ru(rule.rule_id, raw),
        rule_type=rule.rule_type,
        rule_type_ru=rule_type_ru(rule.rule_type),
        required_item=required_item,
        found_in_consultation=found,
        found_status=fstatus,  # type: ignore[arg-type]
        found_status_ru=found_status_ru(fstatus),
        consultation_evidence=ev,
        protocol_evidence=proto_ev[:3],
        decision=decision,  # type: ignore[arg-type]
        decision_ru=decision_ru(decision),
        explanation=expl_ru,
        source_refs=[rule.source] if rule.source.local_path else [],
    )


def build_evidence_map(
    doc: ConsultationDocument,
    rules_check: dict[str, Any] | None,
    *,
    patient: dict[str, Any] | None = None,
) -> list[EvidenceMapItem]:
    """Строит evidence map из findings rule_checker и expected items."""
    patient = patient or {
        "age_years": doc.patient.age_years,
        "sex": doc.patient.sex,
        "pregnancy": doc.patient.pregnancy,
        "adult_or_child": doc.patient.adult_or_child,
    }
    certainty = "confirmed"
    if doc.diagnoses:
        certainty = doc.diagnoses[0].certainty or "unclear"

    icd = [d.icd10_code for d in doc.diagnoses if d.icd10_code]
    hints = set(
        infer_conditions_hints((doc.raw_text or "").lower(), icd)
    )

    items: list[EvidenceMapItem] = []
    for raw in (rules_check or {}).get("findings") or []:
        if not isinstance(raw, dict):
            continue
        if not _relevant_rule(raw, hints):
            continue

        rule = legacy_rule_to_protocol_rule(raw)
        if not rule_applicable_to_patient(rule, patient, diagnosis_certainty=certainty):
            continue

        rt = rule.rule_type
        required_item = rule.expected_items[0] if rule.expected_items else None
        proto_ev = rule.expected_items + [raw.get("message_ru") or ""]
        proto_ev = [p for p in proto_ev if p]

        if rt in ("required_exam_rule", "conditional_exam_rule", "performed_or_recommended_exam_rule"):
            exam_name = required_item or "обследование"
            found, fstatus, ev = _exam_status(doc, exam_name)
            if found and fstatus == "performed":
                decision = "satisfied"
                expl = f"Обследование «{exam_name}» выполнено."
            elif found and fstatus == "recommended":
                decision = "satisfied_by_recommendation"
                expl = f"Обследование «{exam_name}» назначено."
            elif raw.get("passed") is True:
                decision = "satisfied"
                expl = raw.get("message_ru") or "Требование выполнено."
                fstatus = "mentioned"
                found = True
            else:
                decision = "missing"
                expl = raw.get("message_ru") or f"Отсутствует: {exam_name}."
                fstatus = "not_found"
            items.append(
                _make_item(
                    rule=rule, raw=raw, decision=decision, expl=expl,
                    found=found, fstatus=fstatus, ev=ev,
                    required_item=exam_name, proto_ev=proto_ev,
                )
            )
            continue

        passed = bool(raw.get("passed"))
        ev_text = _consult_blobs(doc).get(rule.evidence_targets[0], "") if rule.evidence_targets else ""
        consultation_ev = [ev_text[:200]] if ev_text.strip() else []
        if passed:
            decision = "satisfied"
            expl = raw.get("message_ru") or "Требование протокола подтверждено текстом КЗ."
            fstatus = "mentioned"
        elif rule.confidence < 0.5:
            decision = "manual_review"
            expl = "Правило извлечено с низкой уверенностью - требуется ручная проверка."
            fstatus = "unknown"
        else:
            decision = "missing"
            expl = raw.get("message_ru") or "Требование не выполнено."
            fstatus = "not_found"

        items.append(
            _make_item(
                rule=rule, raw=raw, decision=decision, expl=expl,
                found=passed, fstatus=fstatus, ev=consultation_ev,
                required_item=required_item, proto_ev=proto_ev,
            )
        )
    return items
