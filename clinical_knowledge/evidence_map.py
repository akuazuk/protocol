"""Построение evidence map для проверенных правил (ТЗ improve_kz §12)."""
from __future__ import annotations

import re
from typing import Any

from .consult_schema import ConsultationDocument, EvidenceMapItem, SourceRef
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
    evidence: list[str] = []
    if low in perf or any(tok in perf for tok in low.split() if len(tok) > 4):
        snippet = next((e.exam_name for e in doc.performed_exams if low[:6] in _norm(e.exam_name)), item)
        return True, "performed", [snippet]
    if low in rec or any(tok in rec for tok in low.split() if len(tok) > 4):
        return True, "recommended", [item]
    return False, "not_found", []


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

    items: list[EvidenceMapItem] = []
    for raw in (rules_check or {}).get("findings") or []:
        if not isinstance(raw, dict):
            continue
        rule = legacy_rule_to_protocol_rule(raw)
        if not rule_applicable_to_patient(rule, patient, diagnosis_certainty=certainty):
            items.append(
                EvidenceMapItem(
                    rule_id=rule.rule_id,
                    rule_type=rule.rule_type,
                    found_in_consultation=False,
                    found_status="not_applicable",
                    decision="not_applicable",
                    explanation="Правило неприменимо по возрасту/полу/беременности/статусу диагноза.",
                    protocol_evidence=rule.expected_items,
                )
            )
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
                EvidenceMapItem(
                    rule_id=rule.rule_id,
                    rule_type=rt,
                    required_item=exam_name,
                    found_in_consultation=found,
                    found_status=fstatus,  # type: ignore[arg-type]
                    consultation_evidence=ev,
                    protocol_evidence=proto_ev[:3],
                    decision=decision,  # type: ignore[arg-type]
                    explanation=expl,
                    source_refs=[rule.source] if rule.source.local_path else [],
                )
            )
            continue

        passed = bool(raw.get("passed"))
        ev_text = _consult_blobs(doc).get(rule.evidence_targets[0], "") if rule.evidence_targets else ""
        consultation_ev = [ev_text[:200]] if ev_text.strip() else []
        if passed:
            decision = "satisfied"
            expl = "Требование протокола подтверждено текстом КЗ."
            fstatus = "mentioned"
        elif rule.confidence < 0.5:
            decision = "manual_review"
            expl = "Правило извлечено с низкой уверенностью — требуется ручная проверка."
            fstatus = "unknown"
        else:
            decision = "missing"
            expl = raw.get("message_ru") or "Требование не выполнено."
            fstatus = "not_found"

        items.append(
            EvidenceMapItem(
                rule_id=rule.rule_id,
                rule_type=rt,
                required_item=required_item,
                found_in_consultation=passed,
                found_status=fstatus,  # type: ignore[arg-type]
                consultation_evidence=consultation_ev,
                protocol_evidence=proto_ev[:3],
                decision=decision,  # type: ignore[arg-type]
                explanation=expl,
                source_refs=[rule.source] if rule.source.local_path else [],
            )
        )
    return items
