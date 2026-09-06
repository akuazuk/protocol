"""Shadow label-check Rceth: off-label / противопоказание / возраст (не в SSOT)."""
from __future__ import annotations

import os
import re
from typing import Any

from clinical_knowledge.drug_normalizer import extract_drugs
from clinical_knowledge.rceth_sync.identity import canon_inn
from clinical_knowledge.rceth_sync.label_ctx import load_rceth_label_ctx, lookup_label

ENGINE = "mo_rceth_label_v2"
_SOURCE = "rceth_label_v1"
_CONF_MIN = 0.86

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{4,}")
_STOP = {
    "лечение", "симптоматическое", "применение", "слабой", "средней", "интенсивности",
    "острых", "острой", "приема", "приёма", "внутрь", "детям", "детей", "взрослых",
    "препарат", "средства", "вспомогательным", "веществам", "фаза", "фазе",
    "таблетки", "капсулы", "раствор", "суспензия",
}

_CONTRA_PAIRS = (
    (("язв",), ("язв", "язвенн")),
    (("беремен",), ("беремен",)),
    (("сердечн", "недостаточ"), ("сердечн", "недостаточ")),
    (("почечн", "недостаточ"), ("почечн", "недостаточ")),
    (("печеночн", "недостаточ"), ("печеночн", "недостаточ")),
    (("гиперчувствительн",), ("аллерг", "гиперчувствительн")),
)



def rceth_label_findings_enabled() -> bool:
    raw = (os.environ.get("MO_RCETH_LABEL_FINDINGS") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def rceth_label_primary_enabled() -> bool:
    raw = (os.environ.get("MO_RCETH_LABEL_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _txt(case: dict[str, Any], *keys: str) -> str:
    parts: list[str] = []
    for key in keys:
        val = case.get(key)
        if val:
            parts.append(str(val))
    return " ".join(parts)


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(text or ""):
        tok = raw.lower().replace("ё", "е")
        if tok in _STOP:
            continue
        out.add(tok)
    return out


def _overlap(a: set[str], b: set[str]) -> bool:
    if not a or not b:
        return False
    for left in a:
        for right in b:
            if left == right or left.startswith(right) or right.startswith(left):
                return True
    return False


def _section_text(label: dict[str, Any], key: str) -> str:
    secs = label.get("sections") if isinstance(label.get("sections"), dict) else {}
    chunks = secs.get(key) or []
    if isinstance(chunks, list):
        return " ".join(str(x) for x in chunks)
    return str(chunks or "")


def _edition(label: dict[str, Any]) -> str:
    changes = label.get("nd_changes") or []
    if isinstance(changes, list) and changes:
        return str(changes[-1])[:80]
    return str(label.get("term_from") or label.get("reg_id") or "")


def _unit_years(num: int, unit: str) -> float:
    u = (unit or "").lower()
    if u.startswith("мес"):
        return num / 12.0
    return float(num)


def _min_age_from_contra(text: str) -> float | None:
    """«Детский возраст до 3 месяцев» → 0.25 года."""
    m = re.search(
        r"(?:детск\w*\s+возраст|возраст)\s+до\s+(\d+)\s*(месяц|месяца|месяцев|мес|лет|года|год)",
        text or "",
        re.I,
    )
    if not m:
        return None
    return _unit_years(int(m.group(1)), m.group(2))


def _patient_age(case: dict[str, Any]) -> float | None:
    raw = case.get("patient_age_years")
    if raw is None:
        raw = case.get("age_years")
    try:
        age = float(raw)
    except (TypeError, ValueError):
        return None
    if age < 0 or age > 120:
        return None
    return age


def _citation(surface: str, inn: str, label: dict[str, Any]) -> str:
    rid = str(label.get("reg_id") or "")
    edition = _edition(label)
    bit = f", ред. {edition}" if edition else ""
    return f"{surface} ({inn}) - инструкция rceth {rid}{bit}"


def _finding(
    code: str,
    *,
    title: str,
    detail: str,
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "axis": "safety",
        "severity": "P2",
        "passed": False,
        "title_ru": title,
        "detail_ru": detail,
        "evidence": (evidence or "")[:400],
        "source_ref": _SOURCE,
        "needs_human": True,
        "assessment_status": "candidate",
        "shadow": True,
        "engine": ENGINE,
        "linked_fields": [
            "treatment_recommendations",
            "clinical_diagnosis",
            "mkb_code_main",
        ],
        "link_hint_ru": "Сверьте назначение с официальной инструкцией ЛС (rceth)",
    }


def _off_label(dx_text: str, indications: str) -> bool:
    dx_toks = _tokens(dx_text)
    ind_toks = _tokens(indications)
    if len(dx_toks) < 2 or len(ind_toks) < 2:
        return False
    return not _overlap(dx_toks, ind_toks)


# Conservative sentence guard, not a clinical assertion extractor. Ambiguous
# clauses are withheld; retained matches still require human review.
_UNASSERTED = re.compile(
    r"\b(?:нет|без|отриц\w*|не\s+(?:выяв\w*|отмеч\w*|подтверж\w*|страда\w*|беремен\w*)|"
    r"исключ\w*|подозрен\w*|возможн\w*|риск\w*|вероятн\w*|отсутств\w*|"
    r"семейн\w*|наследствен\w*|у\s+(?:матери|отца|мамы|папы|сестры|брата|бабушки|дедушки))\b|\?"
)


def _contra_hit(clinical: str, contra: str) -> str | None:
    blob_c = (clinical or "").lower().replace("ё", "е")
    blob_l = (contra or "").lower().replace("ё", "е")
    clauses = [part for part in re.split(r"[.!;\n]+", blob_c) if not _UNASSERTED.search(part)]
    for needles_l, needles_c in _CONTRA_PAIRS:
        if not all(n in blob_l for n in needles_l):
            continue
        for clause in clauses:
            # Hypersensitivity/allergy are synonyms; organ failure requires both
            # organ and failure in the same asserted clause, never an isolated word.
            matched = (any(n in clause for n in needles_c) if needles_l == ("гиперчувствительн",)
                       else all(n in clause for n in needles_c))
            if matched:
                return needles_l[0]
    return None


def evaluate_rceth_label_findings(
    case: dict[str, Any] | None,
    label_ctx: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not rceth_label_findings_enabled() or not case:
        return []
    treatment = str(case.get("treatment_recommendations") or "").strip()
    if not treatment:
        return []
    try:
        drugs = extract_drugs(treatment) or []
    except Exception:  # noqa: BLE001
        return []
    ctx = label_ctx if label_ctx is not None else load_rceth_label_ctx()
    if not ctx or not ctx.get("by_inn"):
        return []

    dx = _txt(case, "clinical_diagnosis", "diagnosis_main_text", "mkb_name")
    icd = str(case.get("mkb_code_main") or case.get("diagnosis_code") or "").strip()
    if icd:
        dx = f"{dx} {icd}".strip()
    clinical = "\n".join(_txt(case, key) for key in (
        "clinical_diagnosis", "diagnosis_main_text", "complaints", "anamnesis_doctor"
    ))
    age = _patient_age(case)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for drug in drugs:
        inn = canon_inn(str(drug.get("inn") or "")) or str(drug.get("inn") or "").strip().lower()
        if not inn:
            continue
        try:
            conf = float(drug.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf and conf < _CONF_MIN:
            continue
        forms = drug.get("forms") if isinstance(drug.get("forms"), list) else []
        form = str(forms[0]) if forms else None
        label = lookup_label(ctx, inn, form)
        if not label:
            continue
        surface = str(drug.get("surface") or inn)
        cite = _citation(surface, inn, label)
        indications = _section_text(label, "indications_4_1")
        contra = _section_text(label, "contraindications_4_3")

        if age is not None:
            min_contra = _min_age_from_contra(contra)
            if min_contra is not None and age < min_contra:
                code = "C_rceth_age_outside_label"
                key = f"{code}:{inn}"
                if key not in seen:
                    seen.add(key)
                    out.append(
                        _finding(
                            code,
                            title="Возраст вне инструкции ЛС",
                            detail=cite,
                            evidence=contra,
                        )
                    )
            # A pediatric dosing subsection does not establish a global maximum
            # age. Only explicit contraindication age text is checked above.

        hit = _contra_hit(clinical, contra) if contra else None
        if hit:
            code = "C_rceth_contraindication"
            key = f"{code}:{inn}:{hit}"
            if key not in seen:
                seen.add(key)
                out.append(
                    _finding(
                        code,
                        title="Возможное противопоказание: нужна сверка",
                        detail=cite,
                        evidence=contra,
                    )
                )

        if indications and dx and _off_label(dx, indications):
            code = "C_rceth_off_label"
            key = f"{code}:{inn}"
            if key not in seen:
                seen.add(key)
                out.append(
                    _finding(
                        code,
                        title="Показание к назначению требует сверки",
                        detail=cite,
                        evidence=indications,
                    )
                )
    return out


def merge_rceth_label_into_findings(
    findings: list[dict[str, Any]] | None,
    case: dict[str, Any] | None,
    label_ctx: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out = [dict(item) for item in (findings or []) if isinstance(item, dict)]
    if not rceth_label_findings_enabled() or not case:
        return out
    existing = {str(item.get("code") or item.get("finding_code") or "") for item in out}
    try:
        shadow = evaluate_rceth_label_findings(case, label_ctx=label_ctx)
    except Exception:  # noqa: BLE001
        return out
    for item in shadow:
        code = str(item.get("code") or "")
        if not code or code in existing:
            continue
        row = dict(item)
        row["is_shadow"] = True
        out.append(row)
        existing.add(code)
    return out
