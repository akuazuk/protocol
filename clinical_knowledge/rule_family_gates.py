"""Семейные gates для rule_checker: ICD + specialty → применимость нозологии."""
from __future__ import annotations

import re
from typing import Any


def is_oncology_icd(code: str) -> bool:
    """ЗНО-контекст: C*, D00-D09 in situ, D37-D48 uncertain; не D10-D36 (доброкач.)."""
    c = (code or "").upper().strip()
    if not c:
        return False
    if c.startswith("C"):
        return True
    if c.startswith("D") and len(c) >= 3 and c[1:3].isdigit():
        n = int(c[1:3])
        if 0 <= n <= 9:
            return True
        if 10 <= n <= 36:
            return False
        if 37 <= n <= 48:
            return True
    return False


def expand_specialty_slugs_for_icd(
    slugs: set[str] | list[str] | None,
    icd_codes: list[str] | None,
) -> set[str]:
    """Доп. рубрики каталога по МКБ (ОРВИ → пульмонология/инфекции; D25 → акушерство)."""
    out = {s.strip() for s in (slugs or []) if s and str(s).strip()}
    for raw in icd_codes or []:
        c = str(raw or "").upper().strip()
        if not c:
            continue
        root = c[:3] if len(c) >= 3 else c
        if root.startswith("J06") or root in {"J00", "J02", "J03", "J04", "J05", "J11"}:
            out.update(
                {
                    "pulmonologiya-ftiziatriya",
                    "infektsionnye-zabolevaniya",
                    "terapiya",
                    "otorinolaringologiya",
                }
            )
        if root.startswith("R07") or c.startswith("R07"):
            out.update(
                {
                    "otorinolaringologiya",
                    "pulmonologiya-ftiziatriya",
                    "infektsionnye-zabolevaniya",
                    "terapiya",
                }
            )
        if root.startswith("D25") or root.startswith(("N80", "N81", "N82", "N83", "N84", "N85", "N86", "N87", "N88", "N89", "N90", "N91", "N92", "N93", "N94", "N95", "N97")):
            out.add("akusherstvo-ginekologiya")
        if c.startswith("O") or root.startswith("O"):
            out.add("akusherstvo-ginekologiya")
        if root.startswith(("I10", "I11", "I12", "I13", "I15", "I50")):
            out.update({"bolezni-sistemy-krovoobrashcheniya", "terapiya"})
        if root.startswith(("M32", "M05", "M06")):
            out.add("revmatologiya")
        if root.startswith("J30"):
            out.update({"allergologiya-immunologiya", "otorinolaringologiya"})
        if c.startswith("T2") or c.startswith("T3"):
            out.update({"khirurgiya", "anesteziologiya-reanimatologiya"})
        if root.startswith(("E03", "E04", "E05", "E06", "E55", "E58", "E59")):
            out.add("endokrinologiya-narusheniya-obmena-veshchestv")
        if root in {"M51", "M53", "M54"} or (c.startswith("M5") and len(root) >= 3):
            out.update({"nevrologiya-neyrokhirurgiya", "travmatologiya-ortopediya"})
        if c.startswith("M"):
            out.add("travmatologiya-ortopediya")
        if root.startswith(("I80", "I81", "I82", "I83")):
            out.update({"khirurgiya", "bolezni-sistemy-krovoobrashcheniya"})
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _text_blob(consult_facts: dict[str, Any]) -> str:
    cons = consult_facts.get("consultation") or {}
    parts = [
        cons.get("diagnosis_text") or "",
        " ".join(cons.get("complaints") or []),
        cons.get("clinical_text") or "",
        cons.get("text_sample") or "",
    ]
    return _norm(" ".join(parts))


_ONCOLOGY_SUSPICION_MARKERS = (
    "опухолевое образование",
    "картина опухол",
    "нельзя исключить инваз",
    "подозрени на зло",
    "подозрени на рак",
    "злокачествен",
    "новообразован",
    "метастаз",
)


def has_oncology_clinical_suspicion(consult_facts: dict[str, Any]) -> bool:
    """Подозрение на ЗНО по тексту КЗ (не только код C/D)."""
    cons = consult_facts.get("consultation") or {}
    icd_list = [str(x).upper() for x in (cons.get("icd10") or []) if x]
    if any(is_oncology_icd(c) for c in icd_list):
        return True
    blob = _text_blob(consult_facts)
    return any(m in blob for m in _ONCOLOGY_SUSPICION_MARKERS)


def expand_specialty_slugs_for_clinical_text(
    slugs: set[str] | list[str] | None,
    text: str,
) -> set[str]:
    """Доп. рубрики по клиническому тексту (опухоль сигмы → novoobrazovaniya)."""
    out = {s.strip() for s in (slugs or []) if s and str(s).strip()}
    low = _norm(text or "")
    if any(m in low for m in _ONCOLOGY_SUSPICION_MARKERS):
        out.update({"novoobrazovaniya", "gastroenterologiya", "khirurgiya"})
    try:
        from clinical_knowledge.search_clinical_routing import expand_slugs_for_clinical_routes

        out = expand_slugs_for_clinical_routes(out, text or "", None)
    except Exception:
        pass
    return out


def _has_thyroid_diagnosis_context(diag: str, icd_list: list[str]) -> bool:
    if any(c.startswith(("E03", "E04", "E05", "E06")) for c in icd_list):
        return True
    markers = ("гипотиреоз", "гипертиреоз", "тиреоидит", " аит", "аит,", "зоб ")
    return any(m in diag for m in markers) or ("щитовид" in diag and "?" in diag)


def _is_dyspepsia_primary_visit(diag: str, icd_list: list[str]) -> bool:
    if any(c.startswith("K30") for c in icd_list):
        return True
    return bool(diag) and "диспепс" in diag and "гастрит" not in diag


def _diag_complaints(consult_facts: dict[str, Any]) -> str:
    cons = consult_facts.get("consultation") or {}
    return _norm(f"{cons.get('diagnosis_text') or ''} {' '.join(cons.get('complaints') or [])}")


def _has_icd_prefix(icd_list: list[str], prefixes: tuple[str, ...]) -> bool:
    return any(any(c.startswith(p) for p in prefixes) for c in icd_list)


def condition_family_applies(condition_id: str, consult_facts: dict[str, Any]) -> bool | None:
    """None — использовать стандартную логику rule_checker."""
    cons = consult_facts.get("consultation") or {}
    icd_list = [str(x).upper() for x in (cons.get("icd10") or []) if x]
    diag = _norm(str(cons.get("diagnosis_text") or ""))

    if condition_id == "acute_bronchitis":
        has_bronchitis_icd = any(c.startswith(("J20", "J21")) for c in icd_list)
        has_bronchitis_text = "бронхит" in diag
        if not has_bronchitis_icd and not has_bronchitis_text:
            ent_prefixes = ("H65", "H66", "H67", "H68", "H69", "H70", "H71", "H72", "H73", "H74", "H75", "J01", "J02", "J03", "J32")
            if icd_list and all(any(c.startswith(p) for p in ent_prefixes) for c in icd_list):
                return False
            if any(c.startswith(ent_prefixes) for c in icd_list):
                return False
            urti_only = bool(icd_list) and all(
                c.startswith(("J00", "J01", "J02", "J03", "J04", "J05", "J06")) for c in icd_list
            )
            if urti_only:
                return False

    if condition_id == "obesity":
        if icd_list and all(c.startswith(("E03", "E04", "E05", "E06")) for c in icd_list):
            return False
        if "эутиреоз" in diag or "euthyre" in diag:
            return False

    if condition_id in ("tuberculosis", "pneumonia", "bronchial_asthma"):
        urti_only = bool(icd_list) and all(
            c.startswith(("J00", "J01", "J02", "J03", "J04", "J05", "J06")) for c in icd_list
        )
        if urti_only:
            return False

    if condition_id == "dermatitis":
        if icd_list and not any(c.startswith("L") for c in icd_list):
            blob = _text_blob(consult_facts)
            if not any(m in blob for m in ("дерматит", "экзем", "кож", "сып")):
                return False

    if condition_id == "thyroid_disease":
        if not _has_thyroid_diagnosis_context(diag, icd_list):
            return False

    if condition_id == "gastritis":
        if _is_dyspepsia_primary_visit(diag, icd_list):
            if not any(c.startswith("K29") for c in icd_list) and "гастрит" not in diag:
                return False

    if condition_id == "abdominal_trauma":
        if not any(c.startswith("S36") for c in icd_list):
            complaints = _norm(" ".join(cons.get("complaints") or []))
            if "травм" not in diag and "травм" not in complaints:
                return False

    if condition_id == "foreign_body_gi":
        dc = _diag_complaints(consult_facts)
        if not _has_icd_prefix(icd_list, ("T18",)) and "инородн" not in dc:
            return False

    if condition_id == "intussusception":
        dc = _diag_complaints(consult_facts)
        if not _has_icd_prefix(icd_list, ("K56",)) and "инвагинац" not in dc:
            return False

    if condition_id == "defecation_disorder":
        dc = _diag_complaints(consult_facts)
        if not _has_icd_prefix(icd_list, ("K59", "R15", "R19")) and not any(
            m in dc for m in ("запор", "дисхез", "дефекац", "недержан", "стул")
        ):
            return False

    if condition_id == "bladder_dysfunction":
        urology_surface = ("N40", "N41", "N42", "N43", "N44", "N45", "N46", "N47", "N48", "N49", "N50")
        bladder_icd = ("N30", "N31", "N32", "N33", "N34", "N35", "N36", "N37", "N38", "N39")
        if icd_list and _has_icd_prefix(icd_list, urology_surface):
            if not _has_icd_prefix(icd_list, bladder_icd):
                dc = _diag_complaints(consult_facts)
                if not any(m in dc for m in ("дизур", "недержан", "инконтинен", "мочевой пузыр", "цистит", "n31", "n39")):
                    return False

    if condition_id == "intestinal_obstruction":
        dc = _diag_complaints(consult_facts)
        if not _has_icd_prefix(icd_list, ("K56",)) and "непроходим" not in dc:
            if _is_dyspepsia_primary_visit(diag, icd_list):
                return False

    if condition_id in ("functional_dyspepsia", "gastritis", "gerd") and has_oncology_clinical_suspicion(consult_facts):
        return False

    if condition_id == "neoplasm":
        if not any(is_oncology_icd(c) for c in icd_list):
            suspicion_only = any(
                m in diag
                for m in (
                    "нельзя исключить",
                    "картина опухол",
                    "опухолевое образование",
                    "подозрени",
                )
            )
            established = any(m in diag for m in ("злокачествен", "carcinoma", "рак ", "стадия", "tnm"))
            if suspicion_only and not established:
                return False

    return None
