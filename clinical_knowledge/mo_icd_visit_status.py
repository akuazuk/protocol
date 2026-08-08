"""Статус МКБ на визит для МО Аналитика (чип + сводка).

Проверки владельца:
1) диагноз есть в МО;
2) диагноз/код существует в справочнике МКБ.

История пациента - отдельный трек; здесь только per-visit.
"""
from __future__ import annotations

from typing import Any

# Приоритет для чипа (хуже выше)
_STATUS_RANK = {
    "missing_dx": 0,
    "not_in_directory": 1,
    "weak_name": 2,
    "ok": 3,
    "unknown": 4,
}

_CODE_TO_STATUS = {
    "B_dx_absent": "missing_dx",
    "B_icd_dir_code_unknown": "not_in_directory",
    "B_icd_dir_no_match": "not_in_directory",
    "B_icd_name_no_match": "not_in_directory",
    "B_icd_dir_text_mismatch": "weak_name",
    "B_icd_name_weak_match": "weak_name",
}

CHIP_LABEL_RU = {
    "ok": "МКБ ✓",
    "missing_dx": "нет Dx",
    "not_in_directory": "не в МКБ",
    "weak_name": "слабо МКБ",
    "unknown": "МКБ ?",
}

CHIP_TITLE_RU = {
    "ok": "Диагноз есть и сопоставлен со справочником МКБ",
    "missing_dx": "В МО нет формулировки диагноза и кода МКБ",
    "not_in_directory": "Диагноз или код не найдены в справочнике МКБ",
    "weak_name": "Формулировка слабо совпадает со справочником МКБ",
    "unknown": "Оценка МКБ ещё не посчитана",
}


def diagnosis_text_from_case(case: dict[str, Any] | None) -> str:
    if not isinstance(case, dict):
        return ""
    try:
        from clinical_knowledge.mo_icd_resolve import resolve_diagnosis_text_from_mo

        return str(resolve_diagnosis_text_from_mo(case).get("text") or "").strip()
    except Exception:  # noqa: BLE001
        parts: list[str] = []
        for key in (
            "clinical_diagnosis",
            "mis_diagnos",
            "mis_diagnosis",
            "diagnosis_main_text",
            "diagnosis_short",
            "diagnosis_text",
        ):
            val = case.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        return " ".join(parts).strip()


def chip_label_ru(status: str) -> str:
    return CHIP_LABEL_RU.get(status, CHIP_LABEL_RU["unknown"])


def chip_title_ru(status: str) -> str:
    return CHIP_TITLE_RU.get(status, CHIP_TITLE_RU["unknown"])


def status_from_finding_codes(codes: list[str] | set[str] | str | None) -> str:
    """Свести сохранённые finding codes визита к одному статусу чипа."""
    if codes is None:
        return "unknown"
    if isinstance(codes, str):
        items = [c.strip() for c in codes.split(",") if c.strip()]
    else:
        items = [str(c).strip() for c in codes if str(c).strip()]
    if not items:
        return "unknown"
    best = "ok"
    best_rank = _STATUS_RANK["ok"]
    saw_icd = False
    for code in items:
        mapped = _CODE_TO_STATUS.get(code)
        if not mapped:
            continue
        saw_icd = True
        rank = _STATUS_RANK[mapped]
        if rank < best_rank:
            best = mapped
            best_rank = rank
    if not saw_icd:
        # есть другие findings, но не МКБ-ось - считаем ok по МКБ неизвестным
        # только если среди кодов не было наших; для списка дней лучше unknown
        return "unknown"
    return best


def status_payload(status: str, *, findings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "label_ru": chip_label_ru(status),
        "title_ru": chip_title_ru(status),
        "finding_codes": [
            str(f.get("code") or "")
            for f in (findings or [])
            if isinstance(f, dict) and str(f.get("code") or "") in _CODE_TO_STATUS
        ],
    }


def compute_icd_visit_status(
    case: dict[str, Any] | None,
    *,
    findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Посчитать статус визита: из findings или живой оценкой."""
    from clinical_knowledge.mo_icd_directory_eval import evaluate_diagnosis_against_icd_directory
    from clinical_knowledge.mo_icd_name_match import evaluate_diagnosis_name_only
    from clinical_knowledge.mo_icd_resolve import resolve_icd_codes_from_mo

    codes_from_findings: list[str] = []
    for item in findings or []:
        if isinstance(item, dict) and item.get("code"):
            codes_from_findings.append(str(item["code"]))
    if codes_from_findings:
        mapped = status_from_finding_codes(codes_from_findings)
        if mapped != "unknown":
            return status_payload(mapped, findings=list(findings or []))

    if not isinstance(case, dict):
        return status_payload("unknown")

    text = diagnosis_text_from_case(case)
    resolved = resolve_icd_codes_from_mo(case)
    code_list = list(resolved.get("all") or [])
    main = resolved.get("main")
    if main and main not in code_list:
        code_list.insert(0, str(main))

    dir_result = evaluate_diagnosis_against_icd_directory(text, code_list)
    name_result = evaluate_diagnosis_name_only(text) if text.strip() else {
        "findings": [],
        "verdict": "fail" if not code_list else "skip",
    }
    merged_findings = list(dir_result.get("findings") or []) + list(name_result.get("findings") or [])
    if not text.strip() and not code_list:
        # directory already emits B_dx_absent
        status = status_from_finding_codes([f.get("code") for f in merged_findings])
        if status == "unknown":
            status = "missing_dx"
        return status_payload(status, findings=merged_findings)

    status = status_from_finding_codes([f.get("code") for f in merged_findings])
    if status == "unknown":
        status = "ok"
    return status_payload(status, findings=merged_findings)
