"""Shadow-сверка рекомендаций по обследованию с type_name склада лаборатории.

План: docs/plans/2026-08-26-mo-lab-from-mis-tests-v1.md волна 2.
Канон: finding только shadow; «нет на складе» не штраф (покрытие окна ~22%).
Значения анализов в finding не кладём.
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping

from clinical_knowledge.mo_lab_bundle import (
    lab_payload_for_case,
    lab_primary_enabled,
    lab_reconcile_payload_for_case,
)

ENGINE = "mo_lab_shadow_v1"
_SOURCE = "mo_lab_shadow_v1"
CODE_ORDERED_DONE = "B_lab_ordered_already_done"
CODE_PRESENT_GAP = "B_lab_present_not_in_mo"
LAB_SHADOW_CODES = {
    CODE_ORDERED_DONE,
    CODE_PRESENT_GAP,
    "B_lab_unused_in_dx",
    "B_lab_unused_in_plan",
    "B_lab_ordered_not_used",
    "B_lab_abnormal_ignored",
}

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-zа-я0-9]+", re.I)


def lab_shadow_enabled() -> bool:
    raw = (os.environ.get("MO_LAB_SHADOW") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _norm(text: Any) -> str:
    raw = str(text or "").lower().replace("ё", "е")
    return _WS.sub(" ", raw).strip()


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text or ""))


class _Panel:
    __slots__ = ("pid", "label", "type_needles", "text_re")

    def __init__(self, pid: str, label: str, type_needles: tuple[str, ...], text_re: str):
        self.pid = pid
        self.label = label
        self.type_needles = tuple(_norm(n) for n in type_needles if n)
        self.text_re = re.compile(text_re, re.I)


# Короткие иглы (оак/оам/бак/ттг/срб/пса) - только целый токен, не подстрока.
PANELS: tuple[_Panel, ...] = (
    _Panel(
        "oak",
        "ОАК",
        ("оак", "общий анализ крови", "клинический анализ крови", "гемограмма",
         "развернутый анализ крови", "общий клинический анализ крови"),
        r"\bоак\b|общ\w*\s+анализ\w*\s+кров|клиническ\w*\s+анализ\w*\s+кров|гемограмм",
    ),
    _Panel(
        "oam",
        "ОАМ",
        ("оам", "общий анализ мочи", "клинический анализ мочи"),
        r"\bоам\b|общ\w*\s+анализ\w*\s+моч|клиническ\w*\s+анализ\w*\s+моч",
    ),
    _Panel(
        "bak",
        "БАК",
        ("бак", "биохимический анализ крови", "биохимия крови",
         "биохимическое исследование крови"),
        r"\bбак\b(?!\s*посев)|биохим\w*\s+анализ\w*\s+кров|биохими[яи]\s+кров",
    ),
    _Panel(
        "glucose",
        "глюкоза",
        ("глюкоза", "сахар крови", "глюкоза крови"),
        r"\bглюкоз|сахар\w*\s+кров",
    ),
    _Panel(
        "hba1c",
        "HbA1c",
        ("hba1c", "hba1", "гликированный", "гликированный гемоглобин"),
        r"hba1|гликированн",
    ),
    _Panel(
        "lipid",
        "липидограмма",
        ("липидограмм", "липидный спектр", "холестерин"),
        r"липидограмм|липидн\w*\s+спектр|\bхолестерин",
    ),
    _Panel(
        "coag",
        "коагулограмма",
        ("коагулограмм", "протромбин", "фибриноген", "ачтв"),
        r"коагулограмм|\bмно\b|протромбин|ачтв",
    ),
    _Panel(
        "tsh",
        "ТТГ",
        ("ттг", "тиреотроп"),
        r"\bттг\b|тиреотроп",
    ),
    _Panel(
        "crp",
        "СРБ",
        ("срб", "c-реактивный", "c реактивный", "crp"),
        r"\bсрб\b|c[\-\s]?реактив|\bcrp\b",
    ),
    _Panel(
        "psa",
        "ПСА",
        ("пса", "простатспециф", "psa"),
        r"\bпса\b|простатспециф",
    ),
)

# Волна 1: дополнить PANELS из JSON-канонов (≥15), не ломая legacy ids.
def _extend_panels_from_canons(base: tuple[_Panel, ...]) -> tuple[_Panel, ...]:
    try:
        from clinical_knowledge.lab_canons import lab_panels as _load_canon_panels
    except Exception:  # noqa: BLE001
        return base
    known = {p.pid for p in base}
    extra: list[_Panel] = []
    for row in _load_canon_panels():
        pid = str(row.get("id") or "")
        if not pid or pid in known:
            continue
        needles = tuple(row.get("type_needles") or ())
        patterns = row.get("text_patterns") or ()
        parts = [p.pattern for p in patterns if getattr(p, "pattern", None)]
        text_re = "|".join(parts) if parts else r"a^"
        extra.append(
            _Panel(pid, str(row.get("label") or pid), needles, text_re)
        )
        known.add(pid)
    return base + tuple(extra) if extra else base


PANELS = _extend_panels_from_canons(PANELS)


def _type_hits(type_name: str, panel: _Panel) -> bool:
    n = _norm(type_name)
    if not n:
        return False
    tokens = _tokens(n)
    for needle in panel.type_needles:
        if not needle:
            continue
        if " " not in needle and len(needle) <= 4:
            if needle in tokens:
                return True
            continue
        if needle in n:
            return True
    return False


def _text_hits(text: str, panel: _Panel) -> bool:
    n = _norm(text)
    if not n:
        return False
    return bool(panel.text_re.search(n))


def present_panels(
    bundle: Mapping[str, Any] | None,
    *,
    max_date: str = "",
) -> dict[str, dict[str, Any]]:
    """id панели → label + ближайшая дата на складе."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(bundle, Mapping):
        return out
    for day in bundle.get("days") or []:
        if not isinstance(day, Mapping):
            continue
        test_date = str(day.get("test_date") or "")[:10]
        if max_date and test_date > max_date:
            continue
        for item in day.get("types") or []:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("type_name") or "")
            indicator_names = [
                str(indicator.get("name") or "")
                for indicator in item.get("indicators") or []
                if isinstance(indicator, Mapping)
            ]
            for panel in PANELS:
                if panel.pid in out:
                    continue
                if _type_hits(name, panel) or any(
                    _type_hits(indicator_name, panel)
                    for indicator_name in indicator_names
                ):
                    out[panel.pid] = {
                        "id": panel.pid,
                        "label": panel.label,
                        "test_date": test_date,
                        "source_type_name": name,
                    }
    return out


def post_visit_panels(
    bundle: Mapping[str, Any] | None,
    *,
    visit_date: str,
) -> dict[str, dict[str, Any]]:
    """Панели после визита: UI context, но никогда не documentation gap."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(bundle, Mapping) or not visit_date:
        return out
    for day in bundle.get("days") or []:
        if not isinstance(day, Mapping):
            continue
        test_date = str(day.get("test_date") or "")[:10]
        if not test_date or test_date <= visit_date:
            continue
        for item in day.get("types") or []:
            if not isinstance(item, Mapping):
                continue
            candidates = [str(item.get("type_name") or "")]
            candidates.extend(
                str(indicator.get("name") or "")
                for indicator in item.get("indicators") or []
                if isinstance(indicator, Mapping)
            )
            for panel in PANELS:
                if panel.pid not in out and any(_type_hits(value, panel) for value in candidates):
                    out[panel.pid] = {
                        "id": panel.pid,
                        "label": panel.label,
                        "test_date": test_date,
                    }
    return out


def ordered_panels(recs_text: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    blob = _norm(recs_text)
    if not blob:
        return out
    for panel in PANELS:
        if _text_hits(blob, panel):
            out[panel.pid] = {"id": panel.pid, "label": panel.label}
    return out


def mentioned_in_mo(exam_data: str, recs_text: str, panel: _Panel) -> bool:
    return _text_hits(exam_data, panel) or _text_hits(recs_text, panel)


def source_type_mentioned_in_mo(
    exam_data: str,
    recs_text: str,
    source_type_name: str,
) -> bool:
    """Не считать indicator gap, если в МО указана его родительская панель."""
    return any(
        _type_hits(source_type_name, parent)
        and mentioned_in_mo(exam_data, recs_text, parent)
        for parent in PANELS
    )


def build_lab_reconcile(
    bundle: Mapping[str, Any] | None,
    case: Mapping[str, Any] | None,
) -> dict[str, Any]:
    recs_raw = None if not isinstance(case, Mapping) else case.get("exam_recommendations")
    exam_raw = None if not isinstance(case, Mapping) else case.get("exam_data")
    recs = "" if recs_raw is None else str(recs_raw)
    exam = "" if exam_raw is None else str(exam_raw)
    texts_loaded = recs_raw is not None or exam_raw is not None
    window = bundle.get("window") if isinstance(bundle, Mapping) else {}
    visit_date = str((window or {}).get("visit_date") or "")[:10]
    present = present_panels(bundle, max_date=visit_date)
    post_visit = post_visit_panels(bundle, visit_date=visit_date)
    ordered = ordered_panels(recs)
    ordered_and_present = [
        {"label": present[pid]["label"], "test_date": present[pid]["test_date"]}
        for pid in ordered
        if pid in present
    ]
    ordered_not_in_warehouse = [
        {"label": ordered[pid]["label"]}
        for pid in ordered
        if pid not in present
    ]
    present_not_in_mo: list[dict[str, Any]] = []
    if texts_loaded:
        for pid, row in present.items():
            panel = next((p for p in PANELS if p.pid == pid), None)
            if (
                panel
                and not mentioned_in_mo(exam, recs, panel)
                and not source_type_mentioned_in_mo(
                    exam,
                    recs,
                    str(row.get("source_type_name") or ""),
                )
            ):
                present_not_in_mo.append(
                    {
                        "label": row["label"],
                        "test_date": row["test_date"],
                        "same_day": row["test_date"] == visit_date,
                    }
                )
    return {
        "engine": ENGINE,
        "texts_loaded": texts_loaded,
        "ordered": [{"label": ordered[pid]["label"]} for pid in ordered],
        "present": [
            {"label": row["label"], "test_date": row["test_date"]}
            for row in present.values()
        ],
        "ordered_and_present": ordered_and_present[:8],
        "ordered_not_in_warehouse": ordered_not_in_warehouse[:8],
        "present_not_in_mo": present_not_in_mo[:8],
        "primary_present_not_in_mo": [
            row for row in present_not_in_mo if row.get("same_day")
        ][:8],
        "post_visit_present": [
            {"label": row["label"], "test_date": row["test_date"]}
            for row in post_visit.values()
        ][:8],
        "note_ru": (
            "Сверка рекомендаций с type_name склада. Черновик, не входит в оценку. "
            "«Нет на складе» не штраф: окно покрывает не все визиты. "
            "Результаты после визита не создают замечаний."
        ),
    }


def _finding(code: str, *, title: str, detail: str) -> dict[str, Any]:
    return {
        "code": code,
        "axis": "plan",
        "severity": "P3",
        "severity_label_ru": "Оформление",
        "severity_tone": "formal",
        "passed": False,
        "title_ru": title,
        "detail_ru": detail,
        "evidence": "",
        "source_ref": _SOURCE,
        "needs_human": False,
        "shadow": True,
        "is_shadow": True,
        "engine": ENGINE,
        "linked_fields": ["exam_recommendations", "exam_data"],
        "link_hint_ru": "Сверьте план обследования с лабораторией на складе",
    }


def lab_shadow_findings(reconcile: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not lab_shadow_enabled() or not isinstance(reconcile, Mapping):
        return []
    out: list[dict[str, Any]] = []
    done = list(reconcile.get("ordered_and_present") or [])
    if done:
        bits = ", ".join(
            f"{item.get('label')} ({item.get('test_date')})"
            if item.get("test_date")
            else str(item.get("label") or "")
            for item in done[:6]
        )
        out.append(
            _finding(
                CODE_ORDERED_DONE,
                title="Лаборатория: назначенное обследование уже есть на складе",
                detail=(
                    f"В рекомендациях и на складе совпали: {bits}. "
                    "Повторное назначение может быть лишним. Черновик, не входит в оценку."
                ),
            )
        )
    gap = list(reconcile.get("present_not_in_mo") or [])
    primary_gap = list(reconcile.get("primary_present_not_in_mo") or [])
    gap_for_finding = (
        primary_gap
        if lab_primary_enabled() and primary_gap
        else gap
    )
    if gap:
        bits = ", ".join(
            f"{item.get('label')} ({item.get('test_date')})"
            if item.get("test_date")
            else str(item.get("label") or "")
            for item in gap_for_finding[:6]
        )
        out.append(
            _finding(
                CODE_PRESENT_GAP,
                title="Лаборатория: анализы есть, в МО не указаны",
                detail=(
                    f"На складе за окно визита есть {bits}, "
                    "в рекомендациях и данных обследований не найдены. "
                    "Черновик, не входит в оценку."
                ),
            )
        )
    if lab_primary_enabled():
        promoted = []
        for item in out:
            row = dict(item)
            if str(row.get("code")) == CODE_PRESENT_GAP and primary_gap:
                row["shadow"] = False
                row["is_shadow"] = False
                row["detail_ru"] = str(row.get("detail_ru") or "").replace(
                    " Черновик, не входит в оценку.", ""
                )
            promoted.append(row)
        return promoted
    return out


def merge_lab_shadow_into_findings(
    findings: list[dict[str, Any]] | None,
    extra: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    base = [
        dict(item)
        for item in (findings or [])
        if isinstance(item, Mapping)
        and str(item.get("code") or item.get("finding_code") or "") not in LAB_SHADOW_CODES
    ]
    for item in extra or []:
        if isinstance(item, Mapping) and item.get("code"):
            base.append(dict(item))
    return base


def evaluate_lab_for_case(
    case: dict[str, Any],
    *,
    lab_db: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Один путь для live и batch: display payload + полная reconcile-оценка."""
    payload = lab_payload_for_case(case, lab_db=lab_db)
    reconcile_source = lab_reconcile_payload_for_case(case, lab_db=lab_db)
    recon = build_lab_reconcile(reconcile_source, case)
    payload["reconcile"] = {
        "engine": ENGINE,
        "ordered": list(recon.get("ordered") or []),
        "present": list(recon.get("present") or []),
        "ordered_and_present": list(recon.get("ordered_and_present") or []),
        "ordered_not_in_warehouse": list(recon.get("ordered_not_in_warehouse") or []),
        "present_not_in_mo": list(recon.get("present_not_in_mo") or []),
        "primary_present_not_in_mo": list(
            recon.get("primary_present_not_in_mo") or []
        ),
        "post_visit_present": list(recon.get("post_visit_present") or []),
        "note_ru": recon.get("note_ru") or "",
    }
    findings = lab_shadow_findings(recon)
    try:
        from clinical_knowledge.lab_unused_findings import (
            merge_unused_into_findings,
            unused_lab_findings,
        )

        unused = unused_lab_findings(case, recon)
        findings = merge_unused_into_findings(findings, unused)
    except Exception:  # noqa: BLE001
        pass
    try:
        from clinical_knowledge.lab_abnormal_findings import abnormal_lab_findings

        findings = list(findings) + list(
            abnormal_lab_findings(case, reconcile_source)
        )
    except Exception:  # noqa: BLE001
        pass
    return payload, findings


def apply_lab_to_result(
    result: dict[str, Any],
    case: dict[str, Any],
    *,
    lab_db: Any = None,
) -> dict[str, Any]:
    """Положить result['lab'] и lab-findings. Primary только при MO_LAB_IN_PRIMARY=1
    и только для «есть на складе, в МО не указаны»."""
    payload, extra = evaluate_lab_for_case(case, lab_db=lab_db)
    result["lab"] = payload
    result["findings"] = merge_lab_shadow_into_findings(
        result.get("findings") if isinstance(result.get("findings"), list) else [],
        extra,
    )
    return result
