"""Abnormal lab values ignored in МО (shadow, wave 3)."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from clinical_knowledge.lab_canons import lab_panels, text_hits_panel

ENGINE = "mo_lab_abnormal_v1"
_SOURCE = "mo_lab_abnormal_v1"
CODE_ABNORMAL_IGNORED = "B_lab_abnormal_ignored"
_ROOT = Path(__file__).resolve().parents[1]
_RANGE_PATH = _ROOT / "data" / "lab_canons" / "lab_reference_ranges.json"
_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_WS = re.compile(r"\s+")


def lab_abnormal_enabled() -> bool:
    raw = (os.environ.get("MO_LAB_ABNORMAL") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def lab_abnormal_primary_enabled() -> bool:
    raw = (os.environ.get("MO_LAB_ABNORMAL_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _norm(text: Any) -> str:
    return _WS.sub(" ", str(text or "").lower().replace("ё", "е")).strip()


@lru_cache(maxsize=1)
def load_reference_ranges() -> list[dict[str, Any]]:
    if not _RANGE_PATH.is_file():
        return []
    data = json.loads(_RANGE_PATH.read_text(encoding="utf-8"))
    return list(data.get("ranges") or [])


def _parse_number(value: Any) -> float | None:
    raw = str(value or "").strip().replace(",", ".")
    if not raw:
        return None
    match = _NUM.search(raw)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _unit_ok(actual: str, expected: str) -> bool:
    a = _norm(actual).replace(" ", "")
    e = _norm(expected).replace(" ", "")
    if not e:
        return True
    if not a:
        return True  # soft: missing unit still compared
    aliases = {
        "ед/л": {"е/л", "u/l", "ед/л"},
        "ме/л": {"ме/л", "мме/л", "iu/l"},
        "мг/л": {"мг/л", "mg/l"},
        "ммоль/л": {"ммоль/л", "mmol/l"},
        "мкмоль/л": {"мкмоль/л", "umol/l", "µмоль/л"},
        "г/л": {"г/л", "g/l"},
        "10^9/л": {"10^9/л", "10*9/л", "×10^9/л", "x10^9/л"},
    }
    for key, group in aliases.items():
        if e in group or e == key:
            return a in group or a == key or e in a or a in e
    return a == e or e in a or a in e


def _match_range(indicator_name: str, unit: str) -> dict[str, Any] | None:
    name = _norm(indicator_name)
    for row in load_reference_ranges():
        needles = [_norm(n) for n in (row.get("indicator_needles") or [])]
        if not any(n and n in name for n in needles):
            continue
        if not _unit_ok(unit, str(row.get("unit") or "")):
            continue
        return row
    return None


def abnormal_from_bundle(bundle: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(bundle, Mapping):
        return out
    for day in bundle.get("days") or []:
        if not isinstance(day, Mapping):
            continue
        test_date = str(day.get("test_date") or "")[:10]
        for item in day.get("types") or []:
            if not isinstance(item, Mapping):
                continue
            for ind in item.get("indicators") or []:
                if not isinstance(ind, Mapping):
                    continue
                name = str(ind.get("name") or "")
                unit = str(ind.get("unit") or "")
                value = _parse_number(ind.get("value"))
                if value is None:
                    continue
                ref = _match_range(name, unit)
                if not ref:
                    continue
                low = float(ref["low"])
                high = float(ref["high"])
                if low <= value <= high:
                    continue
                out.append(
                    {
                        "panel_id": ref.get("panel_id"),
                        "indicator": name,
                        "value": value,
                        "unit": unit or ref.get("unit"),
                        "low": low,
                        "high": high,
                        "test_date": test_date,
                    }
                )
    return out


def _mo_acknowledges_abnormal(case: Mapping[str, Any], item: Mapping[str, Any]) -> bool:
    blob = "\n".join(
        str(case.get(k) or "")
        for k in (
            "clinical_diagnosis",
            "diagnosis_main_text",
            "treatment_recommendations",
            "exam_data",
            "exam_recommendations",
        )
    )
    low = _norm(blob)
    if any(
        token in low
        for token in (
            "повышен", "снижен", "ниже нормы", "выше нормы", "отклон",
            "патолог", "аномал", "гипергликем", "анеми",
        )
    ):
        # If doctor mentioned abnormality broadly and the indicator/panel name appears
        panel = next(
            (p for p in lab_panels() if p["id"] == item.get("panel_id")),
            None,
        )
        if panel and text_hits_panel(blob, panel):
            return True
        if _norm(item.get("indicator")) and _norm(item.get("indicator")) in low:
            return True
    panel = next(
        (p for p in lab_panels() if p["id"] == item.get("panel_id")),
        None,
    )
    if panel and text_hits_panel(blob, panel):
        # mentioned but no abnormal language → still ignore if only listed
        return any(
            token in low
            for token in ("повышен", "снижен", "отклон", "патолог", "вне нормы")
        )
    return False


def abnormal_lab_findings(
    case: Mapping[str, Any] | None,
    bundle: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not lab_abnormal_enabled() or not isinstance(case, Mapping):
        return []
    items = abnormal_from_bundle(bundle)
    ignored = [item for item in items if not _mo_acknowledges_abnormal(case, item)]
    if not ignored:
        return []
    bits = ", ".join(
        f"{item.get('indicator')}={item.get('value')} "
        f"(норма {item.get('low')}–{item.get('high')})"
        for item in ignored[:5]
    )
    shadow = not lab_abnormal_primary_enabled()
    return [
        {
            "code": CODE_ABNORMAL_IGNORED,
            "axis": "concordance",
            "severity": "P1",
            "severity_label_ru": "Важно",
            "passed": False,
            "title_ru": "Отклонение анализа не отражено в заключении",
            "detail_ru": (
                f"Вне референса: {bits}."
                + (" Черновик, не входит в оценку." if shadow else "")
            ),
            "evidence": "",
            "source_ref": _SOURCE,
            "needs_human": True,
            "shadow": shadow,
            "is_shadow": shadow,
            "engine": ENGINE,
            "linked_fields": ["exam_data", "clinical_diagnosis"],
            "link_hint_ru": "Отразите отклонение анализа в диагнозе или плане",
        }
    ]
