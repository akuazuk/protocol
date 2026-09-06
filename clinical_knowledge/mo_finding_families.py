"""Реестр семейств findings (drug / lab) + KPI и подоси.

Канон: data/mo_finding_families/families_v1.json
План: docs/plans/2026-09-05-mo-meds-labs-dashboards-v1.md
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "data" / "mo_finding_families" / "families_v1.json"

_EXPLICIT_DRUG_LAB = {
    "C_ddi",
    "C_nsaid_dup",
    "C_ppi_dup",
    "C_antihistamine_dup",
    "C_anticoag_dup",
    "C_statin_dup",
    "C_ace_arb_dup",
    "C_high_alert_no_dose",
    "C_drug_unresolved",
    "C_formulary_unknown",
    "C_drug_disease_mismatch",
    "B_tx_offprotocol",
    "B_exams_gap",
}


@lru_cache(maxsize=1)
def load_families_catalog() -> dict[str, Any]:
    if not _PATH.is_file():
        return {"version": "", "families": [], "penalties": {}, "caps": {}, "overall_blend": {}}
    data = json.loads(_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def list_families() -> list[dict[str, Any]]:
    return [dict(row) for row in (load_families_catalog().get("families") or []) if isinstance(row, dict)]


def family_by_id(family_id: str) -> dict[str, Any] | None:
    want = str(family_id or "").strip().lower()
    for row in list_families():
        if str(row.get("id") or "").strip().lower() == want:
            return row
    return None


def codes_for_family(family_id: str) -> set[str]:
    row = family_by_id(family_id)
    if not row:
        return set()
    return {str(code).strip() for code in (row.get("codes") or []) if str(code).strip()}


def tile_codes(family_id: str, tile_id: str) -> set[str]:
    row = family_by_id(family_id)
    if not row:
        return set()
    for tile in row.get("tiles") or []:
        if not isinstance(tile, Mapping):
            continue
        if str(tile.get("id") or "") != tile_id:
            continue
        codes = {str(c).strip() for c in (tile.get("codes") or []) if str(c).strip()}
        return codes or codes_for_family(family_id)
    return set()


def code_to_family() -> dict[str, str]:
    out: dict[str, str] = {}
    for row in list_families():
        fid = str(row.get("id") or "").strip()
        for code in row.get("codes") or []:
            key = str(code).strip()
            if key:
                out[key] = fid
    return out


def family_for_code(code: str) -> str:
    return code_to_family().get(str(code or "").strip(), "other")


def required_drug_lab_codes() -> set[str]:
    """Коды из labels, которые обязаны быть в реестре (нет сирот)."""
    from .mo_finding_labels_ru import FINDING_TITLE_RU

    out = set(_EXPLICIT_DRUG_LAB)
    for code in FINDING_TITLE_RU:
        if code.startswith("B_lab_") or code.startswith("C_rceth_"):
            out.add(code)
        if code in _EXPLICIT_DRUG_LAB:
            out.add(code)
    return out


def orphan_drug_lab_codes() -> list[str]:
    assigned = set(code_to_family())
    return sorted(required_drug_lab_codes() - assigned)


def duplicate_family_codes() -> list[str]:
    seen: dict[str, str] = {}
    dups: list[str] = []
    for row in list_families():
        fid = str(row.get("id") or "")
        for code in row.get("codes") or []:
            key = str(code).strip()
            if not key:
                continue
            if key in seen and seen[key] != fid:
                dups.append(key)
            seen[key] = fid
    return sorted(set(dups))


def _env_on(name: str) -> bool:
    return (os.environ.get(name) or "0").strip().lower() in {"1", "true", "yes", "on"}


def family_scores_in_overall_enabled() -> bool:
    return _env_on("MO_FAMILY_SCORES_IN_OVERALL")


def _lab_family_primary() -> bool:
    return _env_on("MO_LAB_UNUSED_PRIMARY") or _env_on("MO_LAB_ABNORMAL_PRIMARY")


def _drug_family_primary() -> bool:
    return _env_on("MO_CLASS_DUP_PRIMARY") or _env_on("MO_RCETH_LABEL_PRIMARY")


def _pct(n: int, total: int | None) -> float | None:
    if total is None or int(total) <= 0:
        return None
    return round(100.0 * n / int(total), 1)


def _code_title(code: str) -> str:
    try:
        from .mo_finding_labels_ru import FINDING_TITLE_RU

        return str(FINDING_TITLE_RU.get(code) or code)
    except Exception:  # noqa: BLE001
        return code


def _finding_code(item: Mapping[str, Any]) -> str:
    return str(item.get("code") or item.get("finding_code") or "").strip()


def _finding_passed(item: Mapping[str, Any]) -> bool:
    if item.get("passed") is True:
        return True
    return False


def _finding_shadow(item: Mapping[str, Any]) -> bool:
    return bool(item.get("is_shadow") or item.get("shadow"))


def family_scores_from_findings(
    findings: Iterable[Mapping[str, Any]] | None,
    *,
    shadow_findings: Iterable[Mapping[str, Any]] | None = None,
    completed_families: Iterable[str] = (),
) -> dict[str, Any]:
    """Подоси 0-100: 100 − штрафы; P0 ≤40, P1 ≤60 внутри семейства."""
    catalog = load_families_catalog()
    penalties = {str(k): int(v) for k, v in (catalog.get("penalties") or {}).items()}
    caps = {str(k): float(v) for k, v in (catalog.get("caps") or {}).items()}
    completed = set(completed_families)
    items: list[Mapping[str, Any]] = []
    for is_shadow_source, src in ((False, findings), (True, shadow_findings)):
        for item in src or []:
            if isinstance(item, Mapping):
                items.append({**item, "shadow": True} if is_shadow_source else item)

    def _score(family_id: str, *, primary_only: bool) -> dict[str, Any]:
        codes = codes_for_family(family_id)
        used: list[Mapping[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for item in items:
            code = _finding_code(item)
            if code not in codes or _finding_passed(item):
                continue
            if primary_only and _finding_shadow(item):
                continue
            fingerprint = str(item.get("fingerprint") or item.get("finding_id") or "")
            identity = (code, fingerprint) if fingerprint else tuple(
                str(item.get(key) or "")
                for key in ("source_ref", "rule_id", "target_id", "evidence", "detail_ru")
            ) + (code,)
            if identity in seen:
                continue
            seen.add(identity)
            used.append(item)
        penalty = 0
        worst = ""
        for item in used:
            sev = str(item.get("severity") or "P3").upper()
            penalty += int(penalties.get(sev, 5))
            if sev == "P0":
                worst = "P0"
            elif sev == "P1" and worst != "P0":
                worst = "P1"
        score = max(0.0, 100.0 - penalty)
        if worst == "P0":
            score = min(score, float(caps.get("P0", 40)))
        elif worst == "P1":
            score = min(score, float(caps.get("P1", 60)))
        return {
            "score": round(score, 1) if used or family_id in completed else None,
            "status": "completed" if family_id in completed else ("partial" if used else "not_evaluated"),
            "n_findings": len(used),
            "worst_severity": worst or None,
            "codes": sorted({_finding_code(x) for x in used}),
        }

    drug_all = _score("drug", primary_only=False)
    lab_all = _score("lab", primary_only=False)
    drug_pri = _score("drug", primary_only=True)
    lab_pri = _score("lab", primary_only=True)
    in_overall = family_scores_in_overall_enabled()
    return {
        "drug_score": drug_all["score"],
        "lab_score": lab_all["score"],
        "drug_score_primary": drug_pri["score"],
        "lab_score_primary": lab_pri["score"],
        "drug": drug_all,
        "lab": lab_all,
        "shadow": True,
        "in_overall": bool(in_overall),
        "note_ru": (
            "Подоси лекарств и анализов в общей оценке"
            if in_overall
            else "черновик, не в общей оценке"
        ),
    }


def maybe_blend_family_into_axes(
    axes: Mapping[str, Any] | None,
    family_scores: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """D4: min(ось, подось) только при флаге overall + primary семейства."""
    out = dict(axes or {})
    meta: dict[str, Any] = {
        "applied": False,
        "rule": "min(axis, family_score) when flag+primary",
        "changed": [],
    }
    if not family_scores_in_overall_enabled():
        return out, meta
    scores = family_scores or {}
    if _lab_family_primary() and out.get("clinical_concordance") is not None:
        lab = scores.get("lab_score_primary")
        if lab is not None:
            prev = float(out["clinical_concordance"])
            out["clinical_concordance"] = round(min(prev, float(lab)), 1)
            meta["changed"].append("clinical_concordance")
            meta["applied"] = True
    if _drug_family_primary() and out.get("safety") is not None:
        drug = scores.get("drug_score_primary")
        if drug is not None:
            prev = float(out["safety"])
            out["safety"] = round(min(prev, float(drug)), 1)
            meta["changed"].append("safety")
            meta["applied"] = True
    return out, meta


def family_dashboard_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    total_cases: int,
    cases_with_lab: int | None = None,
    lab_coverage_available: bool = False,
) -> dict[str, Any]:
    """Агрегат для разделов Лекарства / Анализы (без склада)."""
    material = [r for r in rows if isinstance(r, Mapping)]
    by_code: Counter[str] = Counter()
    cases_by_code: dict[str, set[Any]] = defaultdict(set)
    cases_by_family: dict[str, set[Any]] = defaultdict(set)
    cases_by_tile: dict[str, set[Any]] = defaultdict(set)
    spec_cases: dict[str, dict[str, set[Any]]] = defaultdict(lambda: defaultdict(set))
    doc_cases: dict[str, dict[str, set[Any]]] = defaultdict(lambda: defaultdict(set))
    doc_spec: dict[str, str] = {}
    day_cases: dict[str, dict[str, set[Any]]] = defaultdict(lambda: defaultdict(set))

    for row in material:
        code = str(row.get("finding_code") or row.get("code") or "").strip()
        if not code:
            continue
        mid = row.get("mis_id")
        by_code[code] += int(row.get("cases") or 1)
        if mid is None:
            continue
        cases_by_code[code].add(mid)
        fam = family_for_code(code)
        if fam != "other":
            cases_by_family[fam].add(mid)
        spec = str(row.get("specialty") or "").strip()
        doctor = str(row.get("doctor") or row.get("doctor_name") or "").strip()
        day = str(row.get("visit_date") or row.get("date") or "")[:10]
        if spec and fam != "other":
            spec_cases[fam][spec].add(mid)
        if doctor and fam != "other":
            doc_cases[fam][doctor].add(mid)
            if spec:
                doc_spec[doctor] = spec
        if day and fam != "other":
            day_cases[fam][day].add(mid)

    families_out: dict[str, Any] = {}
    for fam in list_families():
        fid = str(fam.get("id") or "")
        fam_codes = codes_for_family(fid)
        fam_n = len(cases_by_family[fid])
        tiles = []
        for tile in fam.get("tiles") or []:
            if not isinstance(tile, Mapping):
                continue
            tid = str(tile.get("id") or "")
            t_codes = {str(c).strip() for c in (tile.get("codes") or []) if str(c).strip()} or fam_codes
            hits: set[Any] = set()
            for code in t_codes:
                hits |= cases_by_code.get(code, set())
            cases_by_tile[f"{fid}:{tid}"] = hits
            den_key = str(tile.get("denominator") or "")
            uses_lab = den_key == "cases_with_lab" and lab_coverage_available and cases_with_lab is not None
            denom = cases_with_lab if uses_lab else total_cases
            actual_den_key = "cases_with_lab" if uses_lab else "total_cases"
            tiles.append(
                {
                    "id": tid,
                    "title_ru": tile.get("title_ru"),
                    "codes": sorted(t_codes),
                    "cases": len(hits),
                    "pct": _pct(len(hits), denom),
                    "denominator": actual_den_key,
                    "denominator_n": denom,
                }
            )
        by_code_rows = []
        for code, count in by_code.most_common(40):
            if code not in fam_codes:
                continue
            n_cases = len(cases_by_code[code])
            by_code_rows.append(
                {
                    "code": code,
                    "title_ru": _code_title(code),
                    "findings": count,
                    "cases": n_cases,
                    "pct": _pct(n_cases, total_cases),
                }
            )
        by_specialty = [
            {
                "specialty": spec,
                "cases": len(mids),
                "pct": _pct(len(mids), total_cases),
            }
            for spec, mids in sorted(spec_cases[fid].items(), key=lambda kv: -len(kv[1]))[:20]
        ]
        by_doctor = [
            {
                "doctor": doctor,
                "specialty": doc_spec.get(doctor, ""),
                "cases": len(mids),
                "pct": _pct(len(mids), total_cases),
            }
            for doctor, mids in sorted(doc_cases[fid].items(), key=lambda kv: -len(kv[1]))[:30]
        ]
        by_day = [
            {"date": day, "cases": len(mids), "pct": _pct(len(mids), total_cases)}
            for day, mids in sorted(day_cases[fid].items())[-14:]
        ]
        families_out[fid] = {
            "id": fid,
            "title_ru": fam.get("title_ru"),
            "score_key": fam.get("score_key"),
            "feeds_overall_axis": fam.get("feeds_overall_axis"),
            "cases": fam_n,
            "pct": _pct(fam_n, total_cases),
            "tiles": tiles,
            "by_code": by_code_rows,
            "by_specialty": by_specialty,
            "by_doctor": by_doctor,
            "by_day": by_day,
        }

    unused_codes = tile_codes("lab", "unused")
    unused_n = len(set().union(*(cases_by_code.get(c, set()) for c in unused_codes)))
    unused_among_lab = _pct(unused_n, cases_with_lab) if lab_coverage_available else None
    if "lab" in families_out:
        families_out["lab"]["unused_among_lab_pct"] = unused_among_lab
        families_out["lab"]["unused_among_lab_cases"] = unused_n

    top_drug = (families_out.get("drug") or {}).get("by_code") or []
    top_lab = (families_out.get("lab") or {}).get("by_code") or []
    return {
        "families": families_out,
        "denominators": {
            "total_cases": int(total_cases or 0),
            "cases_with_lab": cases_with_lab if lab_coverage_available else None,
            "lab_coverage_available": bool(lab_coverage_available),
            "unused_lab_pct_among_lab": unused_among_lab,
        },
        "strips": {
            "drug": {
                "pct": (families_out.get("drug") or {}).get("pct"),
                "cases": (families_out.get("drug") or {}).get("cases"),
                "top_code": (top_drug[0]["code"] if top_drug else None),
                "top_title_ru": (top_drug[0]["title_ru"] if top_drug else None),
            },
            "lab": {
                "pct": unused_among_lab
                if lab_coverage_available
                else (families_out.get("lab") or {}).get("pct"),
                "cases": unused_n if lab_coverage_available else (families_out.get("lab") or {}).get("cases"),
                "top_code": (top_lab[0]["code"] if top_lab else None),
                "top_title_ru": (top_lab[0]["title_ru"] if top_lab else None),
                "denominator": "cases_with_lab" if lab_coverage_available else "total_cases",
            },
        },
        "shadow_note_ru": "черновик, не в общей оценке",
    }
