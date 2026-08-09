"""Conservative shadow Dx/Plan scores (option B) - not production SSOT."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

ENGINE = "mo_shadow_dx_plan_v1"
SCHEMA_VERSION = 1
UI_DISCLAIMER_RU = "Клиническая калибровка (shadow) - не официальная оценка"

POOR_MAX_SCORE = 45.0
CRITICAL_MAX_SCORE = 30.0
PLAN_ENSEMBLE_DOWNGRADE_MIN = 60.0


def _clip(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _pct(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return max(0.0, min(100.0, parsed))


def _score_from_endpoint(endpoint: str, payload: Mapping[str, Any] | None) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    if endpoint == "dx":
        return _pct(payload.get("dx_evidence_pct"))
    score = _pct(payload.get("plan_protocol_pct"))
    if score is None:
        score = _pct(payload.get("plan_general_llm_pct"))
    return score


def attention_band_for_endpoint(
    *,
    endpoint: str,
    verdict: str,
    score_pct: float | None,
    potential_harm: bool = False,
    plan_ensemble_pct: float | None = None,
) -> dict[str, Any]:
    """Map raw LLM verdict to conservative UI attention band."""
    verdict_norm = str(verdict or "").strip().lower()
    base = {
        "endpoint": endpoint,
        "verdict": verdict_norm or None,
        "score_pct": score_pct,
        "potential_harm": bool(potential_harm),
        "softened": False,
        "soften_reason": "",
        "plan_ensemble_pct": plan_ensemble_pct if endpoint == "plan" else None,
    }
    if verdict_norm in {"", "blocked", "na", "good", "partial"}:
        return {**base, "band": "none"}
    if verdict_norm not in {"poor", "critical"}:
        return {**base, "band": "none", "soften_reason": "unknown_verdict"}

    band = "critical" if verdict_norm == "critical" else "poor"
    if band == "poor" and potential_harm:
        band = "critical"

    if score_pct is None:
        return {
            **base,
            "band": "none",
            "softened": True,
            "soften_reason": "missing_score",
        }
    if band == "poor" and score_pct > POOR_MAX_SCORE:
        return {
            **base,
            "band": "none",
            "softened": True,
            "soften_reason": "poor_score_above_45",
        }
    if band == "critical":
        allow_critical = score_pct <= CRITICAL_MAX_SCORE or (
            potential_harm and score_pct <= POOR_MAX_SCORE
        )
        if not allow_critical:
            if score_pct <= POOR_MAX_SCORE:
                return {
                    **base,
                    "band": "poor",
                    "softened": True,
                    "soften_reason": "critical_score_softened_to_poor",
                }
            return {
                **base,
                "band": "none",
                "softened": True,
                "soften_reason": "critical_score_above_threshold",
            }

    if (
        endpoint == "plan"
        and band == "poor"
        and not potential_harm
        and plan_ensemble_pct is not None
        and plan_ensemble_pct >= PLAN_ENSEMBLE_DOWNGRADE_MIN
    ):
        return {
            **base,
            "band": "none",
            "softened": True,
            "soften_reason": "plan_ensemble_downgrade",
        }
    return {**base, "band": band}


def build_shadow_payload(
    *,
    case_id: str,
    visit_date: str,
    model: str,
    dx_result: Mapping[str, Any] | None,
    plan_result: Mapping[str, Any] | None,
    clinical_concordance_pct: float | None,
    error: str | None = None,
) -> dict[str, Any]:
    dx_score = _score_from_endpoint("dx", dx_result)
    plan_score = _score_from_endpoint("plan", plan_result)
    ensemble = None
    if plan_score is not None and clinical_concordance_pct is not None:
        ensemble = round((plan_score + float(clinical_concordance_pct)) / 2.0, 2)

    dx_attention = attention_band_for_endpoint(
        endpoint="dx",
        verdict=str((dx_result or {}).get("verdict") or ""),
        score_pct=dx_score,
        potential_harm=bool((dx_result or {}).get("potential_harm")),
    )
    plan_attention = attention_band_for_endpoint(
        endpoint="plan",
        verdict=str((plan_result or {}).get("verdict") or ""),
        score_pct=plan_score,
        potential_harm=bool((plan_result or {}).get("potential_harm")),
        plan_ensemble_pct=ensemble,
    )
    case_band = "none"
    if "critical" in {dx_attention["band"], plan_attention["band"]}:
        case_band = "critical"
    elif "poor" in {dx_attention["band"], plan_attention["band"]}:
        case_band = "poor"

    return {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "shadow": True,
        "official_score": False,
        "case_id": str(case_id or "").strip(),
        "visit_date": str(visit_date or "").strip()[:10],
        "model": model,
        "error": _clip(error, 400) if error else None,
        "dx": {
            "verdict": dx_attention["verdict"],
            "score_pct": dx_score,
            "potential_harm": dx_attention["potential_harm"],
            "summary_ru": _clip((dx_result or {}).get("summary_ru"), 200),
            "icd_fit": str((dx_result or {}).get("icd_fit") or "") or None,
            "attention": dx_attention,
        },
        "plan": {
            "verdict": plan_attention["verdict"],
            "score_pct": plan_score,
            "ensemble_pct": ensemble,
            "clinical_concordance_pct": clinical_concordance_pct,
            "potential_harm": plan_attention["potential_harm"],
            "summary_ru": _clip((plan_result or {}).get("summary_ru"), 200),
            "provenance": str((plan_result or {}).get("provenance") or "") or None,
            "attention": plan_attention,
        },
        "case_attention_band": case_band,
        "disclaimer_ru": UI_DISCLAIMER_RU,
    }


def summarize_shadow_for_ui(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "available": False,
            "shadow": True,
            "engine": ENGINE,
            "disclaimer_ru": UI_DISCLAIMER_RU,
            "reason": "Shadow Dx/Plan ещё не посчитан для этого случая",
        }
    if row.get("error") and not row.get("dx") and not row.get("plan"):
        return {
            "available": False,
            "shadow": True,
            "engine": ENGINE,
            "disclaimer_ru": UI_DISCLAIMER_RU,
            "reason": _clip(row.get("error"), 240),
        }
    return {
        "available": True,
        "shadow": True,
        "official_score": False,
        "engine": ENGINE,
        "disclaimer_ru": UI_DISCLAIMER_RU,
        "model": row.get("model"),
        "case_attention_band": row.get("case_attention_band") or "none",
        "dx": row.get("dx") or {},
        "plan": row.get("plan") or {},
        "softened": bool(
            ((row.get("dx") or {}).get("attention") or {}).get("softened")
            or ((row.get("plan") or {}).get("attention") or {}).get("softened")
        ),
    }


def _data_roots() -> list[Path]:
    roots: list[Path] = []
    configured = (os.environ.get("MO_DATA_ROOT") or "").strip()
    if configured:
        roots.append(Path(configured))
    roots.append(Path(__file__).resolve().parents[1] / "data" / "medical_exams")
    var = Path("/var/data/medical_exams")
    if var.is_dir():
        roots.append(var)
    return roots


def shadow_jsonl_path(day: str, *, root: Path | None = None) -> Path:
    y, m, d = str(day)[:10].split("-")
    base = root or _data_roots()[0]
    return base / "llm_shadow_dx_plan" / y / m / d / "shadow.jsonl"


def load_shadow_row(
    case_id: str,
    *,
    visit_date: str,
    roots: list[Path] | None = None,
) -> dict[str, Any] | None:
    cid = str(case_id or "").strip()
    day = str(visit_date or "").strip()[:10]
    if not cid or len(day) != 10:
        return None
    for root in roots or _data_roots():
        path = shadow_jsonl_path(day, root=root)
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    keys = {
                        str(row.get("case_id") or "").strip(),
                        str(row.get("visit_id") or "").strip(),
                        str(row.get("mis_id") or "").strip(),
                    }
                    if cid in keys:
                        return row
        except (OSError, json.JSONDecodeError):
            continue
    return None


def load_shadow_for_case(
    case_id: str,
    *,
    visit_date: str,
    roots: list[Path] | None = None,
) -> dict[str, Any]:
    return summarize_shadow_for_ui(
        load_shadow_row(case_id, visit_date=visit_date, roots=roots)
    )


def load_shadow_index(day: str, *, roots: list[Path] | None = None) -> dict[str, dict[str, Any]]:
    day_s = str(day or "").strip()[:10]
    out: dict[str, dict[str, Any]] = {}
    if len(day_s) != 10:
        return out
    for root in roots or _data_roots():
        path = shadow_jsonl_path(day_s, root=root)
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    ui = summarize_shadow_for_ui(row)
                    for key in (
                        str(row.get("case_id") or "").strip(),
                        str(row.get("visit_id") or "").strip(),
                        str(row.get("mis_id") or "").strip(),
                    ):
                        if key:
                            out[key] = ui
        except (OSError, json.JSONDecodeError):
            continue
        if out:
            break
    return out


def case_has_shadow_attention(ui: Mapping[str, Any] | None) -> bool:
    if not isinstance(ui, Mapping) or not ui.get("available"):
        return False
    return str(ui.get("case_attention_band") or "none") in {"poor", "critical"}
