"""Зоны оценки МО: Оформление / Диагноз / План по протоколу + риск.

Канон: docs/plans/2026-08-08-mo-analytics-mz-sheet-layers-v2.md
UI: docs/plans/2026-08-08-mo-analytics-ui-target-v2.md
Движок опирается на evaluate_mo_rubric_mz и не заменяет deep/№55.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from clinical_knowledge.mo_action_queue_select import QUEUE_INCLUDE_CODES
from clinical_knowledge.mo_rubric_mz import evaluate_mo_rubric_mz, load_rubric_config

ROOT = Path(__file__).resolve().parent.parent
BANDS_PATH = ROOT / "config" / "mo_zone_bands.yaml"
ENGINE = "mo_zones_v1"

ZONE_UI = {
    "documentation": "zone1",
    "diagnosis": "zone2a",
    "plan": "zone2b",
}

_CLINICAL_KEYS = (
    "complaints",
    "anamnesis_doctor",
    "anamnesis_auto",
    "objective_status",
    "clinical_diagnosis",
    "mis_diagnos",
    "mis_diagnosis",
    "diagnosis_mis",
    "exam_data",
    "exam_recommendations",
    "treatment_recommendations",
    "visit_date",
    "visit_time",
)


def zones_scores_enabled() -> bool:
    raw = (os.environ.get("MO_ZONE_SCORES") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def load_zone_bands() -> dict[str, Any]:
    raw = yaml.safe_load(BANDS_PATH.read_text(encoding="utf-8")) or {}
    out = {
        "bad_below": float(raw.get("bad_below") or 50),
        "ok_at_or_above": float(raw.get("ok_at_or_above") or 85),
        "labels_ru": dict(raw.get("labels_ru") or {}),
        "zone_labels_ru": dict(raw.get("zone_labels_ru") or {}),
        "engine": str(raw.get("engine") or ENGINE),
    }
    try:
        from clinical_knowledge.mo_scoring_profile import effective_zone_bands

        out.update(effective_zone_bands(out))
    except Exception:  # noqa: BLE001
        pass
    return out


def _norm(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "null", "-"}:
        return ""
    return re.sub(r"\s+", " ", text)


def clinical_slots_from_mapping(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for src in sources:
        if not isinstance(src, Mapping):
            continue
        for key in _CLINICAL_KEYS:
            if out.get(key):
                continue
            val = src.get(key)
            if val not in (None, ""):
                out[key] = val
        nested = src.get("clinical")
        if isinstance(nested, Mapping):
            for key in _CLINICAL_KEYS:
                if out.get(key):
                    continue
                val = nested.get(key)
                if val not in (None, ""):
                    out[key] = val
    return out


def _cfg_by_id() -> dict[str, dict[str, Any]]:
    cfg = load_rubric_config()
    return {str(c.get("id")): dict(c) for c in (cfg.get("criteria") or []) if c.get("id")}


def _kp_matched(protocol_suggest: Mapping[str, Any] | None) -> tuple[bool, str]:
    if not isinstance(protocol_suggest, Mapping):
        return False, ""
    try:
        from clinical_knowledge.case_protocol_suggest import clinical_kp_hit

        hit = clinical_kp_hit(dict(protocol_suggest))
    except Exception:  # noqa: BLE001
        hit = None
    if hit:
        return True, str(hit.get("title") or "")[:120]
    items = protocol_suggest.get("items")
    if isinstance(items, list) and items:
        # Есть кандидаты, но без clinical hit - unmatched для штрафа «по КП».
        return False, ""
    return False, ""


def _first_text(clinical: Mapping[str, Any], fields: Sequence[str]) -> str:
    parts = [_norm(clinical.get(f)) for f in fields]
    return " ".join(p for p in parts if p).strip()


def _apply_protocol_gate(
    items: list[dict[str, Any]],
    *,
    clinical: Mapping[str, Any],
    kp_matched: bool,
) -> list[dict[str, Any]]:
    """Без подобранного КП не штрафуем план «не по протоколу» - n/a вместо ложного 0/0.5 alignment."""
    cfg = _cfg_by_id()
    out: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        cid = str(item.get("id") or "")
        meta = cfg.get(cid) or {}
        zone = str(meta.get("zone") or item.get("zone") or "")
        requires_protocol = bool(meta.get("requires_protocol", item.get("requires_protocol")))
        optional = bool(meta.get("optional", item.get("optional")))
        item["zone"] = zone or None
        item["requires_protocol"] = requires_protocol
        item["optional"] = optional
        fields = list(meta.get("fields") or [])
        text = _first_text(clinical, fields) if fields else ""
        rule = str(meta.get("rule") or "")

        if requires_protocol and zone == "plan" and not kp_matched:
            if rule == "dynamics_correction" and item.get("score") is None:
                pass
            elif not text and rule in {"plan_present_or_aligned", "follow_up_present"}:
                # Пустой план - дефект присутствия, даже без КП.
                item["score"] = 0.0
                item["score_label"] = "0"
                item["reason"] = "План не указан"
            elif text or rule == "dynamics_correction":
                item["score"] = None
                item["score_label"] = "n/a"
                item["reason"] = "Протокол не подобран - критерий плана не штрафуем за несоответствие протоколу"
                item["na_reason"] = "kp_not_matched"
        out.append(item)
    return out


def _mean_pct(scores: list[float]) -> float | None:
    if not scores:
        return None
    return round(100.0 * sum(scores) / len(scores), 1)


def band_for_zone(
    pct: float | None,
    scored_items: Sequence[Mapping[str, Any]],
    *,
    bands: Mapping[str, Any] | None = None,
) -> str:
    cfg = bands or load_zone_bands()
    bad_below = float(cfg["bad_below"])
    ok_at = float(cfg["ok_at_or_above"])
    if pct is None or not scored_items:
        return "na"
    has_zero = any(
        isinstance(i.get("score"), (int, float)) and float(i["score"]) <= 0.001
        and not i.get("optional")
        for i in scored_items
    )
    if pct < bad_below or has_zero:
        return "bad"
    has_half = any(
        isinstance(i.get("score"), (int, float)) and 0.001 < float(i["score"]) < 0.999
        for i in scored_items
    )
    if pct < ok_at or has_half:
        return "weak"
    return "ok"


def _label_ru(band: str, bands: Mapping[str, Any]) -> str:
    labels = bands.get("labels_ru") if isinstance(bands.get("labels_ru"), Mapping) else {}
    return str(labels.get(band) or band)


def _safety_from_findings(findings: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    codes: list[str] = []
    critical = False
    important = False
    for finding in findings or []:
        if not isinstance(finding, Mapping):
            continue
        code = str(finding.get("code") or finding.get("finding_code") or "").strip()
        if not code or code not in QUEUE_INCLUDE_CODES:
            continue
        if not code.startswith("C_"):
            continue
        codes.append(code)
        sev = str(finding.get("severity") or "").upper()
        title = str(finding.get("title_ru") or finding.get("detail_ru") or "").lower()
        if sev == "P0" or "критич" in title or code in {"C_red_flag", "C_red_flag_unrouted"}:
            critical = True
        else:
            important = True
    if critical:
        band = "critical"
    elif important:
        band = "important"
    else:
        band = "none"
    return {"band": band, "codes": codes[:12]}


def _llm_overlay(llm_action_judge: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(llm_action_judge, Mapping):
        return None
    if llm_action_judge.get("available") is False:
        return None
    # Поддержка плоского и вложенного формата судьи.
    def _block(name: str) -> Any:
        block = llm_action_judge.get(name)
        if isinstance(block, Mapping):
            return block
        nested = llm_action_judge.get("stages") or llm_action_judge.get("answers")
        if isinstance(nested, Mapping) and isinstance(nested.get(name), Mapping):
            return nested.get(name)
        return None

    mapping = {
        "zone1": _block("completeness") or _block("documentation"),
        "zone2a": _block("diagnosis_assessment") or _block("diagnosis"),
        "zone2b": _block("plan_assessment") or _block("plan"),
    }
    if not any(mapping.values()):
        return None
    return mapping


def _attention(
    *,
    safety: Mapping[str, Any],
    zone1_band: str,
    zone2a_band: str,
    zone2b_band: str,
    zone1_items: Sequence[Mapping[str, Any]],
    zone2a_items: Sequence[Mapping[str, Any]],
    zone2b_items: Sequence[Mapping[str, Any]],
    kp_status: str,
) -> tuple[str, str]:
    if safety.get("band") == "critical":
        return "safety", "Есть критичный риск (безопасность)"
    if safety.get("band") == "important":
        return "safety", "Есть важный сигнал риска"
    if zone2a_band == "bad":
        reason = next(
            (str(i.get("reason") or "") for i in zone2a_items if i.get("score") == 0.0),
            "Диагноз слабо оформлен или не обоснован",
        )
        return "zone2a", reason[:180]
    if zone2b_band == "bad" and kp_status == "matched":
        reason = next(
            (str(i.get("reason") or "") for i in zone2b_items if i.get("score") == 0.0),
            "План не соответствует подобранному протоколу",
        )
        return "zone2b", reason[:180]
    if zone1_band == "bad":
        # Узко: только явные пробелы оформления.
        mo = next((i for i in zone1_items if i.get("id") == "mo_complete"), None)
        dx_empty = any(i.get("id") == "mo_complete" and i.get("score") == 0.0 for i in zone1_items)
        if mo and mo.get("score") == 0.0:
            return "zone1", "МО не оформлен в полном объёме"
        if dx_empty:
            return "zone1", "Ключевые блоки оформления пусты"
        # missing diagnosis+code handled in zone2a; здесь только mo_complete=0
        return "none", ""
    return "none", ""


def compute_mo_zone_scores(case_ctx: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Посчитать зоны для одного случая.

    case_ctx keys: clinical, meta, block_scores, findings, patient_history,
    protocol_suggest, llm_action_judge, prior_clinical, document_kind, score_eligible.
    """
    ctx = dict(case_ctx or {})
    bands_cfg = load_zone_bands()
    zone_labels = bands_cfg.get("zone_labels_ru") or {}

    eligible = ctx.get("score_eligible")
    kind = str(ctx.get("document_kind") or "").strip()
    if eligible is False or (kind and kind not in {"clinical_visit", "consultation", ""}):
        if kind and kind not in {"clinical_visit", "consultation"}:
            return {
                "ok": True,
                "engine": ENGINE,
                "skipped": True,
                "reason": "non_clinical",
                "zone1_pct": None,
                "zone2a_pct": None,
                "zone2b_pct": None,
                "zone1_band": "na",
                "zone2a_band": "na",
                "zone2b_band": "na",
                "zone2b_kp_status": "not_applicable",
                "attention_primary": "none",
                "attention_reason_ru": "",
                "rubric_pct": None,
                "rubric_json": None,
                "criteria": [],
                "safety": {"band": "none", "codes": []},
                "llm_overlay": None,
                "layer_engine": ENGINE,
                "layer_updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

    clinical = clinical_slots_from_mapping(
        ctx.get("clinical") if isinstance(ctx.get("clinical"), Mapping) else None,
        ctx,
    )
    meta = dict(ctx.get("meta") or {})
    block_scores = dict(ctx.get("block_scores") or {})
    prior = ctx.get("prior_clinical") if isinstance(ctx.get("prior_clinical"), Mapping) else None
    suggest = ctx.get("protocol_suggest") if isinstance(ctx.get("protocol_suggest"), Mapping) else None
    findings = ctx.get("findings") if isinstance(ctx.get("findings"), list) else []

    # История: prior может прийти из patient_history.summary
    if prior is None:
        hist = ctx.get("patient_history")
        if isinstance(hist, Mapping):
            prior_visit = hist.get("prior_clinical") or hist.get("prior")
            if isinstance(prior_visit, Mapping):
                prior = prior_visit
            elif int((hist.get("summary") or {}).get("n_visits") or hist.get("n_visits") or 0) <= 0:
                prior = None

    rubric = evaluate_mo_rubric_mz(
        clinical=clinical,
        meta=meta,
        block_scores=block_scores,
        prior_clinical=prior,
        protocol_suggest=suggest,
    )
    kp_ok, kp_title = _kp_matched(suggest)
    kp_status = "matched" if kp_ok else "unmatched"

    criteria = _apply_protocol_gate(
        list(rubric.get("criteria") or []),
        clinical=clinical,
        kp_matched=kp_ok,
    )

    by_zone: dict[str, list[dict[str, Any]]] = {
        "documentation": [],
        "diagnosis": [],
        "plan": [],
    }
    for item in criteria:
        z = str(item.get("zone") or "")
        if z in by_zone:
            by_zone[z].append(item)

    def _scored(zone_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            i
            for i in zone_items
            if isinstance(i.get("score"), (int, float)) and not i.get("optional")
        ]

    z1_scored = _scored(by_zone["documentation"])
    # optional exam_data: include only when scored (not n/a)
    for i in by_zone["documentation"]:
        if i.get("optional") and isinstance(i.get("score"), (int, float)):
            z1_scored.append(i)
    z2a_scored = _scored(by_zone["diagnosis"])
    z2b_scored = _scored(by_zone["plan"])

    zone1_pct = _mean_pct([float(i["score"]) for i in z1_scored])
    zone2a_pct = _mean_pct([float(i["score"]) for i in z2a_scored])
    zone2b_pct = _mean_pct([float(i["score"]) for i in z2b_scored])

    zone1_band = band_for_zone(zone1_pct, z1_scored, bands=bands_cfg)
    zone2a_band = band_for_zone(zone2a_pct, z2a_scored, bands=bands_cfg)
    zone2b_band = band_for_zone(zone2b_pct, z2b_scored, bands=bands_cfg)
    if kp_status == "unmatched" and zone2b_pct is None:
        zone2b_band = "na"

    safety = _safety_from_findings(findings)
    attention_primary, attention_reason = _attention(
        safety=safety,
        zone1_band=zone1_band,
        zone2a_band=zone2a_band,
        zone2b_band=zone2b_band,
        zone1_items=by_zone["documentation"],
        zone2a_items=by_zone["diagnosis"],
        zone2b_items=by_zone["plan"],
        kp_status=kp_status,
    )

    # Пересчёт rubric_pct после protocol gate
    scored_all = [i for i in criteria if isinstance(i.get("score"), (int, float))]
    rubric_pct = (
        round(100.0 * sum(float(i["score"]) for i in scored_all) / len(scored_all), 1)
        if scored_all
        else None
    )

    llm_overlay = _llm_overlay(
        ctx.get("llm_action_judge") if isinstance(ctx.get("llm_action_judge"), Mapping) else None
    )

    compact_criteria = [
        {
            "id": i.get("id"),
            "title": i.get("title"),
            "zone": i.get("zone"),
            "requires_protocol": bool(i.get("requires_protocol")),
            "optional": bool(i.get("optional")),
            "score": i.get("score"),
            "score_label": i.get("score_label"),
            "reason": i.get("reason"),
            "na_reason": i.get("na_reason"),
        }
        for i in criteria
    ]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "ok": True,
        "engine": ENGINE,
        "skipped": False,
        "zone1": {
            "pct": zone1_pct,
            "band": zone1_band,
            "label_ru": str(zone_labels.get("documentation") or "Оформление"),
            "band_label_ru": _label_ru(zone1_band, bands_cfg),
            "requires_protocol": False,
        },
        "zone2a": {
            "pct": zone2a_pct,
            "band": zone2a_band,
            "label_ru": str(zone_labels.get("diagnosis") or "Диагноз"),
            "band_label_ru": _label_ru(zone2a_band, bands_cfg),
            "requires_protocol": True,
        },
        "zone2b": {
            "pct": zone2b_pct,
            "band": zone2b_band,
            "label_ru": str(zone_labels.get("plan") or "План по протоколу"),
            "band_label_ru": (
                "протокол не подобран"
                if zone2b_band == "na" and kp_status == "unmatched"
                else _label_ru(zone2b_band, bands_cfg)
            ),
            "kp_status": kp_status,
            "kp_title": kp_title or None,
            "requires_protocol": True,
        },
        "zone1_pct": zone1_pct,
        "zone2a_pct": zone2a_pct,
        "zone2b_pct": zone2b_pct,
        "zone1_band": zone1_band,
        "zone2a_band": zone2a_band,
        "zone2b_band": zone2b_band,
        "zone2b_kp_status": kp_status,
        "attention_primary": attention_primary,
        "attention_reason_ru": attention_reason,
        "safety": safety,
        "criteria": compact_criteria,
        "rubric_pct": rubric_pct,
        "rubric_json": json.dumps(compact_criteria, ensure_ascii=False),
        "llm_overlay": llm_overlay,
        "layer_engine": ENGINE,
        "layer_updated_at": now,
        "scorer_version": rubric.get("scorer_version"),
    }


def zones_api_payload(zones: Mapping[str, Any]) -> dict[str, Any]:
    """Публичный блок для case detail / overview."""
    if not isinstance(zones, Mapping) or not zones.get("ok"):
        return {"ok": False, "engine": ENGINE}
    return {
        "ok": True,
        "engine": zones.get("engine") or ENGINE,
        "zone1": zones.get("zone1"),
        "zone2a": zones.get("zone2a"),
        "zone2b": zones.get("zone2b"),
        "safety": zones.get("safety") or {"band": "none", "codes": []},
        "attention_primary": zones.get("attention_primary") or "none",
        "attention_reason_ru": zones.get("attention_reason_ru") or "",
        "criteria": zones.get("criteria") or [],
        "rubric_pct": zones.get("rubric_pct"),
        "llm_overlay": zones.get("llm_overlay"),
        "layer_engine": zones.get("layer_engine") or ENGINE,
        "layer_updated_at": zones.get("layer_updated_at"),
    }


def warehouse_zone_columns(zones: Mapping[str, Any]) -> dict[str, Any]:
    """Плоские колонки для fact_mo_case."""
    if not isinstance(zones, Mapping):
        return {}
    return {
        "zone1_pct": zones.get("zone1_pct"),
        "zone2a_pct": zones.get("zone2a_pct"),
        "zone2b_pct": zones.get("zone2b_pct"),
        "zone1_band": zones.get("zone1_band") or "na",
        "zone2a_band": zones.get("zone2a_band") or "na",
        "zone2b_band": zones.get("zone2b_band") or "na",
        "zone2b_kp_status": zones.get("zone2b_kp_status") or "unmatched",
        "attention_primary": zones.get("attention_primary") or "none",
        "attention_reason_ru": (zones.get("attention_reason_ru") or "")[:240],
        "rubric_json": zones.get("rubric_json"),
        "rubric_pct": zones.get("rubric_pct"),
        "layer_engine": zones.get("layer_engine") or ENGINE,
        "layer_updated_at": zones.get("layer_updated_at"),
    }
