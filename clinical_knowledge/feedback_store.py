"""Append-only ML feedback store for Methodist Workbench (on-prem JSONL + secure KZ text)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent

_VALID_VERDICTS = frozenset({"correct", "mostly_correct", "partially_wrong", "wrong"})
_VALID_TAGS = frozenset({
    "wrong_protocol",
    "missed_protocol",
    "false_positive_rule",
    "missed_issue",
    "wrong_population",
    "cisz_wrong",
    "score_misleading",
    "other",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def feedback_dir() -> Path:
    raw = (os.environ.get("ML_FEEDBACK_DIR") or "").strip()
    return Path(raw) if raw else ROOT / "data" / "ml" / "feedback"


def secure_kz_dir() -> Path:
    return ROOT / "data" / "ml" / "secure" / "kz_text"


def analyses_dir() -> Path:
    return ROOT / "data" / "ml" / "analyses"


def methodist_token_expected() -> str:
    return (os.environ.get("METHODIST_TOKEN") or os.environ.get("METHODIST_PIN") or "").strip()


def methodist_default_reviewer() -> str:
    return (
        os.environ.get("METHODIST_REVIEWER")
        or os.environ.get("METHODIST_DEFAULT_REVIEWER")
        or ""
    ).strip()


def methodist_ui_auto_login() -> bool:
    return os.environ.get("METHODIST_UI_AUTO_LOGIN", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def methodist_auth_enabled() -> bool:
    return bool(methodist_token_expected())


def normalize_consult_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    return t.strip()


def text_hash(text: str) -> str:
    normalized = normalize_consult_text(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def verify_methodist_token(token: str | None) -> bool:
    expected = methodist_token_expected()
    if not expected:
        return False
    got = (token or "").strip()
    return bool(got) and got == expected


def token_from_request_headers(headers: Mapping[str, str]) -> str:
    return (headers.get("x-methodist-token") or headers.get("X-Methodist-Token") or "").strip()


def is_methodist_authenticated(
    headers: Mapping[str, str],
    *,
    body_methodist_mode: bool = False,
) -> bool:
    if verify_methodist_token(token_from_request_headers(headers)):
        return True
    return bool(body_methodist_mode) and verify_methodist_token(token_from_request_headers(headers))


def store_secure_kz_text(text: str) -> str:
    """Сохраняет полный текст КЗ по hash (on-prem, не в git)."""
    h = text_hash(text)
    digest = h.split(":", 1)[-1]
    d = secure_kz_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{digest}.txt"
    if not path.is_file():
        path.write_text(normalize_consult_text(text), encoding="utf-8")
    return h


def save_analysis_snapshot(analysis_id: str, snapshot: dict[str, Any]) -> Path | None:
    if not analysis_id:
        return None
    d = analyses_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{analysis_id}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _event_path(event_type: str) -> Path:
    safe = re.sub(r"[^a-z0-9_]+", "_", (event_type or "unknown").lower())
    return feedback_dir() / f"{safe}.jsonl"


def append_feedback_event(event: dict[str, Any]) -> str:
    """Append one validated event; returns event_id."""
    normalized = validate_and_normalize_event(event)
    event_id = normalized.get("event_id") or str(uuid.uuid4())
    normalized["event_id"] = event_id
    d = feedback_dir()
    d.mkdir(parents=True, exist_ok=True)
    et = normalized["event_type"]
    line = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    with _event_path(et).open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    with (d / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return event_id


def validate_and_normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("Ожидается JSON-объект")
    et = (event.get("event_type") or "").strip()
    if not et:
        raise ValueError("Поле event_type обязательно")

    out = dict(event)
    out["event_type"] = et
    out.setdefault("ts", _utc_now())

    reviewer = (out.get("reviewer") or "").strip()
    if et in ("analysis_review", "methodist_override", "retrieval_fix", "consult_gold_candidate"):
        if not reviewer:
            raise ValueError("Поле reviewer обязательно для оценки методиста")

    if et == "analysis_review":
        if out.get("rating") is None:
            raise ValueError("analysis_review: rating обязателен")
        verdict = (out.get("verdict") or "").strip()
        if verdict not in _VALID_VERDICTS:
            raise ValueError("analysis_review: неверный verdict")
        tags = out.get("tags") or []
        if not isinstance(tags, list):
            raise ValueError("analysis_review: tags должен быть списком")
        for tag in tags:
            if tag not in _VALID_TAGS:
                raise ValueError(f"analysis_review: неизвестный тег {tag!r}")
        note = (out.get("note") or "").strip()
        if len(note) > 2000:
            raise ValueError("analysis_review: note слишком длинный")
        overrides = out.get("overrides") or []
        if not isinstance(overrides, list):
            raise ValueError("analysis_review: overrides должен быть списком")
        for ov in overrides:
            if not (ov.get("rule_id") or "").strip():
                raise ValueError("override: rule_id обязателен")
            cmt = (ov.get("note") or "").strip()
            if len(cmt) > 280:
                raise ValueError("override: note ≤ 280 символов")

    elif et == "methodist_override":
        if not (out.get("rule_id") or "").strip():
            raise ValueError("methodist_override: rule_id обязателен")

    elif et == "retrieval_fix":
        if not (out.get("chosen_path") or "").strip():
            raise ValueError("retrieval_fix: chosen_path обязателен")

    elif et == "kz_analysis":
        if not (out.get("analysis_id") or "").strip():
            raise ValueError("kz_analysis: analysis_id обязателен")
        if not (out.get("text_hash") or "").strip():
            raise ValueError("kz_analysis: text_hash обязателен")

    return out


def _first_rubric(result: dict[str, Any]) -> str | None:
    sa = result.get("structured_analysis") or {}
    rub = sa.get("rubric_specifics") or {}
    rubrics = rub.get("rubrics") or rub.get("rubric_slugs") or []
    if rubrics:
        return str(rubrics[0])
    matches = sa.get("matches") or result.get("matches") or []
    for m in matches:
        if isinstance(m, dict) and m.get("rubric_slug"):
            return str(m["rubric_slug"])
    paths = result.get("retrieval_paths") or []
    if paths:
        p0 = str(paths[0])
        if "/" in p0:
            return p0.split("/", 1)[0]
    return None


def _send_decision_from_result(result: dict[str, Any]) -> str:
    sg = result.get("send_gate") or {}
    if not sg:
        comp = (result.get("structured_analysis") or {}).get("compliance") or {}
        sg = comp.get("send_gate") or {}
    sd = (sg.get("sign_decision") or sg.get("send_decision") or "").strip()
    if sd:
        return sd
    if sg.get("gate_allowed") is False:
        return "blocked"
    if sg.get("requires_override"):
        return "needs_review"
    risk = (sg.get("send_risk_level") or "").strip()
    if risk == "blocked":
        return "blocked"
    if risk in ("medium", "high"):
        return "allowed_with_warnings"
    return "allowed"


def _failed_rule_ids(result: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    cr = result.get("clinical_rules") or {}
    rc = cr.get("rules_check") or {}
    for f in rc.get("findings") or []:
        if f.get("passed") is False and not f.get("skipped"):
            rid = (f.get("rule_id") or "").strip()
            if rid:
                ids.append(rid)
    comp = (result.get("structured_analysis") or {}).get("compliance") or {}
    for issue in comp.get("critical_issues") or []:
        if isinstance(issue, dict):
            rid = (issue.get("rule_id") or issue.get("code") or "").strip()
            if rid:
                ids.append(rid)
    return sorted(set(ids))


def _rules_compliance_pct(result: dict[str, Any]) -> float | None:
    cr = result.get("clinical_rules") or {}
    rc = cr.get("rules_check") or {}
    pct = rc.get("rules_compliance_pct")
    if pct is not None:
        return float(pct)
    comp = (result.get("structured_analysis") or {}).get("compliance") or {}
    bd = comp.get("score_breakdown") or {}
    if isinstance(bd, dict) and bd.get("protocol_rules") is not None:
        return float(bd["protocol_rules"])
    return None


def build_kz_analysis_event(
    *,
    result: dict[str, Any],
    tier: str,
    full_text: str,
    consultation_id: str = "",
    latency_ms: int | None = None,
    sandbox: bool = False,
    reviewer: str = "",
) -> dict[str, Any]:
    analysis_id = str(uuid.uuid4())
    th = store_secure_kz_text(full_text)
    sg = result.get("send_gate") or {}
    if not sg:
        comp = (result.get("structured_analysis") or {}).get("compliance") or {}
        sg = comp.get("send_gate") or {}

    retrieval = list(result.get("retrieval_paths") or [])[:5]
    matched = []
    cr = result.get("clinical_rules") or {}
    for mp in cr.get("matched_protocols") or []:
        if isinstance(mp, dict) and mp.get("path"):
            matched.append(str(mp["path"]))
    if not matched:
        matched = list(retrieval)

    comp = (result.get("structured_analysis") or {}).get("compliance") or {}
    gate_score = sg.get("gate_score")
    if gate_score is None:
        gate_score = comp.get("overall_score") or result.get("overall_score")

    embed_used = bool(result.get("rag_used") or result.get("embed_rerank_used"))
    model_embed = (os.environ.get("GEMINI_EMBEDDING_MODEL") or "intfloat/multilingual-e5-small").strip()

    event = {
        "event_type": "kz_analysis",
        "analysis_id": analysis_id,
        "ts": _utc_now(),
        "text_hash": th,
        "consultation_id": consultation_id or result.get("consultation_id") or "",
        "tier": (tier or result.get("review_tier") or "L2").upper(),
        "rubric": _first_rubric(result) or "",
        "gate_score": gate_score,
        "send_decision": _send_decision_from_result(result),
        "overall_status": comp.get("overall_status") or result.get("overall_status") or "",
        "rules_compliance_pct": _rules_compliance_pct(result),
        "matched_protocol_paths": matched[:12],
        "retrieval_top_paths": retrieval,
        "failed_rule_ids": _failed_rule_ids(result),
        "latency_ms": latency_ms,
        "embed_rerank_used": embed_used,
        "model_embed": model_embed,
        "sandbox": bool(sandbox),
    }
    if reviewer:
        event["reviewer"] = reviewer
    return event


def enrich_result_with_methodist_autolog(
    result: dict[str, Any],
    *,
    tier: str,
    full_text: str,
    consultation_id: str = "",
    latency_ms: int | None = None,
    sandbox: bool = False,
    reviewer: str = "",
) -> dict[str, Any]:
    """Пишет kz_analysis, снимок и добавляет analysis_id/text_hash в ответ API."""
    if not full_text.strip():
        return result
    event = build_kz_analysis_event(
        result=result,
        tier=tier,
        full_text=full_text,
        consultation_id=consultation_id,
        latency_ms=latency_ms,
        sandbox=sandbox,
        reviewer=reviewer,
    )
    append_feedback_event(event)
    excerpt = normalize_consult_text(full_text)[:500]
    snapshot = {
        "analysis_id": event["analysis_id"],
        "text_hash": event["text_hash"],
        "tier": event["tier"],
        "saved_at": event["ts"],
        "text_excerpt": excerpt,
        "api_result": result,
    }
    save_analysis_snapshot(event["analysis_id"], snapshot)
    out = dict(result)
    out["analysis_id"] = event["analysis_id"]
    out["text_hash"] = event["text_hash"]
    out["model_info"] = {
        "embed_rerank_used": event.get("embed_rerank_used"),
        "model_embed": event.get("model_embed"),
        "tier": event["tier"],
    }
    return out


def expand_analysis_review_events(review: dict[str, Any]) -> list[dict[str, Any]]:
    """Разворачивает analysis_review в отдельные methodist_override и retrieval_fix."""
    events: list[dict[str, Any]] = [review]
    analysis_id = review.get("analysis_id")
    text_hash = review.get("text_hash")
    reviewer = review.get("reviewer")
    ts = review.get("ts") or _utc_now()

    for ov in review.get("overrides") or []:
        events.append({
            "event_type": "methodist_override",
            "ts": ts,
            "analysis_id": analysis_id,
            "text_hash": text_hash,
            "reviewer": reviewer,
            "rule_id": ov.get("rule_id"),
            "system_pass": ov.get("system_pass"),
            "human_pass": ov.get("human_pass"),
            "note": (ov.get("note") or "")[:280],
        })

    rf = review.get("retrieval_fix")
    if isinstance(rf, dict) and rf.get("chosen_path"):
        events.append({
            "event_type": "retrieval_fix",
            "ts": ts,
            "analysis_id": analysis_id,
            "reviewer": reviewer,
            "query": (rf.get("query") or "")[:500],
            "rejected_path": rf.get("rejected_path") or "",
            "chosen_path": rf.get("chosen_path") or "",
        })
    return events
