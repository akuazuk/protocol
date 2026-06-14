"""Append-only ML feedback store for Methodist Workbench (on-prem JSONL + secure KZ text)."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FEEDBACK_DIR = ROOT / "data" / "ml" / "feedback"
_log = logging.getLogger(__name__)

_VALID_VERDICTS = frozenset({"correct", "mostly_correct", "partially_wrong", "wrong"})
_VALID_TAGS = frozenset({
    "wrong_protocol",
    "missed_protocol",
    "false_positive_rule",
    "missed_issue",
    "wrong_population",
    "score_misleading",
    "wrong_diagnosis_block",
    "wrong_treatment_block",
    "query_too_vague",
    "other",
})
_VALID_KZ_COMPLIANCE_GOLD = frozenset({
    "compliant",
    "mostly_compliant",
    "partially_compliant",
    "non_compliant",
    "insufficient_data",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _probe_writable_dir(path: Path) -> bool:
    """True если каталог существует/создаётся и доступен для записи."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def feedback_dir() -> Path:
    """Каталог JSONL feedback. ML_FEEDBACK_DIR — только если путь реально доступен для записи.

    На Render без Persistent Disk ``/var/data/...`` недоступен (Permission denied) —
    откат на ``data/ml/feedback`` внутри проекта, чтобы consult-review не падал с 500.
    """
    raw = (os.environ.get("ML_FEEDBACK_DIR") or "").strip()
    if raw:
        configured = Path(raw)
        if _probe_writable_dir(configured):
            return configured
        _log.warning(
            "ML_FEEDBACK_DIR=%s недоступен для записи; fallback %s",
            configured,
            _DEFAULT_FEEDBACK_DIR,
        )
    if _probe_writable_dir(_DEFAULT_FEEDBACK_DIR):
        return _DEFAULT_FEEDBACK_DIR
    return _DEFAULT_FEEDBACK_DIR


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


def load_analysis_snapshot(analysis_id: str) -> dict[str, Any] | None:
    if not analysis_id:
        return None
    path = analyses_dir() / f"{analysis_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def load_secure_kz_text(text_hash: str) -> str | None:
    h = (text_hash or "").strip()
    if not h:
        return None
    digest = h.split(":", 1)[-1]
    path = secure_kz_dir() / f"{digest}.txt"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


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

        gold = (out.get("kz_compliance_gold") or "").strip()
        if gold and gold not in _VALID_KZ_COMPLIANCE_GOLD:
            raise ValueError("analysis_review: неверный kz_compliance_gold")

        block_overrides = out.get("block_overrides") or []
        if not isinstance(block_overrides, list):
            raise ValueError("analysis_review: block_overrides должен быть списком")
        for bo in block_overrides:
            if not (bo.get("block_key") or "").strip():
                raise ValueError("block_override: block_key обязателен")
            cmt = (bo.get("note") or "").strip()
            if len(cmt) > 280:
                raise ValueError("block_override: note ≤ 280 символов")

    elif et == "methodist_override":
        if not (out.get("rule_id") or "").strip():
            raise ValueError("methodist_override: rule_id обязателен")

    elif et == "retrieval_fix":
        if not (out.get("chosen_path") or "").strip():
            raise ValueError("retrieval_fix: chosen_path обязателен")
        tags = [str(t).strip() for t in (out.get("tags") or []) if str(t).strip()]
        if "wrong_protocol" in tags and not (out.get("rejected_path") or "").strip():
            raise ValueError("retrieval_fix: rejected_path обязателен при теге wrong_protocol")

    elif et == "search_review":
        if not (out.get("reviewer") or "").strip():
            raise ValueError("search_review: reviewer обязателен")
        verdict = str(out.get("methodist_verdict") or out.get("ranking_verdict") or "").strip()
        if verdict and verdict not in _VALID_VERDICTS:
            raise ValueError("search_review: неверный verdict")

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

    from clinical_knowledge.methodist_context import (
        build_methodist_review_context,
        structured_block_scores_dict,
    )

    ctx = build_methodist_review_context(result, full_text)
    comp_ctx = ctx.get("compliance") or {}
    rev = result.get("review") or {}
    comp_parts = rev.get("overall_compliance_components") or {}

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
        "compliance_overall_pct": comp_ctx.get("overall_pct"),
        "structured_pct": comp_ctx.get("structured_pct") or comp_parts.get("structured"),
        "rules_pct": comp_ctx.get("rules_pct") or comp_parts.get("rules"),
        "overall_status": comp_ctx.get("overall_status") or comp.get("overall_status") or "",
        "structured_block_scores": structured_block_scores_dict(result),
        "llm_criteria_count": len(ctx.get("llm_criteria") or []),
        "gate_score": gate_score,
        "send_decision": _send_decision_from_result(result),
        "rules_compliance_pct": _rules_compliance_pct(result),
        "matched_protocol_paths": matched[:12],
        "retrieval_top_paths": retrieval,
        "failed_rule_ids": _failed_rule_ids(result),
        "latency_ms": latency_ms,
        "embed_rerank_used": embed_used,
        "model_embed": model_embed,
        "sandbox": bool(sandbox),
        "review_focus": "protocol_compliance",
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
    category_slugs: str = "",
) -> dict[str, Any]:
    """Пишет kz_analysis, снимок и добавляет analysis_id/text_hash в ответ API."""
    if not full_text.strip():
        return result
    from clinical_knowledge.methodist_enrich import enrich_methodist_tier_payload

    result = enrich_methodist_tier_payload(
        result,
        tier=tier,
        full_text=full_text,
        category_slugs=category_slugs,
        latency_ms=latency_ms,
    )
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
    from clinical_knowledge.methodist_context import build_methodist_review_context

    out["methodist_review_context"] = build_methodist_review_context(out, full_text)
    return out


def normalize_since_param(since: str | None) -> str | None:
    """Нормализует ?since= для фильтра JSONL по полю ts (ISO-8601)."""
    s = (since or "").strip()
    if not s:
        return None
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return f"{s}T00:00:00Z"
    return s


def list_feedback_jsonl_files(*, include_events_aggregate: bool = False) -> list[Path]:
    d = feedback_dir()
    if not d.is_dir():
        return []
    paths = sorted(d.glob("*.jsonl"))
    if not include_events_aggregate:
        paths = [p for p in paths if p.name != "events.jsonl"]
    return paths


def collect_feedback_export_lines(*, since: str | None = None) -> dict[str, list[str]]:
    """Собирает строки JSONL по файлам; опционально только события с ts ≥ since."""
    since_norm = normalize_since_param(since)
    out: dict[str, list[str]] = {}
    for path in list_feedback_jsonl_files():
        lines_out: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if since_norm:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = (row.get("ts") or "").strip()
                if ts and ts < since_norm:
                    continue
            lines_out.append(line)
        out[path.name] = lines_out
    return out


def build_feedback_export_tar_gz(*, since: str | None = None) -> tuple[bytes, dict[str, Any]]:
    """Архив feedback/*.jsonl (без текста КЗ) + _manifest.json для sync с Render."""
    since_norm = normalize_since_param(since)
    files = collect_feedback_export_lines(since=since)
    event_count = sum(len(v) for v in files.values())
    manifest: dict[str, Any] = {
        "exported_at": _utc_now(),
        "since": since_norm,
        "feedback_dir_label": "ml/feedback",
        "files": {k: len(v) for k, v in sorted(files.items())},
        "event_count": event_count,
    }

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        manifest_info = tarfile.TarInfo(name="feedback/_manifest.json")
        manifest_info.size = len(manifest_bytes)
        tar.addfile(manifest_info, io.BytesIO(manifest_bytes))
        for fname, lines in sorted(files.items()):
            body = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
            file_info = tarfile.TarInfo(name=f"feedback/{fname}")
            file_info.size = len(body)
            tar.addfile(file_info, io.BytesIO(body))
    return buf.getvalue(), manifest


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
