"""Агрегация статистики кабинета методиста для дашборда ML / обучения."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

RUBRIC_LABELS: dict[str, str] = {
    "nevrologiya-neyrokhirurgiya": "Неврология / нейрохирургия",
    "akusherstvo-ginekologiya": "Акушерство и гинекология",
    "bolezni-sistemy-krovoobrashcheniya": "Болезни системы кровообращения",
    "gastroenterologiya": "Гастроэнтерология",
    "endokrinologiya-narusheniya-obmena-veshchestv": "Эндокринология",
    "pulmonologiya-ftiziatriya": "Пульмонология / фтизиатрия",
    "dermatovenerologiya": "Дermatovenerология",
    "stomatologiya": "Стоматология",
    "novoobrazovaniya": "Новообразования",
    "pediatriya": "Педиатрия",
    "terapiya": "Терапия",
}

ML_READINESS_THRESHOLDS: dict[str, Any] = {
    "unique_kz_analyzed": {"target": 50, "label": "Уникальных КЗ в базе (kz_analysis)"},
    "analysis_reviews": {"target": 30, "label": "Оценок методиста (analysis_review)"},
    "retrieval_fix_feedback": {"target": 50, "label": "Исправлений RAG (retrieval_fix, не bootstrap)"},
    "entailment_pairs": {"target": 100, "label": "Пар entailment из overrides"},
    "per_rubric_reviews": {"target": 8, "label": "Оценок на рубрику (минимум)"},
    "ai_approved_reviews": {"target": 20, "label": "AI-оценок, одобренных методистом"},
}

SEARCH_READINESS_THRESHOLDS: dict[str, Any] = {
    "search_retrieval_fix": {"target": 20, "label": "retrieval_fix из вкладки поиска"},
    "search_reviews": {"target": 30, "label": "Оценок поиска (search_review + AI-одобрено)"},
    "search_ai_approved": {"target": 15, "label": "AI-оценок поиска, одобренных методистом"},
}

TAG_LABELS: dict[str, str] = {
    "wrong_protocol": "Неверный КП в RAG",
    "missed_protocol": "Не нашли нужный КП",
    "query_too_vague": "Запрос слишком общий",
    "false_positive_rule": "Ложное замечание правила",
    "missed_issue": "Пропущена ошибка в КЗ",
    "wrong_population": "Ошибка популяции",
    "wrong_rubric": "Неверная рубрика",
    "wrong_condition": "Неверная нозология",
    "wrong_section": "Неверный раздел",
    "wrong_icd_suggestion": "Ошибка МКБ",
    "score_misleading": "Итоговый % вводит в заблуждение",
    "wrong_diagnosis_block": "Неверно оценён блок «Диагноз»",
    "wrong_treatment_block": "Неверно оценён блок «Лечение»",
    "other": "Прочее",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rubric_label(slug: str) -> str:
    s = (slug or "").strip()
    return RUBRIC_LABELS.get(s, s.replace("-", " ").title() if s else "Не указана")


def _short_hash(h: str) -> str:
    h = (h or "").replace("sha256:", "")
    return h[:8] if h else ""


def iter_feedback_events(feedback_dir: Path) -> list[dict[str, Any]]:
    from clinical_knowledge.feedback_store import list_feedback_jsonl_files

    events: list[dict[str, Any]] = []
    if not feedback_dir.is_dir():
        return events
    for path in list_feedback_jsonl_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row["_source_file"] = path.name
                events.append(row)
    return events


def _pct(n: int, target: int) -> float:
    if target <= 0:
        return 100.0
    return round(min(100.0, 100.0 * n / target), 1)


def _compute_reanalysis_deltas(kz_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in kz_events:
        th = ev.get("text_hash") or ""
        if th:
            by_hash[th].append(ev)
    deltas: list[dict[str, Any]] = []
    for th, rows in by_hash.items():
        if len(rows) < 2:
            continue
        rows_sorted = sorted(rows, key=lambda r: r.get("ts") or "")
        first, last = rows_sorted[0], rows_sorted[-1]
        if first.get("ts") == last.get("ts"):
            continue
        b_over = first.get("compliance_overall_pct")
        a_over = last.get("compliance_overall_pct")
        b_rules = len(first.get("failed_rule_ids") or [])
        a_rules = len(last.get("failed_rule_ids") or [])
        if b_over == a_over and b_rules == a_rules:
            continue
        deltas.append(
            {
                "text_hash_short": _short_hash(th),
                "rubric": _rubric_label(str(first.get("rubric") or "")),
                "runs": len(rows_sorted),
                "first_ts": first.get("ts"),
                "last_ts": last.get("ts"),
                "before_overall_pct": b_over,
                "after_overall_pct": a_over,
                "delta_overall_pct": round(float(a_over or 0) - float(b_over or 0), 1)
                if a_over is not None and b_over is not None
                else None,
                "before_failed_rules": b_rules,
                "after_failed_rules": a_rules,
            }
        )
    deltas.sort(key=lambda d: abs(d.get("delta_overall_pct") or 0), reverse=True)
    return deltas[:12]


def _rating_histogram(reviews: list[dict[str, Any]]) -> dict[str, int]:
    hist = {str(i): 0 for i in range(1, 6)}
    for ev in reviews:
        try:
            r = int(ev.get("rating"))
        except (TypeError, ValueError):
            continue
        if 1 <= r <= 5:
            hist[str(r)] += 1
    return hist


def _events_by_day(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for ev in events:
        ts = (ev.get("ts") or "")[:10]
        if ts:
            counts[ts] += 1
    return [{"date": d, "count": counts[d]} for d in sorted(counts)]


def _load_engine_releases() -> dict[str, Any]:
    path = ROOT / "data" / "ml" / "engine_release_log.json"
    if not path.is_file():
        return {"releases": [], "ml_experiments": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"releases": [], "ml_experiments": []}


def _load_export_manifest() -> dict[str, Any] | None:
    path = ROOT / "ml" / "datasets" / "export_manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_model_manifest() -> dict[str, Any]:
    path = ROOT / "ml" / "registry" / "model_manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _compute_protocol_match_stats(
    kz_events: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    retrieval_fixes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Метрики опоры A: заполненность matched/retrieval и hit@k по retrieval_fix."""
    total_kz = len(kz_events)
    with_matched = sum(1 for e in kz_events if e.get("matched_protocol_paths"))
    with_top = sum(1 for e in kz_events if e.get("retrieval_top_paths"))
    kz_by_id = {
        str(e.get("analysis_id")): e for e in kz_events if e.get("analysis_id")
    }

    gold_rows: list[tuple[dict[str, Any], str]] = []
    for rev in reviews:
        rf = rev.get("retrieval_fix")
        if isinstance(rf, dict) and (rf.get("chosen_path") or "").strip():
            gold_rows.append((rev, str(rf["chosen_path"]).strip()))
    for fix in retrieval_fixes:
        chosen = (fix.get("chosen_path") or "").strip()
        if chosen:
            gold_rows.append((fix, chosen))

    labeled = 0
    hit1 = 0
    hit3 = 0
    for ev, chosen in gold_rows:
        kz = kz_by_id.get(str(ev.get("analysis_id") or "")) or {}
        top = list(kz.get("retrieval_top_paths") or [])
        if not top:
            top = list(kz.get("matched_protocol_paths") or [])
        if not top:
            continue
        labeled += 1
        if chosen in top[:1]:
            hit1 += 1
        if chosen in top[:3]:
            hit3 += 1

    embedded_fixes = sum(
        1 for r in reviews if isinstance(r.get("retrieval_fix"), dict) and r["retrieval_fix"].get("chosen_path")
    )
    protocol_tags = sum(
        1
        for r in reviews
        if {"wrong_protocol", "missed_protocol"} & {str(t) for t in (r.get("tags") or [])}
    )

    return {
        "total_kz_runs": total_kz,
        "kz_with_matched_protocol": with_matched,
        "kz_with_matched_protocol_pct": _pct(with_matched, total_kz) if total_kz else 0.0,
        "kz_with_retrieval_top": with_top,
        "kz_with_retrieval_top_pct": _pct(with_top, total_kz) if total_kz else 0.0,
        "labeled_retrieval_gold": labeled,
        "protocol_hit_at_1_pct": _pct(hit1, labeled) if labeled else None,
        "protocol_hit_at_3_pct": _pct(hit3, labeled) if labeled else None,
        "protocol_tag_reviews": protocol_tags,
        "retrieval_fix_events": len(retrieval_fixes) + embedded_fixes,
    }


def _short_path(path: str) -> str:
    p = (path or "").replace("\\", "/").strip()
    return p.rsplit("/", 1)[-1][:72] if p else ""


def _load_search_golden_snapshot() -> dict[str, Any] | None:
    p = Path(__file__).resolve().parents[1] / "data" / "ml" / "search_golden_snapshot.json"
    if not p.is_file():
        return None
    try:
        import json

        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _compute_search_domain_stats(
    events: list[dict[str, Any]],
    retrieval_fixes: list[dict[str, Any]],
) -> dict[str, Any]:
    """KPI поиска протоколов: телеметрия, разметка, Hit@k, AI-оценки."""
    from clinical_knowledge.search_telemetry import aggregate_protocol_search

    protocol_searches = [e for e in events if e.get("event_type") == "protocol_search"]
    search_reviews = [e for e in events if e.get("event_type") == "search_review"]
    search_ui_fixes = [
        f
        for f in retrieval_fixes
        if str(f.get("source") or "") == "protocol_search_ui"
        or f.get("review_source") == "ai_assisted"
    ]

    tag_counts: Counter[str] = Counter()
    for ev in search_ui_fixes + search_reviews:
        for tag in ev.get("tags") or []:
            tag_counts[str(tag)] += 1

    ai_assisted = sum(
        1
        for ev in search_ui_fixes + search_reviews
        if ev.get("review_source") == "ai_assisted" or ev.get("ai_review")
    )
    ai_approved = sum(
        1
        for ev in search_ui_fixes + search_reviews
        if ev.get("methodist_approved") is True
    )
    manual_only = sum(
        1
        for ev in search_ui_fixes
        if not ev.get("ai_review") and ev.get("review_source") != "ai_assisted"
    )

    labeled = 0
    hit1 = 0
    hit3 = 0
    for fix in search_ui_fixes:
        chosen = (fix.get("chosen_path") or "").strip()
        top = list(fix.get("retrieval_top_paths") or [])
        if not chosen or not top:
            continue
        labeled += 1
        if chosen in top[:1]:
            hit1 += 1
        if chosen in top[:3]:
            hit3 += 1

    improvements: Counter[str] = Counter()
    for ev in search_ui_fixes + search_reviews:
        ai = ev.get("ai_review") if isinstance(ev.get("ai_review"), dict) else ev
        for imp in ai.get("engine_improvements_ru") or []:
            s = str(imp).strip()
            if s:
                improvements[s[:120]] += 1

    recent: list[dict[str, Any]] = []
    combined = sorted(
        search_ui_fixes + search_reviews,
        key=lambda r: r.get("ts") or "",
        reverse=True,
    )
    for ev in combined[:15]:
        recent.append(
            {
                "ts": (ev.get("ts") or "")[:19],
                "event_type": ev.get("event_type"),
                "query_hash": (ev.get("query_hash") or "")[:12] or None,
                "rejected_short": _short_path(str(ev.get("rejected_path") or "")),
                "chosen_short": _short_path(str(ev.get("chosen_path") or "")),
                "tags": list(ev.get("tags") or [])[:4],
                "methodist_approved": ev.get("methodist_approved"),
                "ranking_rating": ev.get("ranking_rating"),
                "reviewer": ev.get("reviewer"),
            }
        )

    telemetry = aggregate_protocol_search(protocol_searches)
    search_fix_count = len([f for f in retrieval_fixes if str(f.get("source") or "") == "protocol_search_ui"])

    readiness_items = [
        {
            "key": "search_retrieval_fix",
            "label": SEARCH_READINESS_THRESHOLDS["search_retrieval_fix"]["label"],
            "current": search_fix_count,
            "target": SEARCH_READINESS_THRESHOLDS["search_retrieval_fix"]["target"],
            "pct": _pct(search_fix_count, SEARCH_READINESS_THRESHOLDS["search_retrieval_fix"]["target"]),
        },
        {
            "key": "search_reviews",
            "label": SEARCH_READINESS_THRESHOLDS["search_reviews"]["label"],
            "current": len(search_reviews) + ai_approved,
            "target": SEARCH_READINESS_THRESHOLDS["search_reviews"]["target"],
            "pct": _pct(len(search_reviews) + ai_approved, SEARCH_READINESS_THRESHOLDS["search_reviews"]["target"]),
        },
        {
            "key": "search_ai_approved",
            "label": SEARCH_READINESS_THRESHOLDS["search_ai_approved"]["label"],
            "current": ai_approved,
            "target": SEARCH_READINESS_THRESHOLDS["search_ai_approved"]["target"],
            "pct": _pct(ai_approved, SEARCH_READINESS_THRESHOLDS["search_ai_approved"]["target"]),
        },
    ]
    readiness_overall = round(sum(r["pct"] for r in readiness_items) / len(readiness_items), 1) if readiness_items else 0.0

    fixes_by_day = _events_by_day(search_ui_fixes + search_reviews)

    funnel_step_counts: Counter[int] = Counter()
    for ev in search_ui_fixes + search_reviews:
        fs = ev.get("funnel_step")
        if fs is None:
            continue
        try:
            funnel_step_counts[int(fs)] += 1
        except (TypeError, ValueError):
            continue

    golden_snapshot = _load_search_golden_snapshot()

    return {
        "telemetry": telemetry,
        "protocol_search_count": len(protocol_searches),
        "search_ui_retrieval_fix": search_fix_count,
        "search_review_count": len(search_reviews),
        "ai_assisted_labels": ai_assisted,
        "ai_approved_labels": ai_approved,
        "manual_labels": manual_only,
        "labeled_with_top_paths": labeled,
        "hit_at_1_pct": _pct(hit1, labeled) if labeled else None,
        "hit_at_3_pct": _pct(hit3, labeled) if labeled else None,
        "tag_counts": [
            {"tag": t, "label": TAG_LABELS.get(t, t), "count": c}
            for t, c in tag_counts.most_common(10)
        ],
        "engine_improvements_top": [
            {"text": t, "count": c} for t, c in improvements.most_common(12)
        ],
        "readiness": {"overall_pct": readiness_overall, "items": readiness_items},
        "recent_labels": recent,
        "funnel_step_breakdown": [
            {"step": step, "count": cnt}
            for step, cnt in sorted(funnel_step_counts.items())
        ],
        "golden_eval_snapshot": golden_snapshot,
        "charts": {
            "labels_by_day": fixes_by_day,
            "tags": [{"tag": t, "label": TAG_LABELS.get(t, t), "count": c} for t, c in tag_counts.most_common(8)],
        },
    }


def build_methodist_dashboard_stats(*, feedback_dir: Path | None = None) -> dict[str, Any]:
    """Полная статистика для GET /api/methodist/stats."""
    from clinical_knowledge.feedback_store import feedback_dir as resolve_feedback_dir

    fb = feedback_dir or resolve_feedback_dir()
    events = iter_feedback_events(fb)

    by_type: Counter[str] = Counter()
    for ev in events:
        by_type[str(ev.get("event_type") or "unknown")] += 1

    kz_events = [e for e in events if e.get("event_type") == "kz_analysis"]
    reviews = [e for e in events if e.get("event_type") == "analysis_review"]
    overrides = [e for e in events if e.get("event_type") == "methodist_override"]
    retrieval_fixes = [e for e in events if e.get("event_type") == "retrieval_fix"]

    unique_hashes_kz = {e.get("text_hash") for e in kz_events if e.get("text_hash")}
    reviewed_ids = {e.get("analysis_id") for e in reviews if e.get("analysis_id")}
    kz_ids = {e.get("analysis_id") for e in kz_events if e.get("analysis_id")}
    pending_review = len(kz_ids - reviewed_ids)

    # latest review per text_hash
    reviews_by_hash: dict[str, dict[str, Any]] = {}
    for ev in sorted(reviews, key=lambda r: r.get("ts") or ""):
        th = ev.get("text_hash") or ""
        if th:
            reviews_by_hash[th] = ev

    priority_count = sum(
        1
        for ev in reviews
        if isinstance(ev.get("rating"), (int, float)) and int(ev["rating"]) <= 2
    )

    ai_approved = sum(1 for ev in reviews if ev.get("methodist_approved") is True)
    ai_assisted = sum(1 for ev in reviews if ev.get("review_source") == "ai_assisted")

    tag_counts: Counter[str] = Counter()
    for ev in reviews:
        for tag in ev.get("tags") or []:
            tag_counts[str(tag)] += 1

    rule_override_counts: Counter[str] = Counter()
    for ev in reviews + overrides:
        if ev.get("event_type") == "analysis_review":
            for ov in ev.get("overrides") or []:
                rid = (ov.get("rule_id") or "").strip()
                if rid:
                    rule_override_counts[rid] += 1
        elif ev.get("event_type") == "methodist_override":
            rid = (ev.get("rule_id") or "").strip()
            if rid:
                rule_override_counts[rid] += 1

    block_override_counts: Counter[str] = Counter()
    for ev in reviews:
        for bo in ev.get("block_overrides") or []:
            bk = (bo.get("block_key") or "").strip()
            if bk:
                block_override_counts[bk] += 1

    rubric_kz: Counter[str] = Counter()
    rubric_reviews: Counter[str] = Counter()
    hash_to_rubric: dict[str, str] = {}
    for ev in kz_events:
        slug = str(ev.get("rubric") or "")
        rubric_kz[slug] += 1
        th = ev.get("text_hash") or ""
        if th and slug:
            hash_to_rubric[th] = slug
    for ev in reviews:
        th = ev.get("text_hash") or ""
        slug = hash_to_rubric.get(th, "")
        if slug:
            rubric_reviews[slug] += 1

    per_rubric: list[dict[str, Any]] = []
    per_rubric_target = int(ML_READINESS_THRESHOLDS["per_rubric_reviews"]["target"])
    for slug, kz_n in rubric_kz.most_common():
        rev_n = rubric_reviews.get(slug, 0)
        unique_in_rubric = len(
            {e.get("text_hash") for e in kz_events if e.get("rubric") == slug and e.get("text_hash")}
        )
        per_rubric.append(
            {
                "rubric_slug": slug,
                "rubric_label": _rubric_label(slug),
                "kz_runs": kz_n,
                "unique_kz": unique_in_rubric,
                "reviews": rev_n,
                "reviews_pct": _pct(rev_n, per_rubric_target),
                "reviews_needed": max(0, per_rubric_target - rev_n),
                "ready": rev_n >= per_rubric_target,
            }
        )

    entailment_from_feedback = 0
    for ev in reviews:
        entailment_from_feedback += len(ev.get("overrides") or [])
        entailment_from_feedback += len(
            [bo for bo in (ev.get("block_overrides") or []) if bo.get("verdict") == "disagree" or bo.get("human_agrees") is False]
        )
    entailment_from_feedback += len(overrides)

    export_manifest = _load_export_manifest()
    entailment_exported = (export_manifest or {}).get("counts", {}).get("entailment_pairs", entailment_from_feedback)
    retrieval_feedback = len(retrieval_fixes)

    readiness_items = [
        {
            "key": "unique_kz_analyzed",
            "label": ML_READINESS_THRESHOLDS["unique_kz_analyzed"]["label"],
            "current": len(unique_hashes_kz),
            "target": ML_READINESS_THRESHOLDS["unique_kz_analyzed"]["target"],
            "pct": _pct(len(unique_hashes_kz), ML_READINESS_THRESHOLDS["unique_kz_analyzed"]["target"]),
        },
        {
            "key": "analysis_reviews",
            "label": ML_READINESS_THRESHOLDS["analysis_reviews"]["label"],
            "current": len(reviews),
            "target": ML_READINESS_THRESHOLDS["analysis_reviews"]["target"],
            "pct": _pct(len(reviews), ML_READINESS_THRESHOLDS["analysis_reviews"]["target"]),
        },
        {
            "key": "retrieval_fix_feedback",
            "label": ML_READINESS_THRESHOLDS["retrieval_fix_feedback"]["label"],
            "current": retrieval_feedback,
            "target": ML_READINESS_THRESHOLDS["retrieval_fix_feedback"]["target"],
            "pct": _pct(retrieval_feedback, ML_READINESS_THRESHOLDS["retrieval_fix_feedback"]["target"]),
        },
        {
            "key": "entailment_pairs",
            "label": ML_READINESS_THRESHOLDS["entailment_pairs"]["label"],
            "current": entailment_exported,
            "target": ML_READINESS_THRESHOLDS["entailment_pairs"]["target"],
            "pct": _pct(entailment_exported, ML_READINESS_THRESHOLDS["entailment_pairs"]["target"]),
        },
        {
            "key": "ai_approved_reviews",
            "label": ML_READINESS_THRESHOLDS["ai_approved_reviews"]["label"],
            "current": ai_approved,
            "target": ML_READINESS_THRESHOLDS["ai_approved_reviews"]["target"],
            "pct": _pct(ai_approved, ML_READINESS_THRESHOLDS["ai_approved_reviews"]["target"]),
        },
    ]
    readiness_overall = round(sum(r["pct"] for r in readiness_items) / len(readiness_items), 1)

    avg_rating = None
    ratings = [int(e["rating"]) for e in reviews if isinstance(e.get("rating"), (int, float))]
    if ratings:
        avg_rating = round(sum(ratings) / len(ratings), 2)

    engine_log = _load_engine_releases()
    reanalysis = _compute_reanalysis_deltas(kz_events)
    protocol_match = _compute_protocol_match_stats(kz_events, reviews, retrieval_fixes)
    search = _compute_search_domain_stats(events, retrieval_fixes)

    try:
        from rag_server import BUILD_VERSION  # type: ignore
        build_version = BUILD_VERSION
    except Exception:
        build_version = None

    return {
        "generated_at": _utc_now(),
        "feedback_dir": str(fb),
        "build_version": build_version,
        "summary": {
            "total_events": len(events),
            "events_by_type": dict(by_type),
            "unique_kz": len(unique_hashes_kz),
            "total_kz_runs": len(kz_events),
            "analysis_reviews": len(reviews),
            "pending_review": pending_review,
            "priority_cases": priority_count,
            "avg_methodist_rating": avg_rating,
            "ai_assisted_reviews": ai_assisted,
            "ai_approved_reviews": ai_approved,
            "readiness_overall_pct": readiness_overall,
            "protocol_hit_at_3_pct": protocol_match.get("protocol_hit_at_3_pct"),
            "kz_with_matched_protocol_pct": protocol_match.get("kz_with_matched_protocol_pct"),
        },
        "protocol_match": protocol_match,
        "search": search,
        "pool": {
            "in_training_pool": len(unique_hashes_kz),
            "labeled_reviews": len(reviews),
            "labeled_unique_kz": len(reviews_by_hash),
            "unlabeled_unique_kz": max(0, len(unique_hashes_kz) - len(reviews_by_hash)),
            "pending_analysis_ids": pending_review,
            "methodist_overrides": len(overrides),
            "retrieval_fixes": retrieval_feedback,
            "entailment_pairs_estimated": entailment_exported,
        },
        "ml_readiness": {
            "overall_pct": readiness_overall,
            "items": readiness_items,
            "ml_deployed": bool(_load_model_manifest().get("active")),
            "next_steps": [
                "Накопить retrieval_fix из реальных ошибок RAG (сейчас мало)",
                "Довести reviews до 30+ и по 8+ на рубрику",
                "После порога: finetune_embedder.py на retrieval_pairs_resolved.jsonl",
                "Engine fixes остаются основным ROI до 100+ reviews",
            ],
        },
        "charts": {
            "events_by_type": [
                {"type": k, "label": k, "count": v} for k, v in by_type.most_common()
            ],
            "rating_histogram": _rating_histogram(reviews),
            "tags_top": [
                {"tag": t, "label": TAG_LABELS.get(t, t), "count": c}
                for t, c in tag_counts.most_common(10)
            ],
            "activity_by_day": _events_by_day(events),
            "rubric_kz_runs": [
                {"rubric": _rubric_label(s), "slug": s, "count": n}
                for s, n in rubric_kz.most_common()
            ],
        },
        "specialties": per_rubric,
        "rule_overrides_top": [
            {"rule_id": rid, "count": c} for rid, c in rule_override_counts.most_common(12)
        ],
        "block_overrides_top": [
            {"block_key": bk, "count": c} for bk, c in block_override_counts.most_common(8)
        ],
        "reanalysis_deltas": reanalysis,
        "engine_releases": engine_log.get("releases") or [],
        "ml_experiments": engine_log.get("ml_experiments") or [],
        "export_manifest": export_manifest,
        "verdict_breakdown": dict(Counter(str(e.get("verdict") or "") for e in reviews)),
    }
