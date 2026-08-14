"""Golden-eval: верно / неверно найден КП по диагнозу МО.

Fixture: tests/fixtures/mo_kp_suggest_golden.jsonl
Планы: docs/plans/2026-08-08-mo-kp-history-episode-suggest-v1.md,
docs/plans/2026-08-14-mo-kp-suggest-accuracy-v2.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "tests" / "fixtures" / "mo_kp_suggest_golden.jsonl"


def load_mo_kp_suggest_golden(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or DEFAULT_GOLDEN
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _path_has_all(path: str, fragments: list[str]) -> bool:
    pl = (path or "").lower()
    return all(str(f).lower() in pl for f in fragments if str(f).strip())


def _path_has_any(path: str, fragments: list[str]) -> bool:
    pl = (path or "").lower()
    return any(str(f).lower() in pl for f in fragments if str(f).strip())


def evaluate_mo_kp_suggest_row(
    row: dict[str, Any],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    """Прогон одной golden-строки через suggest_protocols_for_case."""
    import os

    from clinical_knowledge.case_protocol_suggest import suggest_protocols_for_case

    os.environ.setdefault("CASE_PROTOCOL_SUGGEST", "1")
    clinical = dict(row.get("clinical") or {})
    record = dict(row.get("record") or {})
    history_visits = list(row.get("history_visits") or [])
    result = suggest_protocols_for_case(
        clinical=clinical,
        record=record,
        history_visits=history_visits,
        limit=int(row.get("limit") or limit),
    )
    items = list(result.get("items") or [])
    paths = [str(it.get("source_path") or "") for it in items]
    kinds = [str(it.get("match_kind") or "") for it in items]
    top = items[0] if items else {}
    top_path = str(top.get("source_path") or "")
    top_kind = str(top.get("match_kind") or "")
    top_score = float(top.get("score") or 0)

    expect_all = list(row.get("expected_path_contains_all") or [])
    expect_any = list(row.get("expected_path_contains_any") or [])
    reject_any = list(row.get("reject_path_contains_any") or [])
    expect_kind = str(row.get("expect_match_kind") or "").strip()
    min_score = row.get("min_top_score")
    expect_no_clinical = bool(row.get("expect_no_clinical"))
    expect_episode_mode = str(row.get("expect_episode_mode") or "").strip()
    reject_top_all = list(row.get("reject_top_path_contains_all") or [])

    errors: list[str] = []
    hit = False
    if expect_all:
        hit = _path_has_all(top_path, expect_all)
        if not hit:
            # допускаем Hit@k если явно указано
            k = int(row.get("hit_at_k") or 1)
            hit = any(_path_has_all(p, expect_all) for p in paths[:k])
        if not hit:
            errors.append(f"expected_path_contains_all not in top: {expect_all}; top={top_path[:120]}")
    if expect_any and not any(_path_has_any(p, expect_any) for p in paths[: int(row.get("hit_at_k") or 1)]):
        # если expect_all уже задан - any опционален как доп. сигнал
        if not expect_all:
            errors.append(f"expected_path_contains_any miss: {expect_any}")
            hit = False
        elif not hit:
            pass
    if expect_kind and top_kind and top_kind != expect_kind:
        errors.append(f"expect_match_kind={expect_kind} got={top_kind}")
    if min_score is not None and top_score < float(min_score):
        errors.append(f"min_top_score {min_score} got {top_score}")
    if expect_no_clinical and any(k == "clinical" for k in kinds):
        errors.append("expect_no_clinical but clinical present in top")
    if "expect_available" in row:
        want_avail = bool(row.get("expect_available"))
        got_avail = bool(result.get("available"))
        if got_avail != want_avail:
            errors.append(f"expect_available={want_avail} got={got_avail} top={top_path[:80]}")
    if reject_any:
        bad = [p for p in paths if _path_has_any(p, reject_any)]
        if bad:
            errors.append(f"reject_path hit: {bad[0][:120]}")
    if reject_top_all and _path_has_all(top_path, reject_top_all):
        errors.append(f"reject_top_path_contains_all hit: {reject_top_all}")
    episode = result.get("dx_episode") or {}
    if expect_episode_mode and str(episode.get("mode") or "") != expect_episode_mode:
        errors.append(
            f"expect_episode_mode={expect_episode_mode} got={episode.get('mode')}"
        )

    # negative kind: достаточно отсутствия expected или наличия reject-check
    kind = str(row.get("kind") or "positive")
    if kind == "negative" and not errors and not expect_all and not expect_any:
        # чисто reject-тест
        pass

    ok = not errors
    return {
        "id": row.get("id"),
        "kind": kind,
        "ok": ok,
        "errors": errors,
        "top_path": top_path,
        "top_kind": top_kind,
        "top_score": top_score,
        "paths": paths,
        "dx_episode": episode,
        "hit": hit if expect_all or expect_any else None,
    }


def evaluate_mo_kp_suggest_golden(
    path: Path | None = None,
    *,
    ids: set[str] | None = None,
) -> dict[str, Any]:
    rows = load_mo_kp_suggest_golden(path)
    if ids:
        rows = [r for r in rows if str(r.get("id")) in ids]
    results = [evaluate_mo_kp_suggest_row(r) for r in rows]
    passed = sum(1 for r in results if r["ok"])
    return {
        "n": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(100.0 * passed / len(results), 1) if results else None,
        "results": results,
    }
