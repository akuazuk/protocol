#!/usr/bin/env python3
"""Э4.1: стратифицированная gold-выборка КЗ для калибровки/валидации оценки.

Страты: специальность (клинические) × L1-банд overall (0-49/50-59/60-69/70-79/80+),
с гарантией покрытия справок (pay_type=12) и red-flag страты. Источник - L1 cases jsonl
(+ CSV для pay_type). Манифест с visit_id пишем на /var/data (ПДн-смежное), в git -
только агрегатная таблица страт без ФИО/диагнозов.

  PYTHONPATH=. python3 scripts/build_kz_gold_sample.py \\
    --cases /var/data/mis_protocol/kz_l1_2026-07_cases.jsonl \\
    --csv /var/data/mis_protocol/mis_protocol_2026-07.csv \\
    --out /var/data/mis_protocol/kz_gold/gold_sample.jsonl \\
    --target-n 300
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Неклинические роли - исключаем из gold (см. план §9).
_EXCLUDE_SPEC = {
    "медицинская сестра", "медсестра", "стоматолог-терапевт", "стоматолог",
    "логопед", "-", "—", "", "лаборатория",
}
_RED_FLAG_KW = (
    "образование", "опухол", "нельзя исключить", "малигн", "зно", "c-r", "cr ",
    "тромб", "эмбол", "сепсис", "кровотеч", "анеми", "суицид", "инфаркт", "инсульт",
)


def _band(pct: float | None) -> str:
    if pct is None:
        return "na"
    if pct < 50:
        return "0-49"
    if pct < 60:
        return "50-59"
    if pct < 70:
        return "60-69"
    if pct < 80:
        return "70-79"
    return "80+"


def _spec_norm(s: str) -> str:
    return (s or "").strip().lower()


def _load_pay_types(csv_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not csv_path.is_file():
        return out
    try:
        from clinical_knowledge.mis_pay_type import normalize_pay_type_code
    except Exception:
        def normalize_pay_type_code(x):  # type: ignore[misc]
            return str(x or "").strip()
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csvmod.DictReader(f):
            vid = str(row.get("visit_id") or "").strip()
            if vid:
                out[vid] = normalize_pay_type_code(row.get("pay_type"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", type=Path,
                    default=ROOT / "data" / "ml" / "kz_gold" / "strata_summary.md")
    ap.add_argument("--target-n", type=int, default=300)
    ap.add_argument("--min-per-stratum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pay = _load_pay_types(args.csv) if args.csv else {}

    cases: list[dict] = []
    for line in args.cases.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        if c.get("error"):
            continue
        spec = _spec_norm(c.get("doctor_specialization"))
        if spec in _EXCLUDE_SPEC:
            continue
        vid = str(c.get("visit_id") or "")
        if not vid:
            continue
        pct = c.get("overall_pct")
        diag = (c.get("diagnosis_short") or "").lower()
        red = bool(c.get("status") == "manual_review_required" or any(k in diag for k in _RED_FLAG_KW))
        cases.append({
            "visit_id": vid,
            "specialty": c.get("doctor_specialization") or "-",
            "band": _band(pct if isinstance(pct, (int, float)) else None),
            "overall_pct": pct,
            "pay_type": pay.get(vid, ""),
            "red_flag": red,
        })

    rng = random.Random(args.seed)
    rng.shuffle(cases)

    # Страты по (специальность, банд).
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cases:
        strata[(c["specialty"], c["band"])].append(c)

    total = len(cases)
    picked: list[dict] = []
    picked_ids: set[str] = set()

    # 1) минимум на страту.
    for key, items in strata.items():
        for c in items[: args.min_per_stratum]:
            if c["visit_id"] not in picked_ids:
                picked.append(c)
                picked_ids.add(c["visit_id"])

    # 2) добор пропорционально до target-n.
    if len(picked) < args.target_n:
        remaining = [c for c in cases if c["visit_id"] not in picked_ids]
        need = args.target_n - len(picked)
        for c in remaining[:need]:
            picked.append(c)
            picked_ids.add(c["visit_id"])

    # 3) гарантируем справки (pay_type=12) и red-flag.
    def _ensure(pred, want: int, label: str) -> None:
        have = sum(1 for c in picked if pred(c))
        if have >= want:
            return
        pool = [c for c in cases if c["visit_id"] not in picked_ids and pred(c)]
        for c in pool[: want - have]:
            picked.append(c)
            picked_ids.add(c["visit_id"])

    _ensure(lambda c: str(c["pay_type"]) == "12", 20, "справки")
    _ensure(lambda c: c["red_flag"], 30, "red-flag")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for c in picked:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Агрегатный отчёт (без ПДн) в git.
    by_spec = Counter(c["specialty"] for c in picked)
    by_band = Counter(c["band"] for c in picked)
    n_spravki = sum(1 for c in picked if str(c["pay_type"]) == "12")
    n_red = sum(1 for c in picked if c["red_flag"])
    lines = [
        "# Gold-выборка КЗ (Э4) - страты",
        "",
        f"Источник: {args.cases.name}  |  всего клинич. кейсов: {total}  |  "
        f"отобрано: **{len(picked)}** (seed={args.seed})",
        f"Справок (pay_type=12): {n_spravki}  |  red-flag: {n_red}",
        "",
        "## По специальностям",
        "| Специальность | n |",
        "|--|--|",
    ]
    for spec, n in by_spec.most_common():
        lines.append(f"| {spec} | {n} |")
    lines += ["", "## По L1-бандам", "| Банд | n |", "|--|--|"]
    for band in ("0-49", "50-59", "60-69", "70-79", "80+", "na"):
        if by_band.get(band):
            lines.append(f"| {band} | {by_band[band]} |")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")

    print(f"gold отобрано: {len(picked)} / {total} клинич.")
    print(f"справок(12): {n_spravki}  red-flag: {n_red}  специальностей: {len(by_spec)}")
    print(f"манифест -> {args.out}")
    print(f"отчёт    -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
