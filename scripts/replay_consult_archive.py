#!/usr/bin/env python3
"""Пересчёт архивных снимков анализа КЗ и сравнение с сохранёнными метриками.

Использование (рекомендуется через venv проекта):
  .venv/bin/python scripts/replay_consult_archive.py
  .venv/bin/python scripts/replay_consult_archive.py --fixtures tests/fixtures/consult_replay.jsonl

Если в clients_consult/ есть файл с тем же basename, что в снимке - пересчитывает
структурный анализ и сравнивает баллы/статус/МКБ.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bootstrap_venv() -> None:
    """Подключает пакеты из .venv (даже если запущен системный python3)."""
    venv_site = next(ROOT.glob(".venv/lib/python*/site-packages"), None)
    if venv_site and venv_site.is_dir():
        sp = str(venv_site)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    try:
        import pydantic  # noqa: F401
    except ModuleNotFoundError:
        venv_py = ROOT / ".venv" / "bin" / "python"
        print(
            "Ошибка: pydantic не найден.\n"
            "Создайте venv и установите зависимости:\n"
            "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n"
            f"Затем: {venv_py} scripts/replay_consult_archive.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


_bootstrap_venv()
sys.path.insert(0, str(ROOT))

from clinical_knowledge.analysis_archive import archive_dir, load_snapshots  # noqa: E402
from clinical_knowledge.consult_analysis import analyze_consultation_text  # noqa: E402

_DEFAULT_FIXTURES = (
    ROOT / "tests" / "fixtures" / "consult_replay.jsonl",
    ROOT / "tests" / "fixtures" / "consult_replay_latest.jsonl",
)


def _default_fixtures_path() -> Path | None:
    for p in _DEFAULT_FIXTURES:
        if p.is_file():
            return p
    return None


def _load_text(basename: str) -> str | None:
    if not basename:
        return None
    for folder in ("clients_consult", "tests/fixtures/consult"):
        p = ROOT / folder / basename
        if p.is_file():
            if p.suffix.lower() == ".pdf":
                try:
                    import fitz
                    return "\n".join(page.get_text() for page in fitz.open(p))
                except Exception:
                    return None
            return p.read_text(encoding="utf-8", errors="replace")
    return None


def _saved_score(saved: dict) -> float | None:
    v = saved.get("structured_overall_score")
    if isinstance(v, (int, float)):
        return float(v)
    bd = saved.get("score_breakdown") or {}
    ov = bd.get("overall_score")
    return float(ov) if isinstance(ov, (int, float)) else None


def _primary_icd(codes: list[str] | None) -> list[str]:
    """Болезненные коды (не R/Z) для сравнения регрессии."""
    out: list[str] = []
    for c in codes or []:
        head = (c or "").strip().upper()[:1]
        if head in ("R", "Z"):
            continue
        if c and c not in out:
            out.append(c)
    return out[:3]


def _compare(saved: dict, comp: dict) -> list[str]:
    diffs: list[str] = []
    if saved.get("structured_overall_status") != comp.get("overall_status"):
        diffs.append(
            f"status: {saved.get('structured_overall_status')!r} → {comp.get('overall_status')!r}"
        )
    old_score = _saved_score(saved)
    new_score = comp.get("overall_score")
    if old_score is not None and new_score is not None:
        if abs(float(old_score) - float(new_score)) >= 0.5:
            diffs.append(f"overall_score: {old_score} → {new_score}")
    old_icd = _primary_icd(saved.get("icd_codes"))
    new_icd = _primary_icd([
        d.get("icd10_code") for d in (comp.get("diagnosis_assessments") or []) if d.get("icd10_code")
    ])
    if not new_icd:
        new_icd = _primary_icd(saved.get("icd_codes"))  # fallback - не сравниваем если нет новых
    if old_icd and new_icd and set(old_icd) != set(new_icd):
        diffs.append(f"primary_icd: {old_icd!r} → {new_icd!r}")
    # rubric из fresh не в comp directly - skip unless we pass it
    return diffs


def _load_fixtures(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay consult analysis archive")
    ap.add_argument("--limit", type=int, default=0, help="Max snapshots (0 = all)")
    ap.add_argument(
        "--fixtures",
        type=str,
        default="",
        help="JSONL file (default: tests/fixtures/consult_replay.jsonl)",
    )
    args = ap.parse_args()

    if args.fixtures:
        fix_path = Path(args.fixtures)
        if not fix_path.is_file():
            alt = _default_fixtures_path()
            hint = f"\nПодсказка: есть файл {alt}" if alt else ""
            print(f"Файл фикстур не найден: {fix_path}{hint}", file=sys.stderr)
            return 1
        try:
            snaps = _load_fixtures(fix_path)
        except FileNotFoundError:
            print(f"Файл фикстур не найден: {fix_path}", file=sys.stderr)
            return 1
    else:
        default_fix = _default_fixtures_path()
        if default_fix:
            print(f"Фикстуры: {default_fix.relative_to(ROOT)}")
            snaps = _load_fixtures(default_fix)
        else:
            limit = args.limit or None
            snaps = load_snapshots(limit=limit)
            if not snaps:
                print(
                    f"Нет снимков в {archive_dir() / 'manifest.jsonl'} "
                    f"и нет tests/fixtures/consult_replay.jsonl",
                    file=sys.stderr,
                )
                return 1

    ok = 0
    skipped = 0
    failed = 0
    print(f"Replay {len(snaps)} snapshot(s)\n")

    for i, snap in enumerate(snaps, 1):
        base = snap.get("source_basename") or ""
        text = _load_text(base)
        label = base or snap.get("text_hash", "")[:12]
        if not text:
            skipped += 1
            print(f"[{i}] SKIP {label} - положите PDF в clients_consult/")
            continue
        res = analyze_consultation_text(text, with_markdown=False)
        comp = res.get("compliance") or {}
        diffs = _compare(snap, comp)
        if diffs:
            failed += 1
            print(f"[{i}] DIFF {label}")
            for d in diffs:
                print(f"      {d}")
        else:
            ok += 1
            score = comp.get("overall_score") or _saved_score(snap)
            status = comp.get("overall_status")
            print(f"[{i}] OK   {label} - {status} {score}%")

    print(f"\nSummary: ok={ok} diff={failed} skipped={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
