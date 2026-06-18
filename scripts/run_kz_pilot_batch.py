#!/usr/bin/env python3
"""Пилот batch: clients_consult/ → L1 (+ опционально L0/L2) → AI-предразметка → очередь разметки."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import env_load

    env_load.load_project_env(ROOT)
except ImportError:
    pass

CLIENTS = ROOT / "clients_consult"
SUPPORTED = {".pdf", ".txt", ".md", ".docx", ".rtf", ".odt", ".html"}


def _discover(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED and p.name != "README.md"
    )


def _import_from(src: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not files:
        raise SystemExit(f"Нет PDF/TXT в {src}")
    copied: list[Path] = []
    for i, src_file in enumerate(files, start=1):
        stem = src_file.stem
        if stem == "report":
            name = "report_1" + src_file.suffix.lower()
        elif stem.startswith("report-"):
            name = "report_" + stem.split("-", 1)[1] + src_file.suffix.lower()
        else:
            name = f"{stem}{src_file.suffix.lower()}"
        dst = dest / name
        shutil.copy2(src_file, dst)
        copied.append(dst)
    return copied


def _run_batch(folder: Path, *, tier: str, out: Path, ai_review: str, workers: int) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_methodist_batch.py"),
        "--folder",
        str(folder),
        "--tier",
        tier,
        "--workers",
        str(workers),
        "--ai-review",
        ai_review,
        "--out",
        str(out),
    ]
    print("→", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    report = out / "report.json"
    if not report.is_file():
        raise SystemExit(f"Нет report.json в {out}")
    return report


def _run_ai_review_all(report: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "review_batch_priority_cases.py"),
        "--report",
        str(report),
        "--all",
    ]
    print("→", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _write_pilot_report(out: Path, *, folder: Path, tiers: list[str], n_files: int) -> None:
    report_path = out / "report.json"
    data = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    summary = data.get("summary") or {}
    reports = data.get("reports") or []
    lines = [
        f"# Пилот batch КЗ ({date.today().isoformat()})",
        "",
        f"- **Папка:** `{folder}`",
        f"- **Файлов:** {n_files}",
        f"- **Уровни:** {', '.join(tiers)}",
        f"- **OK:** {summary.get('ok', len(reports))}/{summary.get('total', n_files)}",
        "",
        "## Сводка L1",
        "",
        "| case_id | overall % | rules % | failed | analysis_id |",
        "|---------|-----------|---------|--------|-------------|",
    ]
    for r in reports:
        aid = str(r.get("analysis_id") or "")
        lines.append(
            f"| {r.get('case_id', '')} | {r.get('overall_pct', '—')}% | "
            f"{r.get('rules_pct', '—')}% | {r.get('failed_rules_count', 0)} | "
            f"`{aid[:8]}…` |"
        )
    lines.extend(
        [
            "",
            "## Дальше",
            "",
            "1. UI: **Кабинет методиста** → **Очередь** → **Открыть** по `analysis_id`.",
            "2. Проверьте overrides → **Одобрить — сохранить для обучения движка**.",
            "3. `python3 scripts/export_training_feedback.py` → `ml/datasets/priority_cases.jsonl`.",
            "",
            f"Подробнее: `{out.name}/REVIEW_QUEUE.md`",
        ]
    )
    (out / "PILOT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_feedback() -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "export_training_feedback.py")]
    print("→", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", type=Path, default=CLIENTS)
    ap.add_argument(
        "--import-from",
        type=Path,
        default=None,
        help="Скопировать PDF/TXT из папки (напр. ~/Downloads/КЗ)",
    )
    ap.add_argument("--expected", type=int, default=10)
    ap.add_argument("--tier", choices=("L0", "L1", "L2"), default="L1")
    ap.add_argument(
        "--compare-tiers",
        action="store_true",
        help="Дополнительно прогнать L0 и L2 (сравнение этапов)",
    )
    ap.add_argument("--ai-review", choices=("off", "auto", "all"), default="auto")
    ap.add_argument("--skip-ai-queue", action="store_true", help="Не вызывать review_batch_priority_cases")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--submit-render", action="store_true")
    ap.add_argument("--render-base", default=None)
    ap.add_argument("--export-feedback", action="store_true")
    args = ap.parse_args()

    folder = args.folder.resolve()
    if args.import_from:
        copied = _import_from(args.import_from.resolve(), folder)
        print(f"Импортировано {len(copied)} файлов → {folder}")

    if not folder.is_dir():
        raise SystemExit(f"Создайте папку {folder} и положите PDF КЗ")

    paths = _discover(folder)
    if not paths:
        raise SystemExit(
            f"Нет файлов КЗ в {folder}. Пример:\n"
            f"  python3 scripts/run_kz_pilot_batch.py --import-from ~/Downloads/КЗ"
        )
    if args.expected and len(paths) != args.expected:
        print(f"⚠ Ожидалось {args.expected} файлов, найдено {len(paths)} — продолжаем.", file=sys.stderr)

    stamp = time.strftime("%Y-%m-%d")
    out = args.out or (ROOT / "ml" / "experiments" / f"kz_pilot_{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    tiers = [args.tier]
    if args.compare_tiers:
        tiers = ["L0", "L1", "L2"]

    main_report: Path | None = None
    for tier in tiers:
        tier_out = out if tier == args.tier else out / f"tier_{tier}"
        tier_out.mkdir(parents=True, exist_ok=True)
        main_report = _run_batch(
            folder,
            tier=tier,
            out=tier_out,
            ai_review=args.ai_review if tier == args.tier else "off",
            workers=args.workers,
        )
        if tier != args.tier:
            print(f"✓ {tier}: {tier_out / 'batch_summary.csv'}")

    assert main_report is not None

    if not args.skip_ai_queue and args.ai_review != "off":
        try:
            _run_ai_review_all(main_report)
        except subprocess.CalledProcessError:
            print("⚠ AI-review queue не построена (нет GOOGLE_API_KEY или METHODIST_AI_REVIEW=0)", file=sys.stderr)

    _write_pilot_report(out, folder=folder, tiers=tiers, n_files=len(paths))

    if args.submit_render:
        import os

        base = args.render_base or os.environ.get("RENDER_URL", "https://protocol-bimy.onrender.com")
        token = (os.environ.get("METHODIST_TOKEN") or os.environ.get("METHODIST_PIN") or "").strip()
        if not token:
            raise SystemExit("Для --submit-render задайте METHODIST_TOKEN в .env")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "submit_priority_reviews_render.py"),
            "--base",
            base,
            "--from-report",
            str(main_report),
            "--out",
            str(out / "render_reviews.json"),
        ]
        print("→ submit render …", flush=True)
        subprocess.run(cmd, check=True)

    if args.export_feedback:
        _export_feedback()

    print(f"\n✓ Готово: {out}")
    print(f"  CSV: {out / 'batch_summary.csv'}")
    print(f"  Очередь: {out / 'REVIEW_QUEUE.md'}")
    print(f"  Отчёт: {out / 'PILOT_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
