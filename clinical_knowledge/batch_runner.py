"""Пакетный запуск проверки КЗ с batch_summary.csv/md (ТЗ §6, §16)."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .consult_analysis import analyze_consultation_text

from .text_extract import SUPPORTED_SUFFIXES, extract_text_from_path

DEFAULT_OUTPUT_DIR = Path("data/reports/kz_checks")


@dataclass
class BatchRow:
    file: str
    consultation_id: str
    overall_status: str | None = None
    overall_score: float | None = None
    confidence_score: float | None = None
    critical_count: int = 0
    major_count: int = 0
    warnings_count: int = 0
    missing_required_count: int = 0
    safety_cap_applied: bool = False
    protocols_matched: int = 0
    error: str | None = None

    def to_csv_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            "file": self.file,
            "consultation_id": self.consultation_id,
            "overall_status": self.overall_status or "",
            "overall_score": self.overall_score if self.overall_score is not None else "",
            "confidence_score": self.confidence_score if self.confidence_score is not None else "",
            "critical_count": self.critical_count,
            "major_count": self.major_count,
            "warnings_count": self.warnings_count,
            "missing_required_count": self.missing_required_count,
            "safety_cap_applied": self.safety_cap_applied,
            "protocols_matched": self.protocols_matched,
            "error": self.error or "",
        }


def _default_extract_text(path: Path) -> str:
    return extract_text_from_path(path)


def analyze_file(
    path: Path,
    *,
    out_dir: Path | None = None,
    text_extractor: Callable[[Path], str] | None = None,
    analysis_mode: str | None = None,
) -> dict[str, Any]:
    """Анализ одного КЗ; опционально сохраняет JSON/MD в out_dir."""
    extract = text_extractor or _default_extract_text
    text = extract(path)
    res = analyze_consultation_text(
        text,
        consultation_id=path.stem,
        source_file=path.name,
        source_file_type=path.suffix.lstrip("."),
        with_markdown=True,
        analysis_mode=analysis_mode,
    )
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        comp = res["compliance"]
        (out_dir / f"{path.stem}.json").write_text(
            json.dumps(comp, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        md = res.pop("report_markdown", "")
        (out_dir / f"{path.stem}.md").write_text(md, encoding="utf-8")
    return res


def run_batch(
    folder: Path,
    *,
    out_dir: Path | None = None,
    text_extractor: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Batch-анализ папки КЗ + batch_summary.csv/md."""
    target = out_dir or DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    rows: list[BatchRow] = []
    for path in files:
        row = BatchRow(file=path.name, consultation_id=path.stem)
        try:
            res = analyze_file(path, out_dir=target, text_extractor=text_extractor)
            comp = res["compliance"]
            row.overall_status = comp.get("overall_status")
            bd = comp.get("score_breakdown") or {}
            row.overall_score = bd.get("overall_score")
            row.confidence_score = comp.get("confidence_score")
            row.critical_count = len(comp.get("critical_issues") or [])
            row.major_count = len(comp.get("major_issues") or [])
            row.warnings_count = len(comp.get("warnings") or [])
            row.missing_required_count = len(comp.get("missing_required_items") or [])
            cap = comp.get("safety_cap") or {}
            row.safety_cap_applied = bool(cap.get("applied"))
            row.protocols_matched = len(comp.get("matched_protocols") or [])
        except Exception as exc:  # noqa: BLE001 — batch не падает на одном файле
            row.error = str(exc)[:300]
        rows.append(row)

    _write_batch_csv(target / "batch_summary.csv", rows)
    _write_batch_md(target / "batch_summary.md", rows, folder=folder)
    return {
        "analyzed": len(rows),
        "output_dir": str(target),
        "results": [r.to_csv_dict() for r in rows],
    }


def _write_batch_csv(path: Path, rows: list[BatchRow]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].to_csv_dict().keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def _write_batch_md(path: Path, rows: list[BatchRow], *, folder: Path) -> None:
    lines = [
        "# Сводка пакетной проверки КЗ",
        "",
        f"Папка: `{folder}`",
        f"Файлов: {len(rows)}",
        "",
        "| Файл | Статус | Оценка | Confidence | Критич. | Major | Предупр. | Пропуски | Cap | Проток. | Ошибка |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for r in rows:
        score = f"{r.overall_score:.0f}" if isinstance(r.overall_score, (int, float)) else "—"
        conf = f"{r.confidence_score:.0f}" if isinstance(r.confidence_score, (int, float)) else "—"
        err = (r.error or "")[:60].replace("|", "/")
        cap = "да" if r.safety_cap_applied else "—"
        lines.append(
            f"| {r.file} | {r.overall_status or '—'} | {score} | {conf} | "
            f"{r.critical_count} | {r.major_count} | {r.warnings_count} | "
            f"{r.missing_required_count} | {cap} | {r.protocols_matched} | {err or '—'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
