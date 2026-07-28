"""Reusable command contract for the existing MIS exporter."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ExportArtifacts:
    parquet: Path
    csv: Path
    meta: Path


def daily_tag(day: date) -> str:
    return f"{day.isoformat()}_{(day + timedelta(days=1)).isoformat()}"


def export_artifacts(out_dir: Path, day: date) -> ExportArtifacts:
    prefix = out_dir / f"mis_protocol_{daily_tag(day)}"
    return ExportArtifacts(
        parquet=prefix.with_suffix(".parquet"),
        csv=prefix.with_suffix(".csv"),
        meta=out_dir / f"mis_protocol_{daily_tag(day)}.meta.json",
    )

def build_export_command(root: Path, out_dir: Path, day: date) -> Sequence[str]:
    return (
        sys.executable,
        str(root / "scripts" / "export_mis_protocol_month.py"),
        "--from",
        day.isoformat(),
        "--to",
        (day + timedelta(days=1)).isoformat(),
        "--out-dir",
        str(out_dir),
    )
