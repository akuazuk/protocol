"""Phase A: services CLI dry-run + LLM job fixture layout (no network, no PHI)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from services.llm_worker.grade_day import main as grade_main
from services.mis_bridge.extract_day import main as extract_main

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "llm_job"


def test_extract_day_dry_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    assert extract_main(["--day", "2026-08-06", "--dry-run", "--run-host", "mac"]) == 0


def test_extract_day_writes_meta(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    assert extract_main(["--day", "2026-08-06", "--run-host", "mac"]) == 0
    meta = tmp_path / "inbound" / "extract" / "mo_2026-08-06.meta.json"
    assert meta.is_file()
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["run_host"] == "mac"
    assert data["day"] == "2026-08-06"


def test_extract_day_packages_from_secure_csv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    secure = tmp_path / "secure_cases" / "2026" / "08"
    secure.mkdir(parents=True)
    src = secure / "mo_2026-08-06.csv"
    src.write_text("visit_id,note\n1,a\n2,b\n", encoding="utf-8")
    assert extract_main(["--day", "2026-08-06", "--from-secure", "--run-host", "mac"]) == 0
    out = tmp_path / "inbound" / "extract" / "mo_2026-08-06.csv"
    meta = tmp_path / "inbound" / "extract" / "mo_2026-08-06.meta.json"
    assert out.is_file()
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["row_count"] == 2
    assert len(data["checksum_sha256"]) == 64
    assert "a" in out.read_text(encoding="utf-8")


def test_llm_grade_day_dry_run(tmp_path: Path) -> None:
    outbox = tmp_path / "llm_outbox" / "fixture-run-1"
    outbox.mkdir(parents=True)
    shutil.copy(FIXTURES / "manifest.json", outbox / "manifest.json")
    shutil.copy(FIXTURES / "cases.jsonl", outbox / "cases.jsonl")
    assert (
        grade_main(
            [
                "--day",
                "2026-08-06",
                "--run-id",
                "fixture-run-1",
                "--data-root",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    result = tmp_path / "llm_inbox" / "fixture-run-1" / "result_manifest.json"
    assert result.is_file()
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["run_id"] == "fixture-run-1"
    assert data["model_primary"] == "dry-run"


def test_llm_job_fixtures_present() -> None:
    assert (FIXTURES / "manifest.json").is_file()
    assert (FIXTURES / "cases.jsonl").is_file()
    lines = (FIXTURES / "cases.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    assert "visit_id" in json.loads(lines[0])
