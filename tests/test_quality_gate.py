"""Тесты порога качества поиска (eval/quality_gate.py).

Гейт - это то, что не пускает в релиз регрессию отбора протоколов. Сломанный
гейт (всегда exit 0) не падает и ничем себя не выдаёт, поэтому релизы просто
перестают быть защищёнными. Отсюда тесты именно на коды возврата.

Полный прогон eval требует корпуса и ключа API, в CI их нет
(см. scripts/ops/run_search_quality_gate.sh), поэтому здесь на вход гейту
подаются синтетические отчёты.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "eval" / "quality_gate.py"


def _run(report: Path, min_rate: str | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(GATE), "--report", str(report)]
    if min_rate is not None:
        cmd += ["--min-pass-rate", min_rate]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "report.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def test_gate_passes_above_threshold(tmp_path: Path) -> None:
    report = _write(tmp_path, {"summary": {"total": 10, "passed": 10, "pass_rate": 1.0}})
    res = _run(report, "0.9")
    assert res.returncode == 0, res.stderr


def test_gate_fails_below_threshold(tmp_path: Path) -> None:
    """Главный тест: просадка отбора обязана валить гейт."""
    report = _write(tmp_path, {"summary": {"total": 10, "passed": 5, "pass_rate": 0.5}})
    res = _run(report, "0.9")
    assert res.returncode == 1, f"регрессия не поймана: {res.stdout} {res.stderr}"


def test_gate_passes_exactly_at_threshold(tmp_path: Path) -> None:
    """Ровно на пороге - проход: иначе гейт краснел бы от ошибки округления."""
    report = _write(tmp_path, {"summary": {"total": 10, "passed": 9, "pass_rate": 0.9}})
    res = _run(report, "0.9")
    assert res.returncode == 0, res.stderr


def test_gate_errors_on_empty_report(tmp_path: Path) -> None:
    """Ноль кейсов - это код 2 (ошибка), а не «порог пройден».

    Иначе пустой отчёт из упавшего eval выглядел бы как успешная проверка.
    """
    report = _write(tmp_path, {"summary": {"total": 0, "passed": 0, "pass_rate": 0.0}})
    res = _run(report, "0.9")
    assert res.returncode == 2, f"пустой отчёт принят за успех: {res.stdout}"


def test_gate_errors_on_missing_report(tmp_path: Path) -> None:
    res = _run(tmp_path / "нет-такого.json", "0.9")
    assert res.returncode == 2


def test_gate_errors_on_broken_json(tmp_path: Path) -> None:
    p = tmp_path / "report.json"
    p.write_text("{не json", encoding="utf-8")
    res = _run(p, "0.9")
    assert res.returncode == 2


def test_gate_computes_summary_from_cases(tmp_path: Path) -> None:
    """Отчёт без summary: доля считается по cases[].ok."""
    report = _write(
        tmp_path,
        {"cases": [{"ok": True}, {"ok": True}, {"ok": False}, {"ok": False}]},
    )
    # 2/4 = 0.5: ниже 0.9 и не ниже 0.5.
    assert _run(report, "0.9").returncode == 1
    assert _run(report, "0.5").returncode == 0


@pytest.mark.parametrize("rate,threshold,expected", [
    (0.95, "0.9", 0),
    (0.90, "0.9", 0),
    (0.89, "0.9", 1),
    (0.0, "0.9", 1),
    (1.0, "1.0", 0),
])
def test_gate_threshold_boundaries(tmp_path: Path, rate: float, threshold: str, expected: int) -> None:
    total = 100
    report = _write(
        tmp_path,
        {"summary": {"total": total, "passed": int(round(rate * total)), "pass_rate": rate}},
    )
    assert _run(report, threshold).returncode == expected


def test_gate_reads_threshold_from_env(tmp_path: Path) -> None:
    """QUALITY_MIN_PASS_RATE - способ поднять порог, не меняя команду в cron."""
    report = _write(tmp_path, {"summary": {"total": 10, "passed": 8, "pass_rate": 0.8}})
    res = subprocess.run(
        [sys.executable, str(GATE), "--report", str(report)],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "QUALITY_MIN_PASS_RATE": "0.95"},
    )
    assert res.returncode == 1, res.stdout
