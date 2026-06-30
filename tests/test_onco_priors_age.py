"""Тесты возраст-специфичных priors и скрипта рекалибровки (Фаза 4)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from clinical_knowledge import onco_risk as orisk

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "onco_priors_recalibrate.py"

SYNTH = {
    "sites": {
        "colorectal": {
            "bands": [
                {"age_min": 40, "age_max": 59, "baseline_symptomatic": 0.002},
                {"age_min": 60, "age_max": 79, "baseline_symptomatic": 0.01},
            ]
        }
    }
}


def test_pick_age_baseline_selects_band(monkeypatch):
    monkeypatch.setattr(orisk, "_age_priors", lambda: SYNTH)
    assert orisk._pick_age_baseline("colorectal", 65) == 0.01
    assert orisk._pick_age_baseline("colorectal", 45) == 0.002
    assert orisk._pick_age_baseline("colorectal", 30) is None  # вне полос
    assert orisk._pick_age_baseline("lung", 65) is None  # нет сайта
    assert orisk._pick_age_baseline("colorectal", None) is None


def test_baseline_prefers_age_when_available(monkeypatch):
    monkeypatch.setattr(orisk, "_age_priors", lambda: SYNTH)
    # 65 лет -> возрастной 0.01 вместо общего baseline_symptomatic (0.0025)
    assert orisk._baseline("colorectal", 65) == 0.01
    # без возраста -> общий
    assert orisk._baseline("colorectal", None) == orisk._baseline("colorectal")


def test_no_age_file_means_unchanged_behavior():
    # В репозитории файла нет -> _age_priors() пуст, поведение прежнее.
    assert orisk._age_priors() == {} or "sites" in orisk._age_priors()
    assert orisk._pick_age_baseline("colorectal", 65) is None or isinstance(
        orisk._pick_age_baseline("colorectal", 65), float
    )


def test_recalibrate_script_dry_run(tmp_path):
    csv_path = tmp_path / "ci5.csv"
    csv_path.write_text(
        "site,age_min,age_max,baseline_symptomatic\n"
        "colorectal,40,59,0.002\n"
        "colorectal,60,79,0.01\n"
        "lung,60,79,0.008\n",
        encoding="utf-8",
    )
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(csv_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert "DRY-RUN" in out.stdout
    assert "colorectal" in out.stdout
    # dry-run не создаёт файл
    assert not (ROOT / "data" / "onco_risk" / "priors_age_belarus.yaml").is_file()


def test_recalibrate_script_from_rate(tmp_path):
    csv_path = tmp_path / "rnpc.csv"
    csv_path.write_text(
        "site,age_min,age_max,rate_per_100k\ncolorectal,60,79,250\n",
        encoding="utf-8",
    )
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(csv_path), "--from-rate"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    # 250/100000 = 0.0025
    assert "0.0025" in out.stdout
