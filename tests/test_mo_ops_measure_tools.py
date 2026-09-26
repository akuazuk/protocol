"""Инструменты измерения МО Аналитики (план 2026-09-26, волна T).

Проверяется, что скрипты собираются, не печатают тела ответов / PHI и
считают метрики плана на маленьком складе-фикстуре.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "scripts" / "ops"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probe_list_covers_hot_endpoints_and_hides_query_text() -> None:
    probe = _load("mo_api_latency_probe")
    names = {p[0] for p in probe.PROBES}
    assert {"cases_month_p1", "cases_queue_only", "cases_sort_score", "facets", "freshness", "score_dashboard"} <= names
    for _name, _path, _params, threshold in probe.PROBES:
        assert 100 <= threshold <= 2000
    src = (OPS / "mo_api_latency_probe.py").read_text(encoding="utf-8")
    assert '"q": "<hidden>"' in src
    assert "X-Methodist-Token" in src
    assert "print(body" not in src


def test_probe_month_bounds() -> None:
    probe = _load("mo_api_latency_probe")
    assert probe._month_bounds("2026-02") == ("2026-02-01", "2026-02-28")
    # Текущий и будущий месяц обрезаются по «вчера» (Минск): API иначе отвечает 422.
    first, last = probe._month_bounds(probe._yesterday_minsk().strftime("%Y-%m"))
    assert last == probe._yesterday_minsk().isoformat() or last == first
    future = (probe._yesterday_minsk() + dt.timedelta(days=40)).strftime("%Y-%m")
    assert probe._month_bounds(future)[1] == probe._month_bounds(future)[0]


def test_probe_compare_flags_regression(tmp_path: Path) -> None:
    probe = _load("mo_api_latency_probe")
    before = {"results": [{"name": "cases_month_p1", "warm_ms": 900}, {"name": "facets", "warm_ms": 300}]}
    after = {"results": [{"name": "cases_month_p1", "warm_ms": 950}, {"name": "facets", "warm_ms": 700}], "failures": ["facets"]}
    (tmp_path / "b.json").write_text(json.dumps(before), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps(after), encoding="utf-8")
    assert probe.compare(str(tmp_path / "b.json"), str(tmp_path / "a.json"), 20.0) == 1
    after["results"][1]["warm_ms"] = 320
    after["failures"] = []
    (tmp_path / "a.json").write_text(json.dumps(after), encoding="utf-8")
    assert probe.compare(str(tmp_path / "b.json"), str(tmp_path / "a.json"), 20.0) == 0


def _fixture_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE fact_mo_case (
            mis_id TEXT, visit_date TEXT, document_kind TEXT, overall_grade TEXT,
            zone1_band TEXT, zone2a_band TEXT, zone2b_band TEXT, zone2b_kp_status TEXT,
            attention_primary TEXT, reg55_band TEXT, diagnosis_code TEXT, diagnosis_text TEXT
        );
        CREATE TABLE fact_mo_finding (mis_id TEXT, finding_code TEXT, severity TEXT);
        CREATE TABLE dim_diagnosis (diagnosis_key TEXT, diagnosis_label TEXT);
        """
    )
    rows = []
    for i in range(20):
        grade = "fair" if i < 12 else ("poor" if i < 19 else "good")
        z1 = "weak" if i < 15 else ("bad" if i < 18 else "ok")
        z2b = "na" if i < 13 else ("weak" if i < 17 else "bad")
        rows.append((f"m{i}", "2026-09-%02d" % (i % 28 + 1), "clinical_visit", grade, z1, "ok", z2b,
                     "unmatched" if z2b == "na" else "matched", "none", "compliant", "I10" if i % 3 else "", "dx text"))
    rows.append(("c1", "2026-03-05", "consultation", None, None, None, None, None, None, None, "", ""))
    conn.executemany("INSERT INTO fact_mo_case VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.executemany("INSERT INTO fact_mo_finding VALUES (?,?,?)", [("m1", "D_reg55_gap", "P2"), ("m2", "D_reg55_gap", "P2"), ("m2", "C_ddi", "P1")])
    conn.commit()
    conn.close()


def test_warehouse_profile_metrics_on_fixture(tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    _fixture_db(db)
    profile = _load("mo_warehouse_profile")
    doc = profile.profile(str(db), None, "2026-09")
    metrics = doc["metrics"]
    # 12 fair + 7 poor из 20 = 95 %
    assert metrics["top2_overall_grade_share_pct"] == 95.0
    assert metrics["zone1_weak_pct"] == 75.0
    assert metrics["zone2b_na_pct"] == 65.0
    assert metrics["clinical_months_with_zones"] == 1
    assert doc["findings_top"][0]["finding_code"] == "D_reg55_gap"
    text = json.dumps(doc, ensure_ascii=False)
    assert "dx text" not in text  # тексты диагнозов в профиль не попадают
    assert "m1" not in json.dumps(doc["metrics"])


def test_warehouse_profile_cli_writes_metrics(tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    _fixture_db(db)
    out = tmp_path / "profile.json"
    res = subprocess.run(
        [sys.executable, str(OPS / "mo_warehouse_profile.py"), "--db", str(db), "--no-lab", "--month", "2026-09", "--out", str(out)],
        capture_output=True, text=True, check=False, cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists()
    assert "top2_overall_grade_share_pct" in res.stdout


def test_dom_audit_script_is_valid_js_and_has_no_screenshots() -> None:
    src = (OPS / "mo_ui_dom_audit.mjs").read_text(encoding="utf-8")
    assert "screenshot" not in src.lower()
    assert 'details:not([open])' in src
    assert "#app-nav .nav-button[hidden]" in src
    assert "document.fonts" in src
    assert "protocol_methodist_token" in src
    res = subprocess.run(["node", "--check", str(OPS / "mo_ui_dom_audit.mjs")], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
