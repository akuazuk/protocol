from __future__ import annotations

from pathlib import Path

from backend import frontend_paths


def test_frontend_file_fallback_to_root(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "index.html").write_text("root-index", encoding="utf-8")
    (tmp_path / "frontend" / "web").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(frontend_paths, "ROOT", tmp_path)
    monkeypatch.setattr(frontend_paths, "FRONTEND_WEB_ROOT", tmp_path / "frontend" / "web")

    p = frontend_paths.frontend_file("index.html")
    assert p == tmp_path / "index.html"
    assert p.read_text(encoding="utf-8") == "root-index"


def test_frontend_file_prefers_frontend_web(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "index.html").write_text("root-index", encoding="utf-8")
    web = tmp_path / "frontend" / "web"
    (web / "doctor").mkdir(parents=True, exist_ok=True)
    (web / "doctor" / "index.html").write_text("web-index", encoding="utf-8")
    monkeypatch.setattr(frontend_paths, "ROOT", tmp_path)
    monkeypatch.setattr(frontend_paths, "FRONTEND_WEB_ROOT", web)

    p = frontend_paths.frontend_file("index.html")
    assert p == web / "doctor" / "index.html"
    assert p.read_text(encoding="utf-8") == "web-index"


def test_has_frontend_file_false_when_missing(tmp_path: Path, monkeypatch) -> None:
    web = tmp_path / "frontend" / "web"
    web.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(frontend_paths, "ROOT", tmp_path)
    monkeypatch.setattr(frontend_paths, "FRONTEND_WEB_ROOT", web)

    assert frontend_paths.has_frontend_file("missing.file") is False


def test_patient_asset_prefers_canonical_grouped_path(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "patient.html").write_text("legacy-patient", encoding="utf-8")
    web = tmp_path / "frontend" / "web" / "patient"
    web.mkdir(parents=True, exist_ok=True)
    (web / "patient.html").write_text("grouped-patient", encoding="utf-8")
    monkeypatch.setattr(frontend_paths, "ROOT", tmp_path)
    monkeypatch.setattr(frontend_paths, "FRONTEND_WEB_ROOT", tmp_path / "frontend" / "web")

    p = frontend_paths.frontend_file("patient.html")
    assert p == web / "patient.html"
    assert p.read_text(encoding="utf-8") == "grouped-patient"

