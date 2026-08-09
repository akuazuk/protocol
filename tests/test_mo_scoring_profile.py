"""Кабинетный профиль жёсткости оценок МО."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.mo_scoring_profile import (
    default_profile,
    load_scoring_profile,
    save_scoring_profile,
    scoring_config_public,
)
from clinical_knowledge.mo_zone_scores import band_for_zone, load_zone_bands


def test_default_profile_standard(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_SCORING_PROFILE_PATH", str(tmp_path / "cfg" / "profile.json"))
    profile = load_scoring_profile(root=tmp_path)
    assert profile["preset"] == "standard"
    assert profile["zone_bands"]["bad_below"] == 50.0
    assert profile["zone_bands"]["ok_at_or_above"] == 85.0


def test_save_strict_preset_updates_zone_bands(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    path = tmp_path / "cfg" / "profile.json"
    monkeypatch.setenv("MO_SCORING_PROFILE_PATH", str(path))
    saved = save_scoring_profile({"preset": "strict"}, actor="tester", root=tmp_path)
    assert saved["preset"] == "strict"
    assert saved["zone_bands"]["bad_below"] == 60.0
    assert path.is_file()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["profile_version"] == 2  # default 1 + save
    load_zone_bands.cache_clear()
    bands = load_zone_bands()
    assert bands["bad_below"] == 60.0
    assert band_for_zone(55.0, [{"score": 1.0}], bands=bands) == "bad"
    assert band_for_zone(92.0, [{"score": 1.0}], bands=bands) == "ok"


def test_custom_knobs_mark_custom(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_SCORING_PROFILE_PATH", str(tmp_path / "p.json"))
    saved = save_scoring_profile(
        {
            "preset": "standard",
            "zone_bands": {"bad_below": 55, "ok_at_or_above": 88},
        },
        actor="tester",
        root=tmp_path,
    )
    assert saved["preset"] == "custom"
    assert saved["zone_bands"]["bad_below"] == 55.0


def test_scoring_config_public_shape(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_SCORING_PROFILE_PATH", str(tmp_path / "p.json"))
    out = scoring_config_public(root=tmp_path)
    assert out["ok"] is True
    assert "effective" in out
    assert "available_days" in out
    assert default_profile()["presets"]["soft"]["label_ru"] == "Мягкая"
