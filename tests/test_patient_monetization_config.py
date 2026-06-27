"""Настройки монетизации B2C."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_knowledge.patient_monetization_config import (
    load_patient_monetization_config,
    monetization_public_view,
    payment_required_effective,
    save_patient_monetization_config,
    tier_catalog_for_patient,
)


@pytest.fixture()
def monetization_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "patient_monetization.json"
    monkeypatch.setenv("PATIENT_MONETIZATION_CONFIG", str(p))
    return p


def test_default_free_mode(monetization_config_path: Path) -> None:
    cfg = load_patient_monetization_config()
    assert cfg["monetization_enabled"] is False
    assert payment_required_effective() is False
    pub = monetization_public_view()
    assert pub["payment_required"] is False
    assert pub["tiers"]


def test_save_and_require_payment(monetization_config_path: Path) -> None:
    save_patient_monetization_config(
        {"monetization_enabled": True, "payment_required": True},
        reviewer="Test M",
    )
    assert monetization_config_path.is_file()
    raw = json.loads(monetization_config_path.read_text(encoding="utf-8"))
    assert raw["monetization_enabled"] is True
    assert payment_required_effective() is True
    pub = monetization_public_view()
    assert pub["monetization_enabled"] is True
    assert pub["show_tier_picker"] is True


def test_enabled_tier_filter(monetization_config_path: Path) -> None:
    save_patient_monetization_config(
        {
            "monetization_enabled": True,
            "enabled_tier_ids": ["promo", "basic"],
            "default_tier_id": "promo",
        },
        reviewer="Test",
    )
    tiers = tier_catalog_for_patient()
    ids = [t["tier_id"] for t in tiers]
    assert ids == ["promo", "basic"]
    assert tiers[0].get("hint_ru")


def test_env_payment_when_monetization_on(monetization_config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_patient_monetization_config({"monetization_enabled": True, "payment_required": False})
    monkeypatch.setenv("PATIENT_PAYMENT_REQUIRED", "1")
    assert payment_required_effective() is True


def test_methodist_api_get_put(monetization_config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import rag_server

    monkeypatch.setenv("METHODIST_TOKEN", "test-methodist-token")
    client = TestClient(rag_server.app)
    r = client.get(
        "/api/methodist/patient-monetization",
        headers={"X-Methodist-Token": "test-methodist-token", "X-Methodist-Reviewer": "T"},
    )
    assert r.status_code == 200
    cfg = r.json()["config"]
    assert "tier_catalog_all" in cfg
    r2 = client.put(
        "/api/methodist/patient-monetization",
        headers={"X-Methodist-Token": "test-methodist-token", "X-Methodist-Reviewer": "T"},
        json={"monetization_enabled": True, "payment_required": False, "enabled_tier_ids": ["basic", "plus"]},
    )
    assert r2.status_code == 200
    saved = r2.json()["config"]
    assert saved["monetization_enabled"] is True
    assert "plus" in saved["enabled_tier_ids"]
