from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from clinical_knowledge.mo_calibration_methodist_ui import (
    load_review_pack,
    save_label,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture()
def c6_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "calibration"
    review = root / "secret" / "methodist"
    cases = []
    labels = []
    pilot = []
    for index in range(1, 16):
        sample_id = f"S{index:03d}"
        cases.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "required_endpoints": ["dx"],
                "clinical_case": {
                    "sample_id": sample_id,
                    "meta": {"specialty": "Терапевт"},
                    "evidence": {"complaints": "жалобы"},
                    "diagnosis": {"text": "диагноз", "icd": "I10"},
                    "plan": {},
                },
                "plan_route": {"route": "llm_no_kp"},
                "protocol_context": None,
            }
        )
        labels.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "endpoint": "dx",
                "verdict": None,
                "score_pct": None,
                "potential_harm": None,
                "icd_fit": None,
                "confidence": None,
                "rationale": "",
                "reviewer_id": "",
                "reviewed_at": "",
            }
        )
        for pass_no in (1, 2):
            pilot.append(
                {
                    "kind": "pass",
                    "sample_id": sample_id,
                    "pass_no": pass_no,
                    "error": None,
                    "dx_evidence": {"verdict": "partial", "dx_evidence_pct": 60},
                }
            )
        pilot.append(
            {
                "kind": "adjudication",
                "sample_id": sample_id,
                "endpoint": "dx",
                "result": {"verdict": "partial", "dx_evidence_pct": 60},
            }
        )
    _write_jsonl(review / "methodist_cases.jsonl", cases)
    _write_jsonl(review / "methodist_labels.jsonl", labels)
    _write_jsonl(root / "secret" / "blind_pilot.jsonl", pilot)
    (root / "methodist_status.json").write_text(
        json.dumps({"schema_version": 1, "artifact_hashes": {}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MO_CALIBRATION_C6_ROOT", str(root))
    return root


def _save(sample_id: str) -> dict:
    return save_label(
        sample_id=sample_id,
        endpoint="dx",
        verdict="partial",
        score_pct=65,
        potential_harm=False,
        icd_fit="partial",
        confidence=0.8,
        rationale="Диагноз подтверждён только частью данных.",
        reviewer_id="user:methodist",
    )


def test_pack_load_exposes_only_blinded_cases(c6_root: Path) -> None:
    stale_comparison = c6_root / "secret" / "methodist_llm_comparison_unsealed.jsonl"
    stale_comparison.write_text("{}\n", encoding="utf-8")
    result = load_review_pack(actor="user:methodist", role="methodist")
    assert len(result["items"]) == 15
    assert result["status"]["complete_label_n"] == 0
    serialized = json.dumps(result, ensure_ascii=False)
    assert "pass_1" not in serialized
    assert "llm_adjudication" not in serialized
    assert "case_id" not in serialized
    assert stale_comparison.exists() is False
    audit_path = c6_root / "secret" / "methodist" / "methodist_access_audit.jsonl"
    audit_text = audit_path.read_text(encoding="utf-8")
    assert "calibration_pack_open" in audit_text
    assert "жалобы" not in audit_text
    assert os.stat(audit_path).st_mode & 0o777 == 0o600


def test_save_is_atomic_and_unseals_only_after_gate(c6_root: Path) -> None:
    comparison = c6_root / "secret" / "methodist_llm_comparison_unsealed.jsonl"
    first = _save("S001")
    assert first["status"]["complete_label_n"] == 1
    assert first["comparison_unsealed"] is False
    assert comparison.exists() is False
    for index in range(2, 16):
        result = _save(f"S{index:03d}")
    assert result["status"]["passed"] is True
    assert result["comparison_unsealed"] is True
    assert comparison.is_file()
    assert len(comparison.read_text(encoding="utf-8").splitlines()) == 15
    labels = c6_root / "secret" / "methodist" / "methodist_labels.jsonl"
    assert os.stat(labels).st_mode & 0o777 == 0o600
    saved = load_review_pack()["items"][0]["labels"]["dx"]
    assert saved["reviewer_id"] == "user:methodist"
    assert saved["reviewed_at"].endswith("Z")
    access_rows = (
        c6_root / "secret" / "methodist" / "methodist_access_audit.jsonl"
    ).read_text(encoding="utf-8")
    assert "calibration_label_save" in access_rows


def test_invalid_plan_icd_fit_is_rejected(c6_root: Path) -> None:
    with pytest.raises(ValueError, match="plan_icd_fit_must_be_na"):
        save_label(
            sample_id="S001",
            endpoint="plan",
            verdict="partial",
            score_pct=50,
            potential_harm=False,
            icd_fit="fit",
            confidence=0.8,
            rationale="Достаточное клиническое обоснование.",
            reviewer_id="methodist",
        )


def test_stale_label_write_is_rejected(c6_root: Path) -> None:
    saved = _save("S001")
    assert saved["label"]["reviewed_at"]
    with pytest.raises(ValueError, match="label_changed_by_another_reviewer"):
        _save("S001")


def test_api_requires_methodist_auth_and_never_caches(
    c6_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    import rag_server

    monkeypatch.setenv("METHODIST_TOKEN", "test-methodist-token")
    client = TestClient(rag_server.app)
    denied = client.get("/api/methodist/mo/calibration/c6")
    assert denied.status_code == 403
    headers = {"X-Methodist-Token": "test-methodist-token"}
    viewer = client.get(
        "/api/methodist/mo/calibration/c6",
        headers={**headers, "X-Methodist-Role": "viewer"},
    )
    assert viewer.status_code == 403
    response = client.get("/api/methodist/mo/calibration/c6", headers=headers)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    saved = client.put(
        "/api/methodist/mo/calibration/c6/labels/S001/dx",
        headers=headers,
        json={
            "verdict": "partial",
            "score_pct": 65,
            "potential_harm": False,
            "icd_fit": "partial",
            "confidence": 0.8,
            "rationale": "Диагноз подтверждён только частью данных.",
        },
    )
    assert saved.status_code == 200
    assert saved.headers["cache-control"] == "no-store"
    assert saved.json()["label"]["reviewer_id"]


def test_calibration_page_and_assets_are_served(c6_root: Path) -> None:
    from fastapi.testclient import TestClient

    import rag_server

    client = TestClient(rag_server.app)
    page = client.get("/methodist/calibration")
    assert page.status_code == 200
    assert "no-store" in page.headers["cache-control"]
    assert 'id="label-form"' in page.text
    assert "LLM и engine scores скрыты" in page.text
    assert client.get("/mo-calibration.css").status_code == 200
    assert client.get("/mo-calibration.js").status_code == 200


def test_methodist_navigation_links_to_calibration() -> None:
    root = Path(__file__).resolve().parents[1]
    cabinet = (root / "frontend/web/doctor/index.html").read_text(encoding="utf-8")
    assert 'id="methodist-nav-calibration"' in cabinet
    assert 'href="/methodist/calibration"' in cabinet
