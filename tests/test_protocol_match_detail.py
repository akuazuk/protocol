"""Тесты детального match_score."""
from clinical_knowledge.protocol_match_detail import compute_match_detail


def test_admin_protocol_rejected():
    card = {
        "title": "Об утверждении клинических протоколов",
        "source_path": "minzdrav_protocols/x/order.pdf",
        "icd10_all": [],
    }
    d = compute_match_detail(
        card,
        icd_list=["M54.3"],
        audience="adult",
        hints=set(),
        specialty_slug="nevrologiya",
        diag_text="ишиас",
        complaints=["боль в спине"],
        performed_exams=[],
    )
    assert d["match_score"] == 0.0
    assert d["rejected"] is True
    assert "admin_order" in d["pick_risk_flags"]


def test_match_breakdown_keys():
    card = {
        "title": "Ишиас, радикулопатия",
        "source_path": "minzdrav_protocols/nevrologiya/ishiias.pdf",
        "icd10_all": ["M54.3", "M54.4"],
        "specialty_slug": "nevrologiya",
        "status": "active",
    }
    d = compute_match_detail(
        card,
        icd_list=["M54.3"],
        audience="adult",
        hints={"ишиас"},
        specialty_slug="nevrologiya",
        diag_text="ишиас",
        complaints=["боль в пояснице"],
        performed_exams=[],
    )
    assert d["match_score"] > 20
    assert "icd" in d["match_breakdown"]
    assert d["pick_reason_ru"]
