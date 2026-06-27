"""B2C patient report (tier P1)."""
from __future__ import annotations

from clinical_knowledge.patient_report import (
    build_patient_report,
    block_status_for_score,
    traffic_light_for_pct,
)
from clinical_knowledge.patient_review import (
    patient_demographics_from_form,
    run_patient_review,
)

KZ = """\
Врач: флеболог
Дата консультации: 12.04.2024
Дата рождения: 15.08.1970
Пол: женский
Жалобы: отёк правой голени.
Диагноз: I80.1 Флеботромбоз поверхностных вен нижней конечности.
Рекомендации по лечению: ривароксабан 20 мг 1 раз в день постоянно.
Контроль через 3 месяца.
"""


def test_traffic_light_thresholds() -> None:
    assert traffic_light_for_pct(80)[0] == "green"
    assert traffic_light_for_pct(65)[0] == "yellow"
    assert traffic_light_for_pct(40)[0] == "red"
    assert block_status_for_score(80) == "ok"
    assert block_status_for_score(60) == "attention"
    assert block_status_for_score(30) == "concern"


def test_human_doctor_questions() -> None:
    from clinical_knowledge.patient_report import _gap_to_question

    q = _gap_to_question("Нет длительности терапии", "Лечение", "treatment")
    assert "?" in q
    assert "лечен" in q.lower() or "принимать" in q.lower()
    assert "уточните у врача" not in q.lower()


def test_build_patient_report_from_alignment() -> None:
    l1 = {
        "confidence_score": 85,
        "matched_protocols_count": 2,
        "overall_score": 72,
        "alignment": {
            "alignment_mean_score": 68,
            "alignment_cards": [
                {
                    "block_id": "diagnosis",
                    "name_ru": "Диагноз",
                    "score_pct": 80,
                    "comment_ru": "Код МКБ указан.",
                    "gaps_ru": [],
                    "protocol_excerpt": "При флеботромбозе указывают локализацию и стадию.",
                    "protocol_title": "КП ТГВ",
                    "protocol_path": "minzdrav_protocols/bolezni-sistemy-krovoobrashcheniya/kp_tgv.pdf",
                },
                {
                    "block_id": "treatment",
                    "name_ru": "Лечение",
                    "score_pct": 45,
                    "comment_ru": "Доза антикоагулянта не детализирована.",
                    "gaps_ru": ["Нет длительности терапии"],
                    "protocol_excerpt": "",
                },
            ],
            "limitations_ru": "",
        },
    }
    rep = build_patient_report(l1)
    assert rep["traffic_light"] == "yellow"
    assert rep["overall_pct"] == 68
    assert len(rep["blocks"]) >= 2
    assert rep["questions_for_doctor"]
    assert rep["protocol_citations"]
    assert rep["plain_summary_ru"]
    assert rep["document_quality"]["level"] in ("good", "medium", "low")
    assert rep["action_checklist"]
    assert rep["next_steps_ru"]
    assert rep.get("report_schema_version") == 2
    assert rep.get("headline_ru")
    assert rep.get("protocol_links")
    assert rep["blocks"][0].get("protocol_link")
    assert "не является диагнозом" in rep["disclaimer_ru"].lower()


def test_build_patient_report_with_protocol_context() -> None:
    l1 = {
        "confidence_score": 70,
        "matched_protocols_count": 1,
        "alignment": {
            "alignment_mean_score": 55,
            "alignment_cards": [
                {
                    "block_id": "exams",
                    "name_ru": "Обследования",
                    "score_pct": 30,
                    "comment_ru": "Мало деталей.",
                    "gaps_ru": ["нет УЗИ"],
                },
            ],
        },
    }
    protocol_context = {
        "protocol_title": "КП ТГВ",
        "missing_recommended_exams": [
            {
                "exam_name": "УЗИ вен",
                "severity": "high",
                "patient_note_ru": "По протоколу обычно назначают УЗИ вен.",
            },
        ],
    }
    rep = build_patient_report(l1, protocol_context=protocol_context)
    assert rep["protocol_context"] == protocol_context
    assert rep["priority_topics"]
    assert any("УЗИ" in q for q in rep["questions_for_doctor"])


def test_patient_demographics_from_form() -> None:
    meta = patient_demographics_from_form(age_years="48", sex="male")
    assert meta == {"age_years": 48, "sex": "male"}
    assert patient_demographics_from_form(age_years="", sex="") is None


def test_run_patient_review_integration() -> None:
    out = run_patient_review(text=KZ, consultation_id="t-patient")
    assert out["ok"] is True
    assert out["patient_report"]["traffic_light"] in ("green", "yellow", "red")


def test_run_patient_review_rejects_recipe() -> None:
    recipe = "Рецепт салата. Ингредиенты: помидоры, огурцы. Нарезать, заправить маслом."
    out = run_patient_review(text=recipe, consultation_id="t-recipe")
    assert out.get("upload_mismatch") is True
    pr = out["patient_report"]
    assert pr.get("upload_joke")
    assert pr.get("upload_mismatch") is True
    assert not pr.get("blocks")
    assert out.get("matched_protocols_count") == 0


def test_run_patient_review_rejects_protocol_pdf() -> None:
    protocol = (
        "КЛИНИЧЕСКИЙ ПРОТОКОЛ диагностики и лечения. "
        "УТВЕРЖДЕН приказом Министерства здравоохранения Республики Беларусь. "
        "1. Общие положения. 2. Диагностика. Рекомендуется осмотр."
    )
    out = run_patient_review(text=protocol, consultation_id="t-protocol-pdf")
    assert out.get("upload_mismatch") is True
    assert out.get("matched_protocols_count") == 0
    pr = out["patient_report"]
    assert pr.get("upload_joke")
    assert not pr.get("protocol_links")
