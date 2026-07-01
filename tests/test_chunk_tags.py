"""Тесты меток чанков и kz_match."""
from clinical_knowledge.chunk_tags import (
    build_chunk_tags,
    build_protocol_tags,
    chunk_usable_for_retrieval,
    is_administrative_text,
    is_chunk_preamble,
)
from clinical_knowledge.kz_chunk_match import match_kp_item_to_kz


def test_preamble_detected():
    text = "Постановление Министерства здравоохранения об утверждении клинических протоколов"
    assert is_chunk_preamble(text) is True
    tags = build_chunk_tags(text=text, chunk_type="body")
    assert tags["is_preamble"] is True
    assert tags["signal"] == "low"


def test_clinical_chunk_usable():
    text = "Рекомендуется выполнить МРТ поясничного отдела позвоночника при стойком болевом синдроме"
    tags = build_chunk_tags(text=text, chunk_type="diagnostics", icd_codes=["M54.3"])
    ch = {"chunk_type": "diagnostics", "tags": tags, "text": text, "icd10_codes": ["M54.3"]}
    assert chunk_usable_for_retrieval(ch) is True
    assert tags["obligation"] in ("required", "recommended")


def test_protocol_tags_admin():
    tags = build_protocol_tags(
        title="Об утверждении клинических протоколов медицинской помощи",
        source_path="minzdrav_protocols/x/Об_утверждении.pdf",
        protocol_kind="order",
        icd_codes=[],
        chunk_type_counts={"body": 10},
    )
    assert tags["admin_order"] is True
    assert tags["usable_for_kz_review"] is False


def test_administrative_chunk_not_usable():
    # Список утверждаемых протоколов (преамбула приказа), мис-типизирован как drug_list.
    approval = (
        "клинический протокол «Диагностика и лечение пациентов (взрослое население) "
        "с папулосквамозными нарушениями» (прилагается);"
    )
    assert is_administrative_text(approval) is True
    assert chunk_usable_for_retrieval({"chunk_type": "drug_list", "text": approval}) is False

    # Подпись министра / ссылка на приложение приказа в конце куска.
    tail = (
        "«Дискоидная красная волчанка (L93.0).» приложения 3 к приказу Министерства "
        "здравоохранения Республики Беларусь. Министр Д.Л.Пиневич"
    )
    assert is_administrative_text(tail) is True
    assert chunk_usable_for_retrieval({"chunk_type": "appendix", "text": tail}) is False

    # Реальная клиника не должна отсекаться.
    clinical = "Рекомендуется топический кальципотриол/бетаметазон при бляшечном псориазе."
    assert is_administrative_text(clinical) is False
    assert chunk_usable_for_retrieval({"chunk_type": "treatment", "text": clinical}) is True


def test_kz_match_found():
    m = match_kp_item_to_kz("МРТ поясничного отдела", "Назначено: МРТ поясничного отдела позвоночника")
    assert m["kz_match"] in ("found", "partial")
