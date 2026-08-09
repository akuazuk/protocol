"""Section-pack оценка №55 (прил.1 разд. V)."""
from __future__ import annotations

from clinical_knowledge.mo_reg55_section import (
    band_from_pct,
    evaluate_reg55_section,
    load_section_config,
    resolve_section_pack,
)

# Сжатый клинический снимок clients_consult/mo_1_test.pdf (педиатр, первичный).
MO_1_TEST_CASE = {
    "doctor_specialization": "Педиатр",
    "patient_age_years": 8.1,
    "complaints": (
        "на кашель сухой, насморк. Борозды на ногтях. Головная боль, чаще по вечерам. "
        "Неправильная установка стоп при ходьбе. Набор веса неуточненный, по питанию"
    ),
    "anamnesis_doctor": (
        "Семейная АГ,гиперхолестеринемия. Хронические заболевания: Бронхиальная астма с 4х лет"
        "(ремиссия 1 год). Перенсененные заболевания: ОРВИ, ветряная оспа не болел. "
        "Аллергические реакции: нет. Травмы и операции: нет. Проф.прививки: по возрасту (прилагаются)."
    ),
    "objective_status": (
        "Вес 39.7 кг. Рост 134 см. ИМТ 22 кг/м2. Частота дыхания 24. Температура тела 36,4 °С. "
        "АД 100/60 мм рт.ст. ЧСС 88. Состояние удовлетворительное. Кожные покровы без изменения. "
        "Дыхание везикулярное, хрипы не выслушиваются. Живот мягкий, безболезненный."
    ),
    "clinical_diagnosis": (
        "J45 Бронхиальная астма,аллергическая, легкое персистирующее течение,контролируемая. "
        "Персистирующий аллергический ринит. М21.4 Плоско-вальгусная установка стоп? "
        "Е 55.0 Дефицит витамина Д? G44.2 Головная боль напряжения?"
    ),
    "exam_recommendations": (
        "1.Аллергопанель ингаляционная+ IG E,консультация аллерголога,спирограмма "
        "2. консультация невролога,офтальмолога,ортопеда(хирурга) "
        "3.ОАК,ОАМ,кровь на глюкозу,ЭКГ "
        "4. витамин Д, БАК(...) 5.Явка с результатами анализов"
    ),
    "treatment_recommendations": (
        "витамин Д, БАК(...). Явка с результатами анализов"
    ),
    "raw_text": (
        "На основании частей первой и второй статьи 44 Закона Республики Беларусь "
        "пациент устно проинформирован о необходимости проведения простых диагностических "
        "вмешательств и от пациента получено устное информированное добровольное согласие "
        "на проведение диагностических вмешательств. МЕДИЦИНСКИЙ ОСМОТР ПЕДИАТР"
    ),
    "block_scores": {},
}


def test_config_has_pediatrist_and_bands() -> None:
    cfg = load_section_config()
    assert "pediatrist_district" in cfg["packs"]
    assert "gp" in cfg["packs"]
    assert "specialist_amb_core" in cfg["packs"]
    assert cfg["bands"]["compliant_min"]["min_pct"] == 80.0
    ped = cfg["packs"]["pediatrist_district"]["criteria"]
    points = {c["point"] for c in ped}
    assert "42.3" in points and "43.6" in points


def test_router_pediatrist() -> None:
    info = resolve_section_pack({"doctor_specialization": "Врач-педиатр"})
    assert info["pack_id"] == "pediatrist_district"


def test_router_specialist_fallback() -> None:
    info = resolve_section_pack({"doctor_specialization": "Кардиолог"})
    assert info["pack_id"] == "specialist_amb_core"


def test_band_from_pct_matches_instruction() -> None:
    assert band_from_pct(80)["code"] == "compliant_min"
    assert band_from_pct(79.9)["code"] == "compliant_measures"
    assert band_from_pct(55)["code"] == "compliant_measures"
    assert band_from_pct(54.9)["code"] == "noncompliant"
    assert band_from_pct(None)["code"] == "unscored"


def test_mo_1_test_pediatrist_section_score() -> None:
    result = evaluate_reg55_section(MO_1_TEST_CASE)
    assert result["ok"] is True
    assert result["pack_id"] == "pediatrist_district"
    assert result["applicable_n"] >= 8
    assert result["na_n"] >= 8
    # Ожидание калибровки на mo_1_test: зона 55-79.9 (как ручной разбор ~70%).
    assert result["reg55_section_pct"] is not None
    assert 55.0 <= float(result["reg55_section_pct"]) < 80.0
    assert result["reg55_band"] == "compliant_measures"
    by_point = {c["point"]: c for c in result["criteria"]}
    assert by_point["42.5"]["score"] is None  # primary → n/a correction
    assert by_point["42.8"]["score"] is None  # age gate
    assert by_point["43.4"]["score"] == 1.0  # consent
    assert by_point["43.6"]["evidence_from_127"] is True
    # №127 helper не создаёт отдельный %
    assert "reg127_pct" not in result


def test_evidence_127_does_not_change_denominator() -> None:
    result = evaluate_reg55_section(MO_1_TEST_CASE)
    applicable = [c for c in result["criteria"] if c["applicable"]]
    # знаменатель = только applicable пункты №55
    assert result["applicable_n"] == len(applicable)
    with_127 = [c for c in applicable if c.get("evidence_from_127")]
    assert with_127  # есть пункты с опорой на №127
    # все они уже в applicable_n, отдельного множителя нет
    assert result["reg55_section_pct"] == round(
        100.0 * sum(float(c["score"]) for c in applicable) / len(applicable), 1
    )
