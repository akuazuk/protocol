from clinical_knowledge.gemini_model_config import resolve_gemini_model


def test_alias_gemini_3_1_pro_to_25_pro():
    name, warn = resolve_gemini_model("gemini-3.1-pro")
    assert name == "gemini-2.5-pro"
    assert warn


def test_strip_models_prefix():
    name, _ = resolve_gemini_model("models/gemini-2.5-flash")
    assert name == "gemini-2.5-flash"


def test_known_model_passthrough():
    name, warn = resolve_gemini_model("gemini-2.5-flash")
    assert name == "gemini-2.5-flash"
    assert warn is None
