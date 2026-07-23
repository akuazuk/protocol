from clinical_knowledge.gemini_model_config import resolve_gemini_model


def test_alias_gemini_3_1_pro_to_31_pro_preview():
    # 3.x pro без суффикса -preview → реальная доступная 3.1-pro-preview (проверено Render).
    name, warn = resolve_gemini_model("gemini-3.1-pro")
    assert name == "gemini-3.1-pro-preview"
    assert warn


def test_gemini_36_flash_passthrough():
    name, warn = resolve_gemini_model("gemini-3.6-flash")
    assert name == "gemini-3.6-flash"
    assert warn is None


def test_gemini_3_pro_preview_retired_maps_to_31():
    name, warn = resolve_gemini_model("gemini-3-pro-preview")
    assert name == "gemini-3.1-pro-preview"
    assert warn


def test_strip_models_prefix():
    name, _ = resolve_gemini_model("models/gemini-2.5-flash")
    assert name == "gemini-2.5-flash"


def test_known_model_passthrough():
    name, warn = resolve_gemini_model("gemini-2.5-flash")
    assert name == "gemini-2.5-flash"
    assert warn is None
