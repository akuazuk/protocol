"""Все клинические вызовы Gemini идут с одинаковыми порогами безопасности.

Регресс (2026-09-05): клиент Gemini собирался в шести местах, а
`safety_settings` выставляли только три из них. Без них действует дефолт
BLOCK_MEDIUM_AND_ABOVE, который на клиническом тексте (травмы,
самоповреждения, репродуктивное здоровье, онкология) блокирует ответ - модель
возвращает пустоту, и результат записывается битым без ошибки в логе.

Хуже всего это было в `scripts/grade_kz_llm.py`: оценка качества
консультативных заключений создавала `genai.GenerativeModel(name)` без
порогов, то есть именно клинический контур работал на дефолте.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tracked_py_files() -> list[Path]:
    raw = subprocess.run(
        ["git", "ls-files", "-z", "*.py"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    out = []
    for chunk in raw.split(b"\x00"):
        if not chunk:
            continue
        rel = chunk.decode("utf-8", "replace")
        # archive/ - завершённые задачи, к клиническому контуру не относятся.
        if rel.startswith("archive/"):
            continue
        out.append(ROOT / rel)
    return out


def test_generative_model_is_only_built_by_shared_client() -> None:
    """`genai.GenerativeModel(...)` допустим только в общем клиенте.

    Так пороги безопасности физически невозможно забыть: любой новый вызов
    обязан пройти через `clinical_knowledge/gemini_client.build_model`.
    """
    allowed = {ROOT / "clinical_knowledge" / "gemini_client.py"}
    offenders: list[str] = []

    for path in tracked_py_files():
        if path in allowed:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "GenerativeModel(" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else (func.id if isinstance(func, ast.Name) else None)
            )
            if name == "GenerativeModel":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert not offenders, (
        "GenerativeModel создаётся в обход общего клиента - пороги безопасности "
        "уедут на дефолт BLOCK_MEDIUM_AND_ABOVE, и клинический текст начнёт "
        "блокироваться:\n  " + "\n  ".join(offenders)
    )


def test_safety_settings_cover_all_four_categories() -> None:
    from clinical_knowledge.gemini_client import clinical_safety_settings

    try:
        settings = clinical_safety_settings()
    except ImportError:
        pytest.skip("google.generativeai не установлен")

    assert len(settings) == 4, "ожидаются все четыре категории Harm*"
    # HarmCategory/HarmBlockThreshold - целочисленные enum: str() даёт номер,
    # поэтому сверяем по .name.
    categories = {s["category"].name for s in settings}
    for expected in ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"):
        assert any(expected in c for c in categories), f"нет категории {expected}: {categories}"

    # Порог именно BLOCK_ONLY_HIGH: смысл настройки в том, чтобы пропускать
    # клиническую лексику и резать только заведомо опасное.
    thresholds = {s["threshold"].name for s in settings}
    assert thresholds == {"BLOCK_ONLY_HIGH"}, f"пороги разъехались: {thresholds}"


def test_api_key_order_is_stable(monkeypatch) -> None:
    """GOOGLE_API_KEY имеет приоритет над GEMINI_API_KEY.

    Порядок сложился во всех прежних местах; менять его нельзя, иначе на
    машинах с двумя переменными молча поменяется используемый ключ.
    """
    from clinical_knowledge import gemini_client

    monkeypatch.setenv("GOOGLE_API_KEY", "primary")
    monkeypatch.setenv("GEMINI_API_KEY", "secondary")
    assert gemini_client.api_key() == "primary"

    monkeypatch.delenv("GOOGLE_API_KEY")
    assert gemini_client.api_key() == "secondary"

    monkeypatch.delenv("GEMINI_API_KEY")
    assert gemini_client.api_key() is None
    assert not gemini_client.available()
    with pytest.raises(gemini_client.GeminiKeyMissing):
        gemini_client.require_api_key()


def test_blank_key_is_treated_as_missing(monkeypatch) -> None:
    """Пустая или пробельная переменная - это отсутствие ключа, а не ключ."""
    from clinical_knowledge import gemini_client

    monkeypatch.setenv("GOOGLE_API_KEY", "   ")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert gemini_client.api_key() is None


def test_build_model_passes_safety_and_key(monkeypatch) -> None:
    """build_model обязан передать safety_settings и выбранный ключ."""
    from clinical_knowledge import gemini_client

    captured: dict[str, object] = {}

    class FakeGenai:
        @staticmethod
        def configure(*, api_key):
            captured["api_key"] = api_key

        @staticmethod
        def GenerativeModel(name, **kwargs):  # noqa: N802 - имя из SDK
            captured["name"] = name
            captured["kwargs"] = kwargs
            return object()

    monkeypatch.setattr(gemini_client, "_import_genai", lambda: FakeGenai)
    monkeypatch.setattr(
        gemini_client, "clinical_safety_settings", lambda: [{"category": "c", "threshold": "t"}]
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")

    gemini_client.build_model("gemini-test")
    assert captured["api_key"] == "from-env"
    assert captured["name"] == "gemini-test"
    assert captured["kwargs"]["safety_settings"], "safety_settings не переданы"

    # Ротация ключей в методистском контуре передаёт ключ явно.
    gemini_client.build_model("gemini-test", api_key_override="rotated")
    assert captured["api_key"] == "rotated"
