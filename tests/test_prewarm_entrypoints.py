"""Прогрев кешей вызывается только из фоновой загрузки, где обёрнут в except.

В conftest фоновой прогрев выключен (27 с на 477 сводок, под coverage - 90 с,
сессия упиралась в таймаут). Поэтому точки входа проверяем здесь напрямую:
иначе ошибка в них не всплывёт ни в одном тесте, а в проде превратится в
молчаливую потерю прогрева и медленный первый запрос.
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

import pytest

from clinical_knowledge.protocol_summary import loader as summary_loader

ROOT = Path(__file__).resolve().parent.parent


def _clear_module_caches() -> None:
    """Сбросить все lru_cache в loader, чтобы не утащить фикстуру в другие тесты."""
    for obj in vars(summary_loader).values():
        if callable(obj) and hasattr(obj, "cache_clear"):
            obj.cache_clear()


@pytest.fixture
def tiny_summaries_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Две настоящие сводки вместо 477: путь прогрева тот же, цена - миллисекунды."""
    real_yaml = sorted((ROOT / "data" / "protocol_summaries" / "yaml").glob("*.yaml"))[:2]
    assert len(real_yaml) == 2, "нужны образцы сводок в data/protocol_summaries/yaml"

    dest = tmp_path / "yaml"
    dest.mkdir(parents=True)
    for src in real_yaml:
        shutil.copy2(src, dest / src.name)

    monkeypatch.setattr(summary_loader, "_data_root", lambda: tmp_path)
    _clear_module_caches()
    try:
        yield tmp_path
    finally:
        # Кеши держат сводки из tmp_path - без сброса следующий тест увидит фикстуру.
        _clear_module_caches()


def test_prewarm_protocol_summaries_loads_and_indexes(tiny_summaries_dir: Path) -> None:
    assert summary_loader.prewarm_protocol_summaries() == 2

    # Кеши читаем через cache_info, не вызывая их: любой вызов наполнит кеш сам
    # и тест перестанет отличать реальный прогрев от отсутствия прогрева.
    assert summary_loader.load_protocol_summaries.cache_info().currsize == 1, (
        "прогрев не наполнил кеш сводок"
    )
    assert summary_loader._path_to_summary.cache_info().currsize == 1, (
        "прогрев не наполнил path-индекс"
    )
    assert summary_loader._summaries_by_condition_id.cache_info().currsize == 1, (
        "прогрев не наполнил индекс по condition_id"
    )

    assert len(summary_loader.load_protocol_summaries(usable_only=False)) == 2


def test_prewarm_protocol_summaries_survives_empty_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Пустой каталог - это ноль сводок, а не исключение в фоновом потоке."""
    monkeypatch.setattr(summary_loader, "_data_root", lambda: tmp_path)
    _clear_module_caches()
    try:
        assert summary_loader.prewarm_protocol_summaries() == 0
    finally:
        _clear_module_caches()


def test_prewarm_protocol_icd_index_reports_index_shape() -> None:
    from clinical_knowledge.protocol_icd_index import prewarm_protocol_icd_index

    stats = prewarm_protocol_icd_index()

    assert set(stats) >= {"catalog_entries", "icd_exact_keys", "icd_prefix_keys"}
    assert stats["catalog_entries"] > 0, "каталог протоколов пуст - прогрев бесполезен"
    assert stats["icd_exact_keys"] > 0, "нет точных МКБ-ключей - поиск по коду не прогрет"


def test_loader_caches_are_lru_wrapped() -> None:
    """_clear_module_caches полагается на lru_cache: если кеш заменят, тест упадёт."""
    wrapped = [
        name
        for name, obj in vars(summary_loader).items()
        if callable(obj) and hasattr(obj, "cache_clear")
    ]
    assert "load_protocol_summaries" in wrapped
    assert "_path_to_summary" in wrapped
    assert lru_cache is not None
