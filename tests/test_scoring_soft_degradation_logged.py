"""Сломанный блок оценки должен оставлять след в логе.

Оценка КЗ и профиль жёсткости построены на мягкой деградации: если
необязательный блок падает, расчёт продолжается без него. Само по себе это
правильно - один сбойный детектор не должен ронять оценку всего случая.

Опасно другое: пропавшие findings означают, что случай получил оценку **лучше
заслуженной** (не начислился штраф), а `except: pass` не оставлял никаких
следов. Систематически сломанный детектор выглядел бы как «стало меньше
замечаний», то есть как улучшение качества.

Эти тесты фиксируют, что предупреждение действительно пишется. Без них
логирование легко потерять при следующем рефакторинге.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge import kz_deep_eval, mo_scoring_profile


def test_broken_subevaluator_is_logged_and_does_not_break_scoring(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Падение необязательного блока: оценка есть, предупреждение есть."""
    import clinical_knowledge.formulary_findings as ff

    def boom(*_args, **_kwargs):
        raise RuntimeError("синтетический сбой детектора")

    monkeypatch.setattr(ff, "formulary_findings", boom)

    case = {"complaints": "кашель", "clinical_diagnosis": "J06.9 ОРВИ"}

    with caplog.at_level(logging.WARNING, logger="protocol.kz.deep_eval"):
        result = kz_deep_eval.evaluate_kz_deep(case)

    # Мягкая деградация сохранена: оценка посчитана.
    assert isinstance(result, dict)
    assert "axes" in result and "findings" in result

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "сбой блока прошёл бесследно - оценка занижена/завышена молча"
    assert any("formulary" in r.getMessage() for r in warnings), (
        f"в логе нет имени сломанного блока: {[r.getMessage() for r in warnings]}"
    )


def test_cache_invalidation_failure_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Не сброшенный кеш = новые пороги не применились. Должно быть видно.

    Иначе методист меняет жёсткость оценок, получает «сохранено», а расчёт
    идёт по прежним порогам.
    """
    from clinical_knowledge import mo_zone_scores

    def boom() -> None:
        raise RuntimeError("синтетический сбой сброса кеша")

    monkeypatch.setattr(mo_zone_scores._load_zone_bands_yaml, "cache_clear", boom)

    with caplog.at_level(logging.WARNING, logger="protocol.mo.scoring_profile"):
        mo_scoring_profile._invalidate_caches()

    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert messages, "сбой сброса кеша прошёл бесследно"
    assert any("пороги" in m or "кеш" in m for m in messages), messages


def test_all_caches_are_attempted_even_if_one_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Сбой на первом кеше не должен мешать сбросить остальные.

    В прежнем варианте блоки шли последовательно и это свойство держалось
    случайно; после переписывания на цикл его нужно проверять явно.
    """
    from clinical_knowledge import kz_deep_eval as deep
    from clinical_knowledge import mo_zone_scores

    def boom() -> None:
        raise RuntimeError("синтетический сбой")

    monkeypatch.setattr(mo_zone_scores._load_zone_bands_yaml, "cache_clear", boom)

    # Прогреваем кеш последнего блока, чтобы увидеть, что его всё же сбросили.
    deep._load_deep_config_yaml()
    assert deep._load_deep_config_yaml.cache_info().currsize == 1

    with caplog.at_level(logging.WARNING, logger="protocol.mo.scoring_profile"):
        mo_scoring_profile._invalidate_caches()

    assert deep._load_deep_config_yaml.cache_info().currsize == 0, (
        "сбой одного кеша помешал сбросить остальные - часть порогов осталась старой"
    )
