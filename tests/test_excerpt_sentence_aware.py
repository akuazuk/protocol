"""Трек 1: выдержки без обрывков на полуфразе."""
from __future__ import annotations

import rag_server


LONG = (
    "Диагноз устанавливается на основании жалоб, анамнеза и осмотра. "
    "Лабораторное обследование включает общий анализ крови и мочи. "
    "Лечение проводится в амбулаторных условиях с назначением препаратов. "
    "Наблюдение осуществляется врачом по месту жительства регулярно."
)


def test_excerpt_ends_on_sentence_boundary() -> None:
    out = rag_server.format_excerpt_for_display(LONG, 120)
    assert out
    assert len(out) <= 121  # limit + возможное многоточие
    # не должно обрываться на полуслове: последний осмысленный символ - пунктуация конца
    assert out.rstrip()[-1] in ".!?…"
    # взяты целые предложения из начала
    assert out.startswith("Диагноз устанавливается")


def test_excerpt_short_text_untouched() -> None:
    short = "Короткий фрагмент."
    assert rag_server.format_excerpt_for_display(short, 200) == short


def test_excerpt_no_leading_fragment_and_no_midword() -> None:
    out = rag_server.format_excerpt_for_display(LONG, 60)
    assert out
    # многоточие или конец предложения, но не оборванное слово с дефисом
    assert not out.rstrip().endswith("-")
