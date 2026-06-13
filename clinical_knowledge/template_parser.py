"""Разбор автошаблонных КЗ (ТЗ раздел 11).

Формат:
    >>> L30 Экзема кожи ?:
    * ОБСЛЕДОВАНИЯ ОБЯЗАТЕЛЬНЫЕ:
    - ...
    >>> L93.0 Дискоидная красная волчанка:
    * ОБСЛЕДОВАНИЯ ДОПОЛНИТЕЛЬНЫЕ:
    ...
"""
from __future__ import annotations

import re

from .consult_schema import TemplateBlock

RE_DIAG_HEADER = re.compile(r"^\s*>>>\s*(.+?)\s*:?\s*$")
RE_BLOCK_HEADER = re.compile(r"^\s*\*\s*(.+?)\s*:?\s*$")
RE_LEADING_ICD = re.compile(r"^\s*([A-ZА-Я]\d{2}(?:\.\d{1,2})?)\b")

_BLOCK_TYPE_MAP = (
    ("обязательн", "required_exams"),
    ("дополнительн", "additional_exams"),
    ("обследован", "required_exams"),
    ("лечен", "treatment"),
    ("терапи", "treatment"),
    ("явк", "follow_up"),
    ("наблюден", "follow_up"),
    ("уход", "care"),
)


def _block_type(label: str) -> str:
    low = label.lower()
    for needle, btype in _BLOCK_TYPE_MAP:
        if needle in low:
            return btype
    return "unknown"


def has_template_markers(text: str) -> bool:
    return ">>>" in (text or "")


def parse_template_blocks(text: str) -> list[TemplateBlock]:
    """Разбирает автошаблонные блоки. Пустой список, если шаблона нет."""
    if not has_template_markers(text):
        return []
    lines = (text or "").splitlines()
    blocks: list[TemplateBlock] = []
    cur_diag: str | None = None
    cur_icd: str | None = None
    cur_label: str | None = None
    cur_items: list[str] = []
    cur_source_lines: list[str] = []

    def _flush():
        nonlocal cur_label, cur_items, cur_source_lines
        if cur_diag is not None and cur_label is not None:
            blocks.append(
                TemplateBlock(
                    block_diagnosis_text=cur_diag,
                    icd10_code=cur_icd,
                    block_type=_block_type(cur_label),
                    items=[it for it in cur_items if it],
                    source_text="\n".join(cur_source_lines)[:800],
                )
            )
        cur_label = None
        cur_items = []
        cur_source_lines = []

    for line in lines:
        mdiag = RE_DIAG_HEADER.match(line)
        if mdiag:
            _flush()
            cur_diag = mdiag.group(1).strip()
            micd = RE_LEADING_ICD.match(cur_diag)
            cur_icd = micd.group(1).upper() if micd else None
            continue
        mblk = RE_BLOCK_HEADER.match(line)
        if mblk and cur_diag is not None:
            _flush()
            cur_label = mblk.group(1).strip()
            cur_source_lines = [line]
            continue
        if cur_label is not None:
            cur_source_lines.append(line)
            item = line.strip(" - - \t•*")
            if item:
                cur_items.append(item)
    _flush()
    return blocks
