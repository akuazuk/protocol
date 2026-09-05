"""Индекс планов должен совпадать с тем, что лежит на диске.

Правило `plans.mdc` требует читать актуальный план перед работой, а индекс -
единственный вход в каталог из 100+ файлов. Если план на диске есть, а в
индексе нет, его просто не найдут: 2026-09-05 так потерялись четыре плана,
включая свежий про ложный рабочий диагноз.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "docs" / "plans"
README = PLANS / "README.md"

_LINK = re.compile(r"\((\d{4}-\d{2}-\d{2}[^)]*\.md)\)")
_ROW = re.compile(r"^\|\s*\[([^\]]+)\]\(([^)]+\.md)\)\s*\|(.*)\|\s*([^|]+?)\s*\|\s*$")


def _on_disk() -> set[str]:
    return {p.name for p in PLANS.glob("*.md") if p.name != "README.md"}


def _in_index() -> set[str]:
    return set(_LINK.findall(README.read_text(encoding="utf-8")))


def test_every_plan_on_disk_is_indexed() -> None:
    missing = sorted(_on_disk() - _in_index())
    assert not missing, (
        "планы есть на диске, но их нет в docs/plans/README.md - их не найдут:\n  "
        + "\n  ".join(missing)
    )


def test_index_has_no_dangling_entries() -> None:
    dangling = sorted(_in_index() - _on_disk())
    assert not dangling, (
        "индекс ссылается на планы, которых нет на диске:\n  " + "\n  ".join(dangling)
    )


def test_every_row_has_a_status() -> None:
    """Без статуса непонятно, действующий план или уже заменён."""
    allowed_prefixes = ("active", "archived", "completed", "superseded", "draft")
    bad: list[str] = []
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ["):
            continue
        m = _ROW.match(line)
        if not m:
            bad.append(f"строка не разобрана: {line[:70]}")
            continue
        status = m.group(4).strip().lower()
        if not status.startswith(allowed_prefixes):
            bad.append(f"{m.group(2)}: статус {status[:40]!r}")
    assert not bad, "проблемы со статусами в индексе планов:\n  " + "\n  ".join(bad)


def test_plan_filenames_follow_convention() -> None:
    """`plans.mdc`: YYYY-MM-DD-<тема>-vN.md, версии не перезаписываются."""
    bad = [
        n
        for n in sorted(_on_disk())
        if not re.match(r"^\d{4}-\d{2}-\d{2}-[a-z0-9-]+-v\d+\.md$", n)
    ]
    assert not bad, "имена планов не по соглашению:\n  " + "\n  ".join(bad)
