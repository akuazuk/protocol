"""Тесты дашборда открытых PR.

Сетевые вызовы не проверяются: здесь зафиксирована логика, из-за ошибки в
которой агент решил бы, что файл свободен, и начал править занятое.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ops"))

from pr_dashboard import (  # noqa: E402
    FILES_PAGE_CAP,
    STALE_DAYS,
    Pull,
    hard_pairs,
    plural_files,
    who_holds,
)


def _pull(
    number: int,
    branch: str = "cursor/task-pc1",
    files: list[str] | None = None,
    age_days: int = 0,
    rag_version_only: bool = False,
) -> Pull:
    return Pull(
        number=number,
        title=f"PR {number}",
        branch=branch,
        draft=False,
        author="someone",
        created=datetime.now(timezone.utc) - timedelta(days=age_days),
        files=files or [],
        url="",
        rag_version_only=rag_version_only,
    )


def test_owner_comes_from_branch_name() -> None:
    assert _pull(1, "cursor/mo-lab-agent2-pc3").owner == "agent2/pc3"


def test_owner_without_agent_segment_keeps_computer() -> None:
    assert _pull(1, "cursor/mo-lab-pc1").owner == "?/pc1"


def test_owner_falls_back_to_author_when_branch_is_odd() -> None:
    assert _pull(1, "some-random-branch").owner == "someone"


def test_age_counts_from_creation_not_activity() -> None:
    """Комментарий бота не должен «освежать» августовский PR."""
    assert _pull(1, age_days=27).age_days == 27


def test_stale_threshold() -> None:
    assert not _pull(1, age_days=STALE_DAYS - 1).stale
    assert _pull(2, age_days=STALE_DAYS).stale


def test_hard_pair_found_for_shared_file() -> None:
    a = _pull(1, files=["docs/plans/README.md", "a.py"])
    b = _pull(2, files=["docs/plans/README.md", "b.py"])
    pairs = hard_pairs([a, b])
    assert len(pairs) == 1
    assert pairs[0][2] == ["docs/plans/README.md"]


def test_disjoint_prs_have_no_pairs() -> None:
    a = _pull(1, files=["a.py"])
    b = _pull(2, files=["b.py"])
    assert hard_pairs([a, b]) == []


def test_build_version_only_touch_is_not_a_hard_conflict() -> None:
    """Бамп BUILD_VERSION rebase снимает сам, кричать о нём нельзя."""
    a = _pull(1, files=["rag_server.py"], rag_version_only=True)
    b = _pull(2, files=["rag_server.py"], rag_version_only=False)
    assert hard_pairs([a, b]) == []


def test_two_real_rag_server_edits_do_conflict() -> None:
    a = _pull(1, files=["rag_server.py"])
    b = _pull(2, files=["rag_server.py"])
    pairs = hard_pairs([a, b])
    assert len(pairs) == 1
    assert pairs[0][2] == ["rag_server.py"]


def test_who_holds_reports_every_holder() -> None:
    a = _pull(1, files=["x.py"])
    b = _pull(2, files=["x.py", "y.py"])
    held = who_holds([a, b], ["x.py", "y.py", "free.py"])
    assert [p.number for p in held["x.py"]] == [1, 2]
    assert [p.number for p in held["y.py"]] == [2]
    assert held["free.py"] == []


def test_plural_files_russian_forms() -> None:
    assert plural_files(1) == "1 файл"
    assert plural_files(2) == "2 файла"
    assert plural_files(5) == "5 файлов"
    assert plural_files(11) == "11 файлов"
    assert plural_files(21) == "21 файл"
    assert plural_files(114) == "114 файлов"
    assert plural_files(144) == "144 файла"


def test_page_cap_is_the_documented_github_limit() -> None:
    """Если предел изменится, дочитывание крупных PR должно поехать заметно."""
    assert FILES_PAGE_CAP == 100
