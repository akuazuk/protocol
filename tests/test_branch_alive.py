"""Тесты защиты от работы на мёртвой ветке.

Главное, что здесь зафиксировано: squash-merge **не** ловится проверкой
родства с `main`, и именно поэтому нужен признак «remote-ветка исчезла». Если
кто-то решит упростить логику до одного `merge-base --is-ancestor`, эти тесты
обязаны упасть.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ops"))

import check_branch_alive as guard  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """По умолчанию: ветка живая, upstream есть, не смержена."""
    monkeypatch.setattr(guard, "ahead_behind", lambda branch: (2, 0))
    monkeypatch.setattr(guard, "was_published", lambda branch: True)
    monkeypatch.setattr(guard, "remote_branch_exists", lambda branch: True)
    monkeypatch.setattr(guard, "github_pr_state", lambda branch: ("", ""))
    monkeypatch.delenv("ALLOW_DEAD_BRANCH", raising=False)


@pytest.mark.parametrize("branch", sorted(guard.SHARED_BRANCHES))
def test_shared_branches_are_refused(branch: str) -> None:
    v = guard.evaluate(branch)
    assert not v.alive
    assert "общая" in v.reason


def test_published_branch_without_own_work_warns_but_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отличить остаток merge-коммита от новой ветки офлайн нельзя: предупреждаем."""
    monkeypatch.setattr(guard, "ahead_behind", lambda branch: (0, 7))
    v = guard.evaluate("cursor/x-agent1-pc1")
    assert v.alive
    assert "без своих коммитов" in v.reason
    assert "--online" in v.detail


def test_squash_merge_is_caught_by_missing_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ровно случай #192: родство не сработало, ветку удалили после merge."""
    monkeypatch.setattr(guard, "remote_branch_exists", lambda branch: False)
    v = guard.evaluate("cursor/production-readiness-agent1-pc1")
    assert not v.alive
    assert "origin/" in v.reason
    assert "squash" in v.detail


def test_fresh_branch_identical_to_main_is_not_called_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Свежая ветка от main - предок main. Ложное «смержена» здесь недопустимо."""
    monkeypatch.setattr(guard, "ahead_behind", lambda branch: (0, 0))
    v = guard.evaluate("cursor/fresh-agent1-pc1")
    assert v.alive
    assert "смержена" not in v.reason


def test_never_published_branch_with_local_commits_is_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guard, "was_published", lambda branch: False)
    monkeypatch.setattr(guard, "remote_branch_exists", lambda branch: False)
    monkeypatch.setattr(guard, "ahead_behind", lambda branch: (3, 0))
    v = guard.evaluate("cursor/brand-new-agent1-pc1")
    assert v.alive
    assert "не публиковалась" in v.reason


def test_live_branch_passes() -> None:
    assert guard.evaluate("cursor/live-agent1-pc1").alive


def test_detached_head_is_not_treated_as_dead() -> None:
    v = guard.evaluate("")
    assert v.alive
    assert "detached" in v.reason


def test_online_merged_pr_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "github_pr_state", lambda branch: ("MERGED", "192"))
    v = guard.evaluate("cursor/x-agent1-pc1", online=True)
    assert not v.alive
    assert "#192" in v.reason


def test_online_check_runs_even_without_local_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ветки может не быть локально, а PR по ней - уже смержен."""
    monkeypatch.setattr(guard, "was_published", lambda branch: False)
    monkeypatch.setattr(guard, "remote_branch_exists", lambda branch: False)
    monkeypatch.setattr(guard, "github_pr_state", lambda branch: ("MERGED", "148"))
    v = guard.evaluate("cursor/gone-agent1-pc1", online=True)
    assert not v.alive
    assert "#148" in v.reason


def test_online_closed_pr_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "github_pr_state", lambda branch: ("CLOSED", "77"))
    v = guard.evaluate("cursor/x-agent1-pc1", online=True)
    assert not v.alive
    assert "#77" in v.reason


def test_online_open_pr_stays_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "github_pr_state", lambda branch: ("OPEN", "210"))
    assert guard.evaluate("cursor/x-agent1-pc1", online=True).alive


def test_offline_mode_does_not_ask_github(monkeypatch: pytest.MonkeyPatch) -> None:
    """Хук не должен ходить в сеть: без --online PR не опрашивается."""

    def _boom(branch: str) -> tuple[str, str]:
        raise AssertionError("github_pr_state вызван без --online")

    monkeypatch.setattr(guard, "github_pr_state", _boom)
    assert guard.evaluate("cursor/x-agent1-pc1").alive


def test_rebase_in_progress_lets_commit_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Иначе не довести до конца rebase собственной ветки."""
    monkeypatch.setattr(guard, "operation_in_progress", lambda: "rebase-merge")
    monkeypatch.setattr(guard, "current_branch", lambda: "main")
    assert guard.main([]) == 0


def test_explicit_override_lets_commit_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_DEAD_BRANCH", "1")
    monkeypatch.setattr(guard, "current_branch", lambda: "main")
    assert guard.main([]) == 0


def test_main_exits_nonzero_on_dead_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "operation_in_progress", lambda: "")
    monkeypatch.setattr(guard, "current_branch", lambda: "main")
    assert guard.main([]) == 1


def test_hook_is_executable_and_wired() -> None:
    hook = ROOT / ".githooks" / "pre-commit"
    assert hook.is_file(), "нет .githooks/pre-commit"
    text = hook.read_text(encoding="utf-8")
    assert "check_branch_alive.py" in text
    assert "--quiet-ok" in text, "хук должен молчать на живой ветке"
