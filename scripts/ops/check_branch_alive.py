#!/usr/bin/env python3
"""Отказывает работать на мёртвой ветке.

Мёртвая ветка - это ветка, чья работа уже в `main` или заброшена. Коммит в неё
не потерян физически, но он никуда не поедет: PR закрыт, а новый PR от такой
ветки покажет в diff чужие изменения, уже смерженные в `main`.

Почему проверка нужна отдельно от здравого смысла: обычный способ убедиться
«смержена ли ветка» - `git merge-base --is-ancestor` - для squash-merge **не
работает**. Squash создаёт в `main` новый коммит с другим SHA, поэтому исходная
ветка родителем `main` не становится и выглядит живой. Так и вышло 2026-09-06:
`cursor/production-readiness-agent1-pc1` была смержена как #192, а проверка
родства показывала её живой, и рабочая копия осталась стоять на ней.

Надёжный офлайн-признак squash-merge в этом репозитории - удалённая ветка:
в настройках включён `delete_branch_on_merge`, поэтому после merge
`origin/<ветка>` исчезает при первом `git fetch --prune`. Если у локальной
ветки настроен upstream, а remote-ветки больше нет - работа уехала.

Проверка офлайн и без сети: годится для pre-commit. Флаг `--online`
дополнительно спрашивает GitHub о состоянии PR - для preflight, не для хука.

Использование:
  scripts/ops/check_branch_alive.py            # текущая ветка
  scripts/ops/check_branch_alive.py --online   # плюс состояние PR на GitHub
  scripts/ops/check_branch_alive.py --branch X

Коды выхода: 0 - ветка живая, 1 - мёртвая или общая, 2 - ошибка вызова.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

REPO = "akuazuk/protocol"

# Общие мутабельные ветки: в них не работают, их не переиспользуют под задачу.
SHARED_BRANCHES = frozenset({"main", "master", "codex/main-sync", "cursor/main-sync"})

# Операции, во время которых git сам создаёт коммиты. Хук в этот момент
# молчит: иначе rebase собственной ветки не довести до конца.
IN_PROGRESS_MARKERS = (
    "rebase-merge",
    "rebase-apply",
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
)

START_HINT = (
    "scripts/ops/git_task_start.sh <задача> --pc=<pcN> --branch=cursor/<задача>-agent<N>-pc<N>"
)


@dataclass
class Verdict:
    alive: bool
    reason: str
    detail: str = ""

    def report(self) -> str:
        head = ("OK: " if self.alive else "СТОП: ") + self.reason
        return head if not self.detail else f"{head}\n{self.detail}"


def _git(*args: str) -> tuple[int, str]:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return proc.returncode, (proc.stdout or "").strip()


def current_branch() -> str:
    """Имя текущей ветки или пустая строка при detached HEAD."""
    code, out = _git("symbolic-ref", "--quiet", "--short", "HEAD")
    return out if code == 0 else ""


def operation_in_progress() -> str:
    """Идёт ли rebase/merge/cherry-pick: тогда проверку не применяем."""
    code, git_dir = _git("rev-parse", "--git-path", "")
    if code != 0:
        return ""
    base = git_dir or ".git"
    for marker in IN_PROGRESS_MARKERS:
        if os.path.exists(os.path.join(base, marker)):
            return marker
    return ""


def was_published(branch: str) -> bool:
    """Публиковалась ли ветка когда-нибудь на origin.

    Отличать «никогда не пушили» от «удалили после merge» нужно точно, иначе
    страж заблокирует второй коммит в новой ветке. Различие в конфиге:

    - ветка создана от `origin/main` - git ставит upstream на **main**
      (`branch.<X>.merge = refs/heads/main`), remote-ветки `origin/<X>` нет и
      никогда не было;
    - ветку пушили через `push -u` - upstream указывает на её собственную
      удалённую копию (`branch.<X>.merge = refs/heads/<X>`), и эта запись
      остаётся в конфиге даже после того, как remote-ветку удалили.

    Проверено на живом репозитории 2026-09-06: у свежей
    `cursor/dead-branch-guard-agent1-pc1` merge = `refs/heads/main`, у мёртвой
    `cursor/production-readiness-agent1-pc1` - её собственное имя.
    """
    code, out = _git("config", "--get", f"branch.{branch}.merge")
    return code == 0 and out == f"refs/heads/{branch}"


def remote_branch_exists(branch: str) -> bool:
    code, _ = _git("rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}")
    return code == 0


def ahead_behind(branch: str) -> tuple[int, int]:
    """(своих коммитов, отставание от origin/main). (-1, -1) если не посчитать.

    Родство с `main` само по себе о merge не говорит: у свежей ветки от `main`
    своих коммитов ноль, и она тоже предок. Поэтому решение принимается по
    удалённой ветке и состоянию PR, а эти числа нужны только для
    предупреждения «ветка опубликована, но своей работы в ней нет».
    """
    if not remote_ref_exists("origin/main"):
        return (-1, -1)
    code, out = _git("rev-list", "--left-right", "--count", f"origin/main...{branch}")
    if code != 0:
        return (-1, -1)
    parts = out.split()
    if len(parts) != 2:
        return (-1, -1)
    try:
        behind, ahead = int(parts[0]), int(parts[1])
    except ValueError:
        return (-1, -1)
    return (ahead, behind)


def remote_ref_exists(ref: str) -> bool:
    code, _ = _git("rev-parse", "--verify", "--quiet", ref)
    return code == 0


def github_pr_state(branch: str) -> tuple[str, str]:
    """(состояние, номер) самого свежего PR с этой head-веткой. Требует сети."""
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            REPO,
            "--head",
            branch,
            "--state",
            "all",
            "--limit",
            "5",
            "--json",
            "number,state",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "", ""
    import json

    try:
        rows = json.loads(proc.stdout or "[]")
    except ValueError:
        return "", ""
    if not rows:
        return "", ""
    # Самый свежий PR - с наибольшим номером.
    row = max(rows, key=lambda r: int(r.get("number") or 0))
    return str(row.get("state") or ""), str(row.get("number") or "")


def evaluate(branch: str, online: bool = False) -> Verdict:
    if not branch:
        return Verdict(
            True,
            "detached HEAD, это не ветка задачи",
            "Проверка ветки не применяется. Для релизного checkout это норма; "
            "коммитить в detached HEAD не нужно.",
        )

    if branch in SHARED_BRANCHES:
        return Verdict(
            False,
            f"`{branch}` - общая ветка, в ней не работают",
            f"Заведи задачу: {START_HINT}",
        )

    published = was_published(branch)

    # Основной офлайн-признак: ветку публиковали, а на remote её больше нет.
    # В репозитории включено удаление ветки после merge, поэтому это почти
    # всегда squash-merge - тот случай, который проверка родства не видит.
    if published and not remote_branch_exists(branch):
        return Verdict(
            False,
            f"remote-ветки origin/{branch} больше нет",
            "Включено удаление ветки после merge, поэтому это почти всегда "
            "признак squash-merge: работа уже в `main`, ветка мёртвая.\n"
            "Если ветку удалили вручную, а не по merge - убедись сам и повтори "
            "с ALLOW_DEAD_BRANCH=1.\n"
            f"Свежий worktree: {START_HINT}",
        )

    # Состояние PR - самый надёжный признак, но требует сети. В хуке не
    # используется, поэтому проверяется до вывода про новую ветку: ветки может
    # не быть локально, а PR по ней - уже смержен.
    if online:
        state, number = github_pr_state(branch)
        if state == "MERGED":
            return Verdict(
                False,
                f"PR #{number} с этой ветки уже смержен",
                f"Свежий worktree: {START_HINT}",
            )
        if state == "CLOSED":
            return Verdict(
                False,
                f"PR #{number} с этой ветки закрыт без merge",
                "Ветку признали ненужной. Не дописывай в неё: либо осознанно "
                "переоткрывай PR, либо начинай задачу заново.\n"
                f"Свежий worktree: {START_HINT}",
            )

    if not published:
        return Verdict(
            True,
            f"`{branch}` ещё не публиковалась на origin",
            "Это нормально до первого `git push -u`.",
        )

    # Опубликована, remote на месте, но своей работы нет и отстала от main.
    # Не блокируем: так выглядит и ветка, смерженная merge-коммитом без
    # удаления, и просто заново созданная от старого main. Отличить офлайн
    # нельзя, поэтому предупреждаем и подсказываем точную проверку.
    ahead, behind = ahead_behind(branch)
    if ahead == 0 and behind > 0:
        return Verdict(
            True,
            f"`{branch}` без своих коммитов и отстаёт от origin/main на {behind}",
            "Если это остаток смерженной задачи - не продолжай в ней.\n"
            "Точная проверка: scripts/ops/check_branch_alive.py --online",
        )

    return Verdict(True, f"`{branch}` живая")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проверяет, что текущая ветка не мёртвая и не общая."
    )
    parser.add_argument("--branch", default=None, help="проверить указанную ветку")
    parser.add_argument(
        "--online",
        action="store_true",
        help="дополнительно спросить GitHub о состоянии PR (нужна сеть)",
    )
    parser.add_argument("--quiet-ok", action="store_true", help="молчать, если ветка живая")
    args = parser.parse_args(argv)

    if os.environ.get("ALLOW_DEAD_BRANCH", "").strip() == "1":
        return 0

    marker = operation_in_progress()
    if marker:
        return 0

    branch = args.branch if args.branch is not None else current_branch()
    verdict = evaluate(branch, online=args.online)

    if verdict.alive and args.quiet_ok:
        return 0
    print(verdict.report(), file=sys.stdout if verdict.alive else sys.stderr)
    return 0 if verdict.alive else 1


if __name__ == "__main__":
    raise SystemExit(main())
