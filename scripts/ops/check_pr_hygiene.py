#!/usr/bin/env python3
"""Гигиена PR: имя ветки, заполненность описания, размер diff.

Зачем отдельная проверка. Пересечение файлов уже ловит `pr_isolation.py`, а
здесь проверяется то, из чего пересечения возникают: безымянные ветки, PR без
объявленного владельца и зоны изменений, и слишком крупные diff, которые
физически невозможно смержить, не задев соседа.

Проверка блокирующая, но с явным обходом: метка `hygiene-ack` на PR. Обход
остаётся в истории PR, поэтому решение видно, а не растворяется.

Локально:
    python3 scripts/ops/check_pr_hygiene.py local

В CI (по событию pull_request):
    python3 scripts/ops/check_pr_hygiene.py github-event
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "pull_request_template.md"

# Ветка обязана называть задачу и владельца: cursor/<slug>[-agent<N>]-pc<N>.
# Префикс ограничен, чтобы `main`, `HEAD` и случайные имена не проходили.
BRANCH_RE = re.compile(
    r"^(?P<prefix>cursor|codex|hotfix|release)/"
    r"(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*?)"
    r"(?:-agent\d+)?-pc\d+$"
)

# Общие мутабельные ветки: их запрещает AGENTS.md, разделять их между задачами
# означает терять чужие коммиты.
BANNED_BRANCHES = frozenset({"main", "master", "codex/main-sync", "cursor/main-sync"})

# Пороги предупреждений. Не блокируют: программа доведения до прода честно
# бывает крупной. Но крупный PR обязан объяснять, почему он не разбит.
MAX_FILES_SOFT = 40
MAX_LINES_SOFT = 1500

UNTICKED_BOX = re.compile(r"^\s*-\s*\[\s\]\s+(?P<label>.+?)\s*$", re.M)
PLACEHOLDER = re.compile(r"<[^<>\n]{3,}>")


@dataclass
class Report:
    """Итог проверки: ошибки блокируют, предупреждения только печатаются."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def template_placeholders(template_text: str) -> set[str]:
    """Плейсхолдеры шаблона PR, например `<agent1 / pc1>`.

    Список берётся из самого шаблона, а не дублируется здесь: иначе правка
    шаблона тихо отключила бы проверку.
    """
    return {m.group(0) for m in PLACEHOLDER.finditer(template_text)}


def check_branch_name(branch: str) -> list[str]:
    """Ошибки в имени ветки."""
    if not branch:
        return ["не удалось определить имя ветки"]
    if branch in BANNED_BRANCHES:
        return [
            (
                f"ветка `{branch}` общая и мутабельная: заведи задачу через "
                "scripts/ops/git_task_start.sh"
            )
        ]
    if not BRANCH_RE.match(branch):
        return [
            (
                f"имя ветки `{branch}` не по конвенции. Нужно "
                "`cursor|codex|hotfix|release/<задача>[-agent<N>]-pc<N>`, "
                "например `cursor/mo-lab-import-agent1-pc1`. Имя показывает, "
                "какая задача и чей компьютер её держит."
            )
        ]
    return []


def check_body(body: str, placeholders: set[str]) -> list[str]:
    """Ошибки в описании PR: незаполненный шаблон, неотмеченный чек-лист."""
    problems: list[str] = []
    text = body or ""

    if not text.strip():
        problems.append(
            "описание PR пустое: без владельца и зоны изменений соседняя вкладка "
            "не может проверить пересечение"
        )
        return problems

    left = sorted(p for p in placeholders if p in text)
    if left:
        shown = ", ".join(f"`{p}`" for p in left[:4])
        more = f" и ещё {len(left) - 4}" if len(left) > 4 else ""
        problems.append(f"в описании остались шаблоны: {shown}{more}")

    unticked = [m.group("label") for m in UNTICKED_BOX.finditer(text)]
    if unticked:
        shown = "; ".join(unticked[:4])
        problems.append(f"чек-лист не пройден: {shown}")

    return problems


def check_size(changed_files: int, additions: int, deletions: int) -> list[str]:
    """Предупреждения о размере. Крупный PR - причина конфликтов у соседей."""
    warnings: list[str] = []
    lines = additions + deletions
    if changed_files > MAX_FILES_SOFT:
        warnings.append(
            f"{changed_files} файлов (порог {MAX_FILES_SOFT}): чем шире diff, "
            "тем вероятнее, что соседний PR придётся переносить вручную"
        )
    if lines > MAX_LINES_SOFT:
        warnings.append(
            f"{lines} изменённых строк (порог {MAX_LINES_SOFT}): "
            "разбей по смыслу, если причины review независимы"
        )
    return warnings


def evaluate(
    *,
    branch: str,
    body: str,
    placeholders: set[str],
    changed_files: int = 0,
    additions: int = 0,
    deletions: int = 0,
    acknowledged: bool = False,
) -> Report:
    """Собрать отчёт. `acknowledged` переводит ошибки в предупреждения."""
    report = Report()
    problems = check_branch_name(branch) + check_body(body, placeholders)
    report.warnings.extend(check_size(changed_files, additions, deletions))

    if acknowledged and problems:
        report.warnings.extend(f"обойдено меткой hygiene-ack: {p}" for p in problems)
    else:
        report.errors.extend(problems)
    return report


def _current_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def _print(report: Report) -> int:
    for w in report.warnings:
        print(f"WARN: {w}")
    for e in report.errors:
        print(f"ERROR: {e}")
    if report.ok:
        print("OK: гигиена PR соблюдена")
        return 0
    print()
    print("Канон: docs/deploy/multi-agent-workflow-v3.md")
    return 1


def _template_text() -> str:
    try:
        return TEMPLATE.read_text(encoding="utf-8")
    except OSError:
        return ""


def cmd_local() -> int:
    """Проверить только то, что видно без GitHub: имя текущей ветки."""
    report = Report()
    report.errors.extend(check_branch_name(_current_branch()))
    return _print(report)


def cmd_github_event() -> int:
    """Проверить PR по событию `pull_request`."""
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path or not Path(path).is_file():
        print("WARN: GITHUB_EVENT_PATH не задан, проверяю только имя ветки")
        return cmd_local()

    event = json.loads(Path(path).read_text(encoding="utf-8"))
    pr = event.get("pull_request") or {}
    labels = {(lb.get("name") or "") for lb in (pr.get("labels") or [])}

    report = evaluate(
        branch=pr.get("head", {}).get("ref", ""),
        body=pr.get("body") or "",
        placeholders=template_placeholders(_template_text()),
        changed_files=int(pr.get("changed_files") or 0),
        additions=int(pr.get("additions") or 0),
        deletions=int(pr.get("deletions") or 0),
        acknowledged="hygiene-ack" in labels,
    )
    return _print(report)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "local"
    if cmd in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    if cmd == "local":
        return cmd_local()
    if cmd == "github-event":
        return cmd_github_event()
    print(f"неизвестная команда: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
