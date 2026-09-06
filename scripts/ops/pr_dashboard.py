#!/usr/bin/env python3
"""Дашборд открытых PR: кто какие файлы держит и что с чем конфликтует.

Одно место, куда агент смотрит перед началом работы. Отвечает на три вопроса,
из-за которых параллельные вкладки мешают друг другу:

  1. какие файлы уже заняты другим PR;
  2. какие PR нельзя мержить один за другим без переноса;
  3. какие PR висят так долго, что их база разошлась с `main`.

Использование:
    python3 scripts/ops/pr_dashboard.py
    python3 scripts/ops/pr_dashboard.py --files clinical_knowledge/mo_backend.py
    python3 scripts/ops/pr_dashboard.py --json

`--files` отвечает на главный вопрос перед правкой: занят ли этот файл.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ops"))

from pr_isolation import classify_overlap, is_build_version_only_diff  # noqa: E402

REPO = "akuazuk/protocol"
STALE_DAYS = 14
# Предел `gh pr list --json files`: на этом числе список считается обрезанным.
FILES_PAGE_CAP = 100
OWNER_RE = re.compile(r"-(?:(?P<agent>agent\d+)-)?(?P<pc>pc\d+)$")


@dataclass
class Pull:
    number: int
    title: str
    branch: str
    draft: bool
    author: str
    created: datetime
    files: list[str]
    url: str
    # Правит ли PR `rag_server.py` только строкой версии. Такое пересечение
    # снимает rebase-скрипт сам, поэтому оно не должно выглядеть конфликтом:
    # иначе дашборд кричал бы на каждом обычном бампе BUILD_VERSION.
    rag_version_only: bool = False

    @property
    def owner(self) -> str:
        """Владелец по имени ветки; имя ветки для этого и нужно."""
        m = OWNER_RE.search(self.branch)
        if not m:
            return self.author
        agent = m.group("agent") or "?"
        return f"{agent}/{m.group('pc')}"

    @property
    def age_days(self) -> int:
        """Сколько дней PR открыт.

        Считается от создания, а не от `updatedAt`: комментарий бота про
        пересечения обновляет PR и делал бы августовский PR «свежим».
        """
        return (datetime.now(timezone.utc) - self.created).days

    @property
    def stale(self) -> bool:
        return self.age_days >= STALE_DAYS


def plural_files(n: int) -> str:
    """`1 файл`, `3 файла`, `10 файлов`."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} файл"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} файла"
    return f"{n} файлов"


def _all_pr_files(number: int) -> list[str]:
    """Полный список файлов PR через API, без ограничения в 100 штук."""
    try:
        out = subprocess.run(
            [
                "gh", "api", f"repos/{REPO}/pulls/{number}/files", "--paginate",
                "-q", ".[].filename",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _rag_patch_is_version_only(number: int) -> bool:
    """Спросить у API патч `rag_server.py` для одного PR.

    Отдельный вызов делается только для тех PR, которые этот файл трогают:
    `gh pr list` патчей не отдаёт, а без патча нельзя отличить бамп версии от
    настоящей правки.
    """
    # Фильтр отдаётся jq на стороне gh: `--paginate` без него склеивает страницы
    # в несколько JSON-массивов подряд, и это уже не разбирается json.loads.
    # Крупные PR выходят за 100 файлов, поэтому страница может быть не одна.
    try:
        out = subprocess.run(
            [
                "gh", "api", f"repos/{REPO}/pulls/{number}/files", "--paginate",
                "-q", '.[] | select(.filename == "rag_server.py") | .patch',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    patch = out.stdout.strip()
    if not patch:
        return False
    return is_build_version_only_diff(patch)


def fetch_pulls() -> list[Pull]:
    if not shutil.which("gh"):
        print("ERROR: нужен gh (https://cli.github.com)", file=sys.stderr)
        raise SystemExit(2)

    out = subprocess.run(
        [
            "gh", "pr", "list", "--repo", REPO, "--state", "open",
            "--limit", "100", "--json",
            "number,title,headRefName,isDraft,author,createdAt,files,url",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pulls: list[Pull] = []
    for row in json.loads(out.stdout):
        pulls.append(
            Pull(
                number=row["number"],
                title=row.get("title") or "",
                branch=row.get("headRefName") or "",
                draft=bool(row.get("isDraft")),
                author=(row.get("author") or {}).get("login") or "?",
                created=datetime.fromisoformat(
                    row["createdAt"].replace("Z", "+00:00")
                ),
                files=[f["path"] for f in (row.get("files") or []) if f.get("path")],
                url=row.get("url") or "",
            )
        )
    for pr in pulls:
        # `gh pr list` отдаёт максимум 100 файлов на PR. Молча обрезанный
        # список означал бы пропущенное пересечение, поэтому крупные PR
        # дочитываются через API целиком.
        if len(pr.files) >= FILES_PAGE_CAP:
            full = _all_pr_files(pr.number)
            if full:
                pr.files = full
        if "rag_server.py" in pr.files:
            pr.rag_version_only = _rag_patch_is_version_only(pr.number)
    return sorted(pulls, key=lambda p: p.number, reverse=True)


def hard_pairs(pulls: list[Pull]) -> list[tuple[Pull, Pull, list[str]]]:
    """Пары PR с жёстким пересечением файлов."""
    pairs: list[tuple[Pull, Pull, list[str]]] = []
    for i, a in enumerate(pulls):
        for b in pulls[i + 1 :]:
            kind = classify_overlap(
                a.files,
                b.files,
                our_rag_only_version=a.rag_version_only,
                other_rag_only_version=b.rag_version_only,
            )
            if kind["hard"]:
                pairs.append((a, b, sorted(kind["hard"])))
    return pairs


def who_holds(pulls: list[Pull], paths: list[str]) -> dict[str, list[Pull]]:
    """Какие PR держат указанные файлы."""
    held: dict[str, list[Pull]] = {p: [] for p in paths}
    for path in paths:
        for pr in pulls:
            if path in pr.files:
                held[path].append(pr)
    return held


def _print_dashboard(pulls: list[Pull]) -> None:
    print(f"Открытых PR: {len(pulls)}\n")

    for pr in pulls:
        flag = "DRAFT" if pr.draft else "OPEN "
        age = f"открыт {pr.age_days}д" if pr.age_days else "открыт сегодня"
        stale = "  <- завис" if pr.stale else ""
        print(f"  #{pr.number:<4} {flag} {age:>14}  {pr.owner:<12} {pr.title[:46]}{stale}")
        print(f"        {pr.branch}  ({plural_files(len(pr.files))})")

    pairs = hard_pairs(pulls)
    print()
    if not pairs:
        print("Жёстких пересечений между открытыми PR нет: порядок merge любой.")
    else:
        print("Жёсткие пересечения (мержить по одному, второй потом rebase):")
        for a, b, files in pairs:
            shown = ", ".join(files[:3])
            more = f" и ещё {len(files) - 3}" if len(files) > 3 else ""
            print(f"  #{a.number} x #{b.number}: {shown}{more}")

    stale = [p for p in pulls if p.stale]
    if stale:
        print()
        print(f"Зависли больше {STALE_DAYS} дней (база разошлась с main):")
        for pr in stale:
            print(f"  #{pr.number} {pr.age_days}д  {pr.title[:56]}")
        print("  Решение: домержить, переоткрыть от свежего main или закрыть.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", help="проверить, занят ли путь")
    ap.add_argument("--json", action="store_true", help="машинный вывод")
    args = ap.parse_args(argv)

    pulls = fetch_pulls()

    if args.files:
        held = who_holds(pulls, args.files)
        if args.json:
            print(json.dumps(
                {k: [p.number for p in v] for k, v in held.items()},
                ensure_ascii=False,
                indent=2,
            ))
            return 0
        busy = False
        for path, prs in held.items():
            if not prs:
                print(f"свободен  {path}")
                continue
            busy = True
            owners = ", ".join(f"#{p.number} ({p.owner})" for p in prs)
            print(f"ЗАНЯТ     {path} -> {owners}")
        if busy:
            print("\nСначала согласуй с владельцем или дождись merge.")
            return 1
        return 0

    if args.json:
        print(json.dumps(
            [
                {
                    "number": p.number,
                    "owner": p.owner,
                    "branch": p.branch,
                    "draft": p.draft,
                    "age_days": p.age_days,
                    "files": p.files,
                }
                for p in pulls
            ],
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    _print_dashboard(pulls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
