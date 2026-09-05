#!/usr/bin/env python3
"""Ищет скрипты и модули, на которые никто не ссылается.

Зачем: в scripts/ накопились разовые утилиты, и по имени файла нельзя понять,
нужен он ещё или это след завершённой задачи. Скрипт отвечает на один вопрос -
упоминается ли файл где-нибудь, кроме себя самого: в коде, тестах, CI, deploy,
Makefile, документации.

Считается упоминанием: импорт модуля, вызов по пути, ссылка в docs. Поэтому
"orphan" здесь означает "нет ни одной ссылки", а не "не нужен" - решение
принимает человек, но список для разбора получается короткий и проверяемый.

Запуск:
    python3 scripts/ops/find_orphan_scripts.py            # отчёт
    python3 scripts/ops/find_orphan_scripts.py --json     # машинный вывод
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Где искать упоминания. Читаем один раз и держим в памяти: пробегать grep-ом
# по каждому файлу отдельно на ~4700 файлах слишком медленно.
SEARCH_SUFFIXES = {
    ".py", ".sh", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".json", ".jsonl",
    ".md", ".mdc", ".html", ".js", ".css", ".txt", ".in", ".plist", "",
}

# Каталоги, которые не являются "потребителями" кода: ссылка оттуда не
# доказывает, что скрипт нужен сегодня.
WEAK_REFERENCE_DIRS = ("archive/", "docs/reports/", "docs/plans/")

# Точки входа, у которых ссылок быть и не должно.
ENTRYPOINT_ALLOWLIST = {
    "rag_server.py",
    "conftest.py",
}


def tracked_files() -> list[str]:
    raw = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    return [f.decode("utf-8", "replace") for f in raw.split(b"\x00") if f]


def build_reference_index(files: list[str]) -> dict[str, str]:
    """path -> содержимое, только для текстовых файлов разумного размера."""
    index: dict[str, str] = {}
    for rel in files:
        p = ROOT / rel
        if p.suffix.lower() not in SEARCH_SUFFIXES:
            continue
        try:
            if p.stat().st_size > 4 * 1024 * 1024:
                continue
            index[rel] = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
    return index


def candidates(files: list[str]) -> list[str]:
    """Скрипты и модули, которые имеет смысл проверять на сиротство."""
    out = []
    for rel in files:
        if not rel.endswith((".py", ".sh")):
            continue
        if rel.startswith(("archive/", "tests/", "e2e/")):
            continue
        if Path(rel).name in ENTRYPOINT_ALLOWLIST:
            continue
        out.append(rel)
    return out


def find_references(target: str, index: dict[str, str]) -> list[str]:
    name = Path(target).name
    stem = Path(target).stem
    # Ищем и путь, и имя файла, и импорт модуля: scripts/foo.py, foo.py,
    # "from scripts.foo", "import foo", "scripts.foo".
    module_path = target[:-3].replace("/", ".") if target.endswith(".py") else None
    needles = [target, name]
    if module_path:
        needles.append(module_path)

    hits = []
    for rel, text in index.items():
        if rel == target:
            continue
        if any(n in text for n in needles):
            hits.append(rel)
            continue
        # Импорт по имени модуля внутри того же пакета: "import foo" / "from foo".
        if target.endswith(".py") and re.search(
            rf"^\s*(?:from\s+{re.escape(stem)}\s+import|import\s+{re.escape(stem)})\b",
            text,
            re.MULTILINE,
        ):
            hits.append(rel)
    return sorted(hits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="машинный вывод")
    args = ap.parse_args()

    files = tracked_files()
    index = build_reference_index(files)
    result: dict[str, list[str]] = {}
    for target in candidates(files):
        result[target] = find_references(target, index)

    orphans = {k: v for k, v in result.items() if not v}
    weak = {
        k: v
        for k, v in result.items()
        if v and all(r.startswith(WEAK_REFERENCE_DIRS) for r in v)
    }

    if args.json:
        json.dump(
            {"orphans": sorted(orphans), "only_weak_references": {k: v for k, v in sorted(weak.items())}},
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        print()
        return 0

    print(f"проверено файлов: {len(result)}\n")
    print(f"== без единой ссылки: {len(orphans)}")
    for k in sorted(orphans):
        print(f"   {k}")
    print(f"\n== только слабые ссылки (archive/ или завершённые планы/отчёты): {len(weak)}")
    by_dir: dict[str, list[str]] = defaultdict(list)
    for k in sorted(weak):
        by_dir[str(Path(k).parent)].append(Path(k).name)
    for d, names in sorted(by_dir.items()):
        print(f"   {d}/ ({len(names)})")
        for n in names:
            print(f"      {n}  <- {', '.join(weak[f'{d}/{n}'][:2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
