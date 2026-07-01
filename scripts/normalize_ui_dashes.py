#!/usr/bin/env python3
"""Замена длинного/среднего тире на короткий дефис в текстах проекта."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Не трогаем извлечённый корпус, PDF, снимки анализов и клинические фикстуры.
EXCLUDE_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "minzdrav_protocols",
    "corpus_chunks_parts",
    "agent-transcripts",
}

EXCLUDE_PATH_PREFIXES = (
    "data/protocol_summaries/",
    "corpus/",
    "output/",
    "data/ml/analyses/",
    "data/ml/secure/",
    "tests/fixtures/consultations/",
    "tests/fixtures/protocol_summaries/",
    "ml/experiments/",
    "ml/datasets/",
    "ml/registry/checkpoints/",
)

EXCLUDE_FILENAMES = {
    "chunks.json",
    "corpus.json",
    "structured_index.json",
}

INCLUDE_SUFFIXES = {
    ".py",
    ".html",
    ".md",
    ".mdc",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".txt",
}

INCLUDE_FILENAMES = {
    "README",
    "README.md",
    "CURSOR_README.txt",
    ".env.example",
    "env.example",
}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    for prefix in EXCLUDE_PATH_PREFIXES:
        if rel.startswith(prefix):
            return True
    for part in path.parts:
        if part in EXCLUDE_DIR_NAMES:
            return True
    return False


def iter_target_files() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name in EXCLUDE_FILENAMES:
            continue
        if should_skip(path):
            continue
        if path.suffix in INCLUDE_SUFFIXES or path.name in INCLUDE_FILENAMES:
            out.append(path)
    return sorted(out)


def normalize_dashes(text: str) -> str:
    """Em/en dash → короткий дефис (-); диапазоны 2022-2024 → 2022-2024."""
    text = text.replace("\u2013", "-")
    return re.sub(r"\s*\u2014\s*", " - ", text)


_RE_LINE_SKIP = re.compile(
    r"\b(re\.(?:compile|search|match|findall|sub|split|fullmatch)|rf[\"'])"
)


def normalize_file_text(text: str) -> str:
    """Построчно: не трогаем строки с regex (иначе ломаются классы символов)."""
    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if _RE_LINE_SKIP.search(line):
            out_lines.append(line)
        else:
            out_lines.append(normalize_dashes(line))
    return "".join(out_lines)


def main() -> int:
    changed = 0
    total_dashes = 0
    for path in iter_target_files():
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        n_before = raw.count("\u2014") + raw.count("\u2013")
        if not n_before:
            continue
        norm = normalize_file_text(raw)
        if norm == raw:
            continue
        path.write_text(norm, encoding="utf-8")
        rel = path.relative_to(ROOT)
        print(f"updated {rel} ({n_before} dash chars)")
        changed += 1
        total_dashes += n_before
    print(f"done: {changed} file(s), {total_dashes} dash char(s) normalized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
