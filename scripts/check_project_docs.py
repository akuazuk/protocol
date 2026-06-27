#!/usr/bin/env python3
"""Smoke-check that key project docs match current conventions."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOC_PATHS = [
    ROOT / "README.md",
    ROOT / "docs" / "architecture-stages-print.html",
    ROOT / "docs" / "architecture-b2c-patient.md",
    ROOT / "docs" / "architecture-kravira-fhir-mis-print.html",
    ROOT / "docs" / "current_project_audit.md",
    ROOT / "docs" / "ministry-brief-ru.md",
    ROOT / "docs" / "ministry-brief-print.html",
    ROOT / "docs" / "mvp-presentation.html",
    ROOT / "docs" / "project-docs-maintenance.md",
]

STALE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("устаревший URL методиста ?methodist=1", re.compile(r"\?methodist=1")),
    ("устаревшее «6 блоков» в user-facing docs", re.compile(r"6 блок")),
]

REQUIRED_SNIPPETS: list[tuple[Path, str, re.Pattern[str]]] = [
    (ROOT / "README.md", "упоминание patient.html", re.compile(r"patient\.html", re.I)),
    (ROOT / "README.md", "режим methodist ?mode=methodist", re.compile(r"\?mode=methodist")),
    (
        ROOT / "docs" / "architecture-stages-print.html",
        "patient.html / B2C",
        re.compile(r"patient\.html|B2C", re.I),
    ),
    (
        ROOT / "docs" / "architecture-b2c-patient.md",
        "protocol-logo-mini или wordmark",
        re.compile(r"protocol-logo-(mini|wordmark)"),
    ),
    (ROOT / "docs" / "mvp-presentation.html", "ссылка на patient.html", re.compile(r"patient\.html")),
    (
        ROOT / "docs" / "project-docs-maintenance.md",
        "check_project_docs.py",
        re.compile(r"check_project_docs\.py"),
    ),
]


def read_build_version() -> str:
    text = (ROOT / "rag_server.py").read_text(encoding="utf-8")
    m = re.search(r'^BUILD_VERSION\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        raise SystemExit("BUILD_VERSION not found in rag_server.py")
    return m.group(1)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    build_version = read_build_version()
    print(f"BUILD_VERSION: {build_version}")

    for path, label, pattern in REQUIRED_SNIPPETS:
        content = path.read_text(encoding="utf-8")
        if not pattern.search(content):
            errors.append(f"{path.relative_to(ROOT)}: missing {label}")

    for path in DOC_PATHS:
        if not path.exists():
            errors.append(f"missing file: {path.relative_to(ROOT)}")
            continue
        if path.name == "project-docs-maintenance.md":
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in STALE_PATTERNS:
            if pattern.search(text):
                errors.append(f"{path.relative_to(ROOT)}: {label}")

    audit = ROOT / "docs" / "current_project_audit.md"
    if audit.exists():
        audit_text = audit.read_text(encoding="utf-8")
        if build_version not in audit_text:
            warnings.append(
                "docs/current_project_audit.md: BUILD_VERSION not in header (update «Версия сборки»)"
            )

    b2c = ROOT / "docs" / "architecture-b2c-patient.md"
    if b2c.exists():
        b2c_text = b2c.read_text(encoding="utf-8")
        if build_version not in b2c_text:
            warnings.append(
                "docs/architecture-b2c-patient.md: «Last aligned» BUILD_VERSION differs from rag_server.py"
            )

    index = ROOT / "index.html"
    if index.exists():
        index_text = index.read_text(encoding="utf-8")
        if "app-sticky-bar" not in index_text:
            errors.append("index.html: sticky mini logo bar (app-sticky-bar) not found")
        if "protocol-logo-mini.svg" not in index_text:
            errors.append("index.html: protocol-logo-mini.svg not referenced")

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    if errors:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"\nOK - {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
