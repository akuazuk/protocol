#!/usr/bin/env python3
"""Страж персональных данных: не даёт закоммитить ПДн пациентов и непродуктовые файлы.

Запуск:
    python3 scripts/ops/check_no_phi_staged.py              # проверить staged-файлы
    python3 scripts/ops/check_no_phi_staged.py --all        # проверить всё отслеживаемое
    python3 scripts/ops/check_no_phi_staged.py file1 file2  # проверить конкретные файлы

Подключён как локальный pre-commit hook (.pre-commit-config.yaml) и как шаг CI.

Зачем: 2026-09-05 в рабочем дереве лежали `docs/Patient_D/` (ответы на претензию
конкретного пациента + видео) и `myositis_2026.csv` с `patient_id`/`doctor_fio`,
не покрытые .gitignore. Любой `git add -A` отправил бы ПДн в историю навсегда.
См. docs/reports/2026-09-05-prod-state-snapshot.md.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Пути/имена, которые в репозитории недопустимы в принципе.
FORBIDDEN_PATH_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?i)patient_d/", "каталог с юридическими документами по пациенту"),
    (r"(?i)ответ[ _]на[ _]претензи", "ответ на претензию конкретного пациента"),
    (r"(?i)^myositis_.*\.csv$", "выгрузка МИС с patient_id/doctor_fio"),
    (r"(?i)public_realition/|public_relations/", "PR-материалы, не продуктовый код"),
    (r"(?i)^clients_consult/.+\.pdf$", "консультативные заключения пациентов (ПДн)"),
    (r"(?i)secure_cases/", "secure_cases: клинические тексты с ПДн"),
    (r"(?i)^data/medical_exams/", "ежедневный МО-склад с ПДн"),
    (r"(?i)^data/ml/secure/", "тексты КЗ (ПДн)"),
    (r"(?i)^data/KZ/", "консультативные заключения (ПДн)"),
    (r"(?i)^=", "артефакт неудачного pip install"),
)

# Расширения, которые почти всегда означают ПДн/непродуктовый бинарь.
FORBIDDEN_SUFFIXES: dict[str, str] = {
    ".mp4": "видео (возможны клинические записи пациентов)",
    ".mov": "видео (возможны клинические записи пациентов)",
    ".docx": "офисный документ (часто содержит ПДн)",
    ".doc": "офисный документ (часто содержит ПДн)",
    ".xlsx": "таблица (часто содержит ПДн)",
    ".xls": "таблица (часто содержит ПДн)",
}

# Точечные исключения для расширений выше: реально нужные репозиторию файлы.
SUFFIX_ALLOWLIST: tuple[str, ...] = (
    "epam/scheme_mis_protocols.docx",
    "data/icd_reference/mkb10_ru_mkb10su.xlsx",
)
SUFFIX_ALLOWLIST_PREFIXES: tuple[str, ...] = ("archive/docs/konkurs/",)

# Заголовки колонок, выдающие построчную выгрузку с ПДн.
PHI_COLUMN_TOKENS: tuple[str, ...] = (
    "patient_id",
    "visit_id",
    "doctor_fio",
    "specialist_name",
    "patient_fio",
    "patient_bdate",
)

# Сканируем только построчные форматы, где ПДн лежат как данные.
# Агрегатные .json (отчёты, *.meta.json, *_summary.json) намеренно перечисляют
# имена колонок в поле "columns" и гистограммы без значений - это не ПДн,
# они разрешены в .gitignore и не должны ловиться эвристикой.
CSV_SCAN_SUFFIXES: tuple[str, ...] = (".csv", ".tsv")
JSONL_SCAN_SUFFIXES: tuple[str, ...] = (".jsonl",)
MAX_SCAN_BYTES = 64 * 1024


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout


def staged_files() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    return [line.strip() for line in out.splitlines() if line.strip()]


def tracked_files() -> list[str]:
    return [line.strip() for line in _git("ls-files").splitlines() if line.strip()]


def _suffix_allowed(path: str) -> bool:
    if path in SUFFIX_ALLOWLIST:
        return True
    return any(path.startswith(prefix) for prefix in SUFFIX_ALLOWLIST_PREFIXES)


def _resolve(path: str) -> Path | None:
    """Ищет файл сначала относительно корня репо, затем относительно cwd.

    Git отдаёт hook'у пути от корня репозитория, но скрипт должны уметь
    запускать и вручную из любого каталога.
    """
    for candidate in ((ROOT / path), Path(path)):
        if candidate.is_file():
            return candidate
    return None


def _read_head(path: str) -> str | None:
    resolved = _resolve(path)
    if resolved is None:
        return None
    try:
        return resolved.read_bytes()[:MAX_SCAN_BYTES].decode("utf-8", errors="replace")
    except OSError:
        return None


def _looks_like_phi_table(path: str) -> str | None:
    """Ищет ПДн в построчных выгрузках (CSV/TSV/JSONL).

    Проверяется только строка заголовка CSV или ключи первой записи JSONL -
    то есть места, где ПДн лежат как данные. Агрегатные .json не сканируются:
    они законно перечисляют имена колонок в схеме.
    """
    suffix = Path(path).suffix.lower()

    if suffix in CSV_SCAN_SUFFIXES:
        head = _read_head(path)
        if head is None:
            return None
        header = head.splitlines()[0].lower() if head.splitlines() else ""
        fields = {f.strip().strip('"').strip("'") for f in re.split(r"[,;\t]", header)}
        hits = sorted(fields & set(PHI_COLUMN_TOKENS))
        if len(hits) >= 2:
            return "колонки " + ", ".join(hits)
        return None

    if suffix in JSONL_SCAN_SUFFIXES:
        head = _read_head(path)
        if head is None:
            return None
        for line in head.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                return None
            if not isinstance(record, dict):
                return None
            hits = sorted(set(record.keys()) & set(PHI_COLUMN_TOKENS))
            if len(hits) >= 2:
                return "поля записи " + ", ".join(hits)
            return None
        return None

    return None


def check(paths: list[str]) -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []
    for path in paths:
        norm = path.replace("\\", "/")
        for pattern, reason in FORBIDDEN_PATH_PATTERNS:
            if re.search(pattern, norm):
                problems.append((path, reason))
                break
        else:
            suffix = Path(norm).suffix.lower()
            if suffix in FORBIDDEN_SUFFIXES and not _suffix_allowed(norm):
                problems.append((path, FORBIDDEN_SUFFIXES[suffix]))
                continue
            table_reason = _looks_like_phi_table(norm)
            if table_reason:
                problems.append((path, f"похоже на выгрузку с ПДн: {table_reason}"))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="проверить все отслеживаемые файлы"
    )
    parser.add_argument("files", nargs="*", help="конкретные файлы для проверки")
    args = parser.parse_args()

    if args.files:
        paths = args.files
    elif args.all:
        paths = tracked_files()
    else:
        paths = staged_files()

    if not paths:
        return 0

    problems = check(paths)
    if not problems:
        return 0

    print("ОТКАЗ: в коммит попали персональные данные или непродуктовые файлы.\n")
    for path, reason in problems:
        print(f"  {path}\n      причина: {reason}")
    print(
        "\nЧто делать:\n"
        "  1. Убрать из индекса:  git restore --staged <файл>\n"
        "  2. Перенести файл вне репозитория (например ~/Protocol_Private, chmod 700)\n"
        "  3. Убедиться, что путь закрыт в .gitignore\n"
        "\nЕсли файл действительно нужен репозиторию - добавь его в allowlist\n"
        "в scripts/ops/check_no_phi_staged.py и объясни причину в описании коммита."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
