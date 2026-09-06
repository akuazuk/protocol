#!/usr/bin/env python3
"""Проверка, что requirements-rag.lock соответствует requirements-rag.txt.

Зачем не `uv pip compile` + `diff`: пересборка всегда берёт самые свежие версии,
поэтому диff краснел бы от любого релиза в PyPI, никак не связанного с этим PR.
CI падал бы по расписанию чужих проектов.

Что проверяется на самом деле - что lock не отстал от объявленных зависимостей:

1) каждая зависимость из .txt есть в lock;
2) закреплённая в lock версия удовлетворяет спецификатору из .txt
   (то есть нельзя поднять нижнюю границу в .txt и забыть про lock);
3) lock закрепляет версии через `==` (иначе это не lock);
4) в lock есть хеши (`--generate-hashes`) - защита от подмены пакета.

Расхождение версий lock и последнего PyPI - это НЕ ошибка: обновлениями
занимается Dependabot отдельными PR.

Использование:

  python3 scripts/ops/check_requirements_lock.py
  python3 scripts/ops/check_requirements_lock.py --requirements requirements-rag.txt \
      --lock requirements-rag.lock

Коды возврата: 0 - lock согласован, 1 - расхождение, 2 - ошибка чтения.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Строка зависимости: имя[extras]спецификатор. Комментарии и -r уже отброшены.
_REQ_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)\s*(?:\[(?P<extras>[^\]]*)\])?\s*(?P<spec>.*)$"
)
_PIN_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s;\\]+)")


def _normalize(name: str) -> str:
    """PEP 503: Pillow/pillow, PyMuPDF/pymupdf, python_dotenv/python-dotenv - одно и то же."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _parse_requirements(path: Path) -> dict[str, str]:
    """Прямые зависимости файла: имя -> спецификатор. Вложенные -r не разворачиваются."""
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            # -r/-c/--flag: ссылки на другие файлы проверяются отдельным вызовом.
            continue
        m = _REQ_RE.match(line)
        if not m:
            continue
        out[_normalize(m.group("name"))] = (m.group("spec") or "").strip()
    return out


def _parse_lock(path: Path) -> tuple[dict[str, str], bool]:
    """Закреплённые версии lock: имя -> версия. Второе значение - есть ли хеши."""
    text = path.read_text(encoding="utf-8")
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--hash"):
            continue
        m = _PIN_RE.match(line)
        if m:
            pins[_normalize(m.group("name"))] = m.group("version")
    return pins, "--hash=" in text


def _version_tuple(v: str) -> tuple[int, ...]:
    """Числовая часть версии для сравнения. Суффиксы (rc, post) отбрасываются."""
    parts: list[int] = []
    for chunk in v.split(".")[:4]:
        m = re.match(r"^(\d+)", chunk)
        parts.append(int(m.group(1)) if m else 0)
    return tuple(parts)


def _satisfies(version: str, spec: str) -> tuple[bool, str]:
    """Удовлетворяет ли закреплённая версия спецификатору из .txt."""
    if not spec:
        return True, ""
    got = _version_tuple(version)
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause or clause.startswith(";"):
            # Маркер окружения (python_version < ...) здесь не оцениваем.
            continue
        m = re.match(r"^(==|>=|<=|!=|~=|>|<)\s*([0-9][^\s;]*)$", clause)
        if not m:
            continue
        op, want_raw = m.group(1), m.group(2)
        want = _version_tuple(want_raw)
        ok = {
            ">=": got >= want,
            ">": got > want,
            "<=": got <= want,
            "<": got < want,
            "==": got == want,
            "!=": got != want,
            # ~=X.Y: та же мажорная, не ниже X.Y
            "~=": got >= want and got[:1] == want[:1],
        }[op]
        if not ok:
            return False, f"{version} не удовлетворяет {op}{want_raw}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Согласованность lock и requirements")
    ap.add_argument("--requirements", type=Path, default=ROOT / "requirements-rag.txt")
    ap.add_argument("--lock", type=Path, default=ROOT / "requirements-rag.lock")
    args = ap.parse_args()

    if not args.requirements.is_file():
        print(f"нет файла зависимостей: {args.requirements}", file=sys.stderr)
        return 2
    if not args.lock.is_file():
        print(
            f"нет lock-файла: {args.lock}\n"
            "Собрать (обязательно под версию Python из Dockerfile, сейчас 3.11):\n"
            "  uv pip compile requirements-rag.txt --generate-hashes "
            "--python-version 3.11 --output-file requirements-rag.lock",
            file=sys.stderr,
        )
        return 2

    reqs = _parse_requirements(args.requirements)
    pins, has_hashes = _parse_lock(args.lock)

    problems: list[str] = []

    if not pins:
        problems.append("lock не содержит ни одной закреплённой версии (нет строк вида пакет==версия)")
    if not has_hashes:
        problems.append(
            "lock без хешей: пересоберите с --generate-hashes, иначе установка "
            "не защищена от подмены пакета в индексе"
        )

    for name, spec in sorted(reqs.items()):
        if name not in pins:
            problems.append(
                f"{name}: объявлен в {args.requirements.name}, но отсутствует в lock "
                f"- lock устарел, пересоберите"
            )
            continue
        ok, why = _satisfies(pins[name], spec)
        if not ok:
            problems.append(f"{name}: в lock {why} (из {args.requirements.name}: {name}{spec})")

    if problems:
        print("Lock не согласован с зависимостями:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nПересобрать под прод-версию Python (3.11, как в deploy/gcp-app/Dockerfile):\n"
            "  uv pip compile requirements-rag.txt --generate-hashes "
            "--python-version 3.11 --output-file requirements-rag.lock",
            file=sys.stderr,
        )
        return 1

    print(
        f"lock согласован: {len(reqs)} прямых зависимостей, "
        f"{len(pins)} закреплено, хеши на месте"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
