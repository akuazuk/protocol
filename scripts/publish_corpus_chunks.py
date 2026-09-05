#!/usr/bin/env python3
"""Публикует нарезанные чанки в корпус, который читает прод.

Зачем отдельный скрипт, если есть split_chunks_jsonl.py: тот удаляет все части
и нарезает заново из output/chunks/chunks.jsonl. Ночной конвейер пишет туда
только изменённые протоколы, поэтому запуск split на проде заменил бы 104 687
чанков полного корпуса на 34 157 из последнего инкремента. Здесь наоборот:
существующие части не трогаются, новые пути дописываются отдельными частями.

Что нашлось 2026-09-05: ночной sync скачивал новые протоколы, нарезал их и
обновлял каталог, но шага публикации в corpus_chunks_parts не было вовсе.
Каталог знал про КП по артериальной гипертензии 2026 года, поэтому протокол
находился поиском, но текста в индексе не было - врачу цитировалась редакция
2017 года. Так накопилось 71 непубликованный протокол, включая АГ, ОКС, ТЭЛА,
стабильную стенокардию, рак лёгкого и неонатологию 2026 года.

Использование:

    # что будет опубликовано, без записи
    python3 scripts/publish_corpus_chunks.py --corpus <dir> --source <chunks.jsonl> --dry-run

    # опубликовать только пути, которых в корпусе ещё нет
    python3 scripts/publish_corpus_chunks.py --corpus <dir> --source <chunks.jsonl> --add-missing

    # опубликовать конкретные пути (в т.ч. переиздать уже существующие)
    python3 scripts/publish_corpus_chunks.py --corpus <dir> --source <chunks.jsonl> --paths changed.txt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

MAX_PART_BYTES = int(os.environ.get("CHUNK_PART_MAX_BYTES", str(45 * 1024 * 1024)))


def _iter_rows(path: Path):
    """Строки JSONL, устойчиво к битым и обрезанным."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"  пропущена битая строка {path.name}:{lineno}", file=sys.stderr)


def _part_files(corpus: Path) -> list[Path]:
    return sorted(corpus.glob("chunks.part.*.jsonl"))


def survey_paths(files: list[Path]) -> Counter:
    """source_path -> число чанков."""
    counts: Counter = Counter()
    for fp in files:
        for row in _iter_rows(fp):
            sp = row.get("source_path")
            if sp:
                counts[sp] += 1
    return counts


def next_part_index(corpus: Path) -> int:
    existing = _part_files(corpus)
    if not existing:
        return 0
    last = existing[-1].name.split(".")[2]
    return int(last) + 1


def write_parts(corpus: Path, rows: list[str], start_idx: int) -> list[Path]:
    """Пишет строки новыми частями, соблюдая лимит размера части."""
    written: list[Path] = []
    idx = start_idx
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size, idx
        if not buf:
            return
        dest = corpus / f"chunks.part.{idx:03d}.jsonl"
        tmp = dest.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(buf), encoding="utf-8")
        tmp.replace(dest)
        written.append(dest)
        idx += 1
        buf = []
        size = 0

    for line in rows:
        blob = line if line.endswith("\n") else line + "\n"
        encoded = len(blob.encode("utf-8"))
        if size + encoded > MAX_PART_BYTES and buf:
            flush()
        buf.append(blob)
        size += encoded
    flush()
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path, help="каталог corpus_chunks_parts")
    ap.add_argument("--source", required=True, type=Path, help="output/chunks/chunks.jsonl")
    ap.add_argument("--add-missing", action="store_true", help="опубликовать пути, которых нет в корпусе")
    ap.add_argument("--paths", type=Path, help="файл со списком source_path (по строке)")
    ap.add_argument("--dry-run", action="store_true", help="только показать план")
    ap.add_argument(
        "--manifest-script",
        type=Path,
        default=Path("scripts/build_corpus_path_manifest.py"),
        help="скрипт пересборки манифеста",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="интерпретатор для пересборки манифеста (нужны зависимости приложения)",
    )
    args = ap.parse_args()

    corpus: Path = args.corpus
    source: Path = args.source
    if not corpus.is_dir():
        print(f"нет каталога корпуса: {corpus}", file=sys.stderr)
        return 2
    if not source.is_file():
        print(f"нет исходного файла чанков: {source}", file=sys.stderr)
        return 2
    if not args.add_missing and not args.paths:
        print("укажи --add-missing или --paths", file=sys.stderr)
        return 2

    print("== что уже в корпусе ==")
    existing = survey_paths(_part_files(corpus))
    print(f"  путей {len(existing)}, чанков {sum(existing.values())}")

    print("== что в исходном файле ==")
    incoming = survey_paths([source])
    print(f"  путей {len(incoming)}, чанков {sum(incoming.values())}")

    if args.paths:
        wanted = {
            ln.strip()
            for ln in args.paths.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        targets = wanted & set(incoming)
        absent = wanted - set(incoming)
        if absent:
            print(f"  предупреждение: {len(absent)} путей нет в исходном файле")
    else:
        targets = set(incoming) - set(existing)

    if not targets:
        print("публиковать нечего - корпус уже содержит все нужные пути")
        return 0

    replacing = targets & set(existing)
    adding = targets - set(existing)
    print()
    print(f"== план: добавить {len(adding)} путей, переиздать {len(replacing)} ==")
    for sp in sorted(targets)[:10]:
        was = existing.get(sp)
        mark = f"было {was}" if was else "новый"
        print(f"  {incoming[sp]:5} чанков  ({mark})  {Path(sp).name[:70]}")
    if len(targets) > 10:
        print(f"  ... ещё {len(targets) - 10}")

    if args.dry_run:
        print("\n--dry-run: ничего не записано")
        return 0

    if replacing:
        # Переиздание требует переписать существующие части, а это риск для
        # 100k работающих чанков. Пока не понадобилось - осознанно не делаю.
        print(
            "\nпереиздание существующих путей не поддерживается: "
            "оно потребует перезаписи работающих частей корпуса.\n"
            "Убери такие пути из списка или реализуй перезапись осознанно.",
            file=sys.stderr,
        )
        return 3

    # --- отбор строк -----------------------------------------------------------
    print("\n== отбираю строки ==")
    selected: list[str] = []
    with source.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if row.get("source_path") in targets:
                selected.append(stripped)
    print(f"  отобрано строк: {len(selected)}")

    expected = sum(incoming[sp] for sp in targets)
    if len(selected) != expected:
        print(f"отобрано {len(selected)}, ожидалось {expected}", file=sys.stderr)
        return 4

    # --- бэкап манифеста -------------------------------------------------------
    manifest = corpus / "corpus_path_manifest.jsonl"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if manifest.is_file():
        backup = corpus / f"corpus_path_manifest.jsonl.before-{stamp}"
        shutil.copy2(manifest, backup)
        print(f"  манифест сохранён: {backup.name}")

    # --- запись новых частей ---------------------------------------------------
    start = next_part_index(corpus)
    written = write_parts(corpus, selected, start)
    print(f"== записано частей: {len(written)} ==")
    for p in written:
        print(f"  {p.name}  {p.stat().st_size // 1048576} МБ")

    # --- пересборка манифеста --------------------------------------------------
    # Вызываем канонический скрипт подпроцессом, а не импортом: логика
    # агрегации живёт в одном месте, а интерпретатор можно подменить. На проде
    # это важно - у системного python3 нет зависимостей приложения, они есть
    # только внутри контейнера.
    print("== пересобираю манифест ==")
    cmd = [
        args.python,
        str(args.manifest_script),
        "--corpus",
        str(corpus),
        "--output",
        str(manifest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(f"пересборка манифеста не удалась:\n{proc.stderr[-1500:]}", file=sys.stderr)
        print(
            "части уже записаны, но манифест старый - прод их не увидит.\n"
            f"Повтори вручную: {' '.join(cmd)}",
            file=sys.stderr,
        )
        return 7
    print("  " + (proc.stdout.strip().splitlines() or [""])[-1])

    # --- проверка --------------------------------------------------------------
    print("== проверяю результат ==")
    after = survey_paths(_part_files(corpus))
    print(f"  путей {len(after)} (было {len(existing)}), чанков {sum(after.values())} (было {sum(existing.values())})")

    lost = set(existing) - set(after)
    if lost:
        print(f"ПОТЕРЯНЫ пути: {len(lost)}", file=sys.stderr)
        return 5
    still_missing = targets - set(after)
    if still_missing:
        print(f"не появились: {len(still_missing)}", file=sys.stderr)
        return 6

    with manifest.open(encoding="utf-8") as fh:
        head = json.loads(fh.readline())
    print(f"  манифест: путей {head.get('paths_count')}, чанков {head.get('total_chunks')}")
    if head.get("paths_count") != len(after):
        print(
            f"манифест описывает {head.get('paths_count')} путей, а в частях {len(after)}",
            file=sys.stderr,
        )
        return 8
    print("\nготово. Перезапусти приложение, чтобы оно перечитало манифест.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
