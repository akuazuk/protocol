"""Тесты публикации чанков в корпус, который читает прод.

Контекст: 2026-09-05 нашлось, что ночной sync скачивал новые протоколы,
нарезал их и обновлял каталог, но шага публикации в corpus_chunks_parts не
было. Протокол находился поиском, а текста в индексе не было - врачу
цитировалась прошлая редакция. Накопилось 84 непубликованных пути, включая КП
по артериальной гипертензии, ОКС, ТЭЛА и неонатологии 2026 года.

Здесь закрепляются два инварианта, нарушение которых и создало бы повтор:
работающие чанки не теряются, а целевые пути действительно появляются.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publish_corpus_chunks.py"


def _chunk(path: str, idx: int, text: str = "текст протокола") -> str:
    return json.dumps(
        {
            "chunk_id": f"{path}:{idx}",
            "doc_id": f"doc-{abs(hash(path)) % 10000}",
            "source_path": path,
            "text": f"{text} {idx}",
            "chunk_type": "body",
            "page_from": idx,
            "page_to": idx,
        },
        ensure_ascii=False,
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Корпус с двумя уже работающими протоколами."""
    d = tmp_path / "corpus_chunks_parts"
    d.mkdir()
    rows = [_chunk("minzdrav_protocols/kard/старый_КП_2017.pdf", i) for i in range(5)]
    rows += [_chunk("minzdrav_protocols/endo/КП_диабет_2021.pdf", i) for i in range(3)]
    (d / "chunks.part.000.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return d


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """Свежая нарезка: новый протокол 2026 плюс уже опубликованный старый."""
    f = tmp_path / "chunks.jsonl"
    rows = [_chunk("minzdrav_protocols/kard/КП_АГ_2026.pdf", i, "гипертензия") for i in range(4)]
    rows += [_chunk("minzdrav_protocols/kard/старый_КП_2017.pdf", i) for i in range(5)]
    f.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return f


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )


def _paths_in(corpus: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fp in sorted(corpus.glob("chunks.part.*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sp = json.loads(line).get("source_path")
            if sp:
                counts[sp] = counts.get(sp, 0) + 1
    return counts


def test_dry_run_changes_nothing(corpus: Path, source: Path) -> None:
    before = _paths_in(corpus)
    r = _run("--corpus", str(corpus), "--source", str(source), "--add-missing", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "ничего не записано" in r.stdout
    assert _paths_in(corpus) == before


def test_add_missing_publishes_only_new_path(corpus: Path, source: Path, tmp_path: Path) -> None:
    manifest_stub = tmp_path / "fake_manifest.py"
    manifest_stub.write_text("import sys; sys.exit(0)\n", encoding="utf-8")

    r = _run(
        "--corpus", str(corpus),
        "--source", str(source),
        "--add-missing",
        "--manifest-script", str(manifest_stub),
    )
    # Манифест-заглушка не пишет файл, поэтому финальная сверка не пройдёт;
    # проверяем именно публикацию чанков, она происходит до сверки.
    after = _paths_in(corpus)

    assert after["minzdrav_protocols/kard/КП_АГ_2026.pdf"] == 4, r.stdout + r.stderr
    # Работающие пути не тронуты и не продублированы.
    assert after["minzdrav_protocols/kard/старый_КП_2017.pdf"] == 5
    assert after["minzdrav_protocols/endo/КП_диабет_2021.pdf"] == 3


def test_existing_parts_are_never_rewritten(corpus: Path, source: Path, tmp_path: Path) -> None:
    """Главная страховка: 100k работающих чанков не должны перезаписываться."""
    original = corpus / "chunks.part.000.jsonl"
    before_bytes = original.read_bytes()

    manifest_stub = tmp_path / "stub.py"
    manifest_stub.write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    _run(
        "--corpus", str(corpus),
        "--source", str(source),
        "--add-missing",
        "--manifest-script", str(manifest_stub),
    )

    assert original.read_bytes() == before_bytes
    assert (corpus / "chunks.part.001.jsonl").is_file()


def test_republish_is_refused(corpus: Path, source: Path, tmp_path: Path) -> None:
    """Переиздание существующего пути требует перезаписи частей - отказ, не тихая порча."""
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("minzdrav_protocols/kard/старый_КП_2017.pdf\n", encoding="utf-8")

    r = _run("--corpus", str(corpus), "--source", str(source), "--paths", str(paths_file))

    assert r.returncode == 3
    assert "переиздание" in r.stderr.lower()
    # Части не изменились.
    assert sorted(p.name for p in corpus.glob("chunks.part.*.jsonl")) == ["chunks.part.000.jsonl"]


def test_nothing_to_do_is_success(corpus: Path, tmp_path: Path) -> None:
    """Идемпотентность: повторный прогон не должен считаться ошибкой в ночном cron."""
    same = tmp_path / "same.jsonl"
    rows = [_chunk("minzdrav_protocols/kard/старый_КП_2017.pdf", i) for i in range(5)]
    same.write_text("\n".join(rows) + "\n", encoding="utf-8")

    r = _run("--corpus", str(corpus), "--source", str(same), "--add-missing")

    assert r.returncode == 0, r.stderr
    assert "публиковать нечего" in r.stdout


def test_missing_source_fails_loudly(corpus: Path, tmp_path: Path) -> None:
    r = _run("--corpus", str(corpus), "--source", str(tmp_path / "нет.jsonl"), "--add-missing")
    assert r.returncode == 2
    assert "нет исходного файла" in r.stderr


def test_requires_a_selection_mode(corpus: Path, source: Path) -> None:
    r = _run("--corpus", str(corpus), "--source", str(source))
    assert r.returncode == 2
    assert "--add-missing" in r.stderr
