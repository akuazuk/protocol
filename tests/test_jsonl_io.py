"""Запись и чтение .jsonl: одна строка = один syscall, битая строка не ломает файл."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from clinical_knowledge.jsonl_io import append_line, encode_line, is_valid_jsonl_line, iter_jsonl


def test_append_line_writes_one_syscall_per_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Длинная строка обязана уйти в ядро одним write.

    Именно это, а не блокировка файла, защищает от перемешивания: O_APPEND
    неделим только внутри одного write. Если запись разобьётся на несколько,
    чужой append встанет серединой строки.
    """
    calls: list[int] = []
    real_write = os.write

    def counting_write(fd: int, data: bytes) -> int:
        calls.append(len(data))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", counting_write)

    path = tmp_path / "events.jsonl"
    big = {"blob": "щ" * 200_000}
    append_line(path, big)

    assert len(calls) == 1, f"запись разбилась на {len(calls)} write вместо одного"
    assert json.loads(path.read_text(encoding="utf-8")) == big


def test_append_line_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "deep" / "nested" / "events.jsonl"
    append_line(path, {"a": 1})
    assert path.is_file()


def test_append_line_keeps_records_one_per_line(tmp_path: Path):
    """Перевод строки внутри значения не должен разрывать запись на две."""
    path = tmp_path / "events.jsonl"
    append_line(path, {"text": "первая\nвторая\r\nтретья"})
    append_line(path, {"text": "ok"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "многострочное значение разорвало запись"
    assert json.loads(lines[0])["text"] == "первая\nвторая\r\nтретья"


def test_append_line_durable_calls_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    synced: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append(fd))

    append_line(tmp_path / "audit.jsonl", {"a": 1}, durable=True)
    assert synced, "durable=True не вызвал fsync - запись переживёт не всякую перезагрузку"

    synced.clear()
    append_line(tmp_path / "telemetry.jsonl", {"a": 1})
    assert not synced, "fsync по умолчанию слишком дорог для массовой телеметрии"


def test_append_line_raises_on_partial_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Кончилось место - строка обрезана, и это должно быть ошибкой, а не тишиной."""
    real_write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real_write(fd, data[: len(data) // 2]))

    with pytest.raises(OSError, match="Частичная запись"):
        append_line(tmp_path / "events.jsonl", {"blob": "x" * 100})


def test_compact_encoding_has_no_spaces():
    assert encode_line({"a": 1, "b": 2}, compact=True) == b'{"a":1,"b":2}\n'
    assert b", " in encode_line({"a": 1, "b": 2})


def test_iter_jsonl_skips_truncated_tail(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Обрыв процесса посреди записи оставляет неполную строку в конце файла."""
    path = tmp_path / "events.jsonl"
    append_line(path, {"n": 1})
    append_line(path, {"n": 2})
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"n": 3, "blob": "недописан')

    with caplog.at_level(logging.WARNING):
        rows = list(iter_jsonl(path))

    assert [r["n"] for r in rows] == [1, 2], "целые записи должны читаться несмотря на обрыв"
    assert "битых строк" in caplog.text


def test_iter_jsonl_missing_file_is_empty(tmp_path: Path):
    assert list(iter_jsonl(tmp_path / "нет.jsonl")) == []


def test_iter_jsonl_ignores_comments_and_non_objects(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text('# комментарий\n\n{"n": 1}\n[1, 2]\n"строка"\n', encoding="utf-8")
    assert [r["n"] for r in iter_jsonl(path)] == [1]


def test_is_valid_jsonl_line():
    assert is_valid_jsonl_line('{"a": 1}')
    assert not is_valid_jsonl_line('{"a": 1')
    assert not is_valid_jsonl_line("")
    assert not is_valid_jsonl_line("# comment")
    assert not is_valid_jsonl_line("[1, 2]")


def test_concurrent_appends_do_not_interleave(tmp_path: Path):
    """Прод-инвариант: параллельная запись длинных строк не портит файл.

    Проверено на прод-ядре из нескольких процессов; здесь - потоками, чтобы
    тест был быстрым и работал на любой ОС.
    """
    import threading

    path = tmp_path / "events.jsonl"
    payload = "щ" * 30_000
    per_thread = 25
    threads = [
        threading.Thread(
            target=lambda tid=tid: [
                append_line(path, {"tid": tid, "i": i, "blob": payload})
                for i in range(per_thread)
            ]
        )
        for tid in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 8 * per_thread
    assert all(r["blob"] == payload for r in rows)
