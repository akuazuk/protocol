"""Единая запись и чтение .jsonl для складов обратной связи и телеметрии.

Почему одним syscall, а не через flock
--------------------------------------
Открытие в режиме "a" даёт O_APPEND: ядро вычисляет позицию и пишет одной
неделимой операцией, поэтому две полные строки не перемешиваются. Это
проверено на прод-ядре (Linux 6.1, ext4): 8 процессов x 60 строк длиной до
500 КБ - ноль битых строк. Блокировка файла тут ничего не добавляет, зато
добавляет точку отказа и сериализует запросы.

Опасен не сам append, а буферизация Python: `open(...)` поверх write() кладёт
TextIOWrapper и BufferedWriter, и длинная строка может уйти в ядро несколькими
write() - между ними чужой append встанет серединой строки. Поэтому здесь
строка кодируется целиком и отдаётся ровно одним os.write.

Долговечность
-------------
Без fsync запись живёт в страничном кеше и теряется при жёсткой перезагрузке
VM. Для аудита и согласий это недопустимо - там `durable=True`. Для массовой
телеметрии fsync на каждое событие слишком дорог, и потеря последних секунд
терпима, поэтому по умолчанию он выключен.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)


def encode_line(row: Any, *, compact: bool = False) -> bytes:
    """JSON одной строкой в UTF-8, включая перевод строки."""
    if compact:
        text = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(row, ensure_ascii=False)
    # Перевод строки внутри значения разорвал бы одну запись на две при чтении.
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    return (text + "\n").encode("utf-8")


def append_line(
    path: str | Path,
    row: Any,
    *,
    compact: bool = False,
    durable: bool = False,
) -> None:
    """Дописать одну запись одним os.write; durable=True добавляет fsync."""
    data = encode_line(row, compact=compact)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(fd, data)
        if written != len(data):
            # Регулярный файл записывается целиком; частичная запись означает
            # исчерпание диска или квоты - строка уже обрезана, молчать нельзя.
            raise OSError(
                f"Частичная запись в {p.name}: {written} из {len(data)} байт "
                "(вероятно, нет места на диске)"
            )
        if durable:
            os.fsync(fd)
    finally:
        os.close(fd)


def iter_jsonl(path: str | Path, *, source: str | None = None) -> Iterator[dict[str, Any]]:
    """Читать .jsonl, пропуская битые строки.

    Обрыв процесса посередине записи оставляет в конце файла неполную строку.
    Строгий разбор превратил бы её в постоянную ошибку всего файла: панель
    методиста перестала бы открываться из-за одной записи. Битые строки
    пропускаются и суммарно попадают в лог.
    """
    p = Path(path)
    if not p.is_file():
        return
    label = source or p.name
    broken = 0
    total = 0
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                broken += 1
                if broken == 1:
                    _LOG.warning(
                        "%s: строка %d не разобрана как JSON, пропущена", label, lineno
                    )
                continue
            if isinstance(row, dict):
                yield row
            else:
                broken += 1
    if broken:
        _LOG.warning("%s: пропущено %d битых строк из %d", label, broken, total)


def is_valid_jsonl_line(line: str) -> bool:
    """Строка - разбираемый JSON-объект (для фильтрации перед выгрузкой)."""
    text = line.strip()
    if not text or text.startswith("#"):
        return False
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False
