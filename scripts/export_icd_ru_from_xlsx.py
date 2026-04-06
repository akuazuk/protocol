#!/usr/bin/env python3
"""Собрать JSON русского справочника МКБ-10 из Excel (единственный источник кодов для лексикона).

Использование:
  Положите официальный .xlsx в data/icd_reference/ (например mkb10_minzdrav_rb.xlsx) и выполните:
    ICD_RU_XLSX=data/icd_reference/mkb10_minzdrav_rb.xlsx python3 scripts/export_icd_ru_from_xlsx.py

Переменные окружения:
  ICD_RU_XLSX — путь к .xlsx (по умолчанию data/icd_reference/mkb10_ru_mkb10su.xlsx)
  ICD_RU_JSON — куда записать JSON (по умолчанию data/icd_reference/icd10_ru_mkb10su.json)

Формат таблицы как у прежнего mkb10.su: колонка A — код, B — название (строки данных с 5-й).
Без openpyxl: чтение через zip+xml (stdlib).
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XLSX = Path(os.environ.get("ICD_RU_XLSX") or "data/icd_reference/mkb10_ru_mkb10su.xlsx")
if not XLSX.is_absolute():
    XLSX = ROOT / XLSX
OUT = Path(os.environ.get("ICD_RU_JSON") or "data/icd_reference/icd10_ru_mkb10su.json")
if not OUT.is_absolute():
    OUT = ROOT / OUT


def main() -> int:
    if not XLSX.is_file():
        print("Нет файла Excel:", XLSX, file=sys.stderr)
        print("Положите справочник МКБ-10 (.xlsx) и задайте ICD_RU_XLSX при необходимости.", file=sys.stderr)
        return 1
    z = zipfile.ZipFile(XLSX)
    shared = z.read("xl/sharedStrings.xml")
    root = ET.fromstring(shared)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for si in root.findall(".//m:si", ns):
        strings.append("".join(si.itertext()))
    sheet = z.read("xl/worksheets/sheet1.xml")
    r = ET.fromstring(sheet)
    rows = r.findall(".//m:row", ns)

    def cell_value(c: ET.Element) -> str:
        v = c.find("m:v", ns)
        if v is None:
            return ""
        if c.get("t") == "s":
            return strings[int(v.text)]
        return v.text or ""

    out: list[dict] = []
    for row in rows[4:]:
        cells = row.findall("m:c", ns)
        if len(cells) < 2:
            continue
        code = cell_value(cells[0]).strip()
        name = cell_value(cells[1]).strip()
        if not code or not name:
            continue
        if code.startswith("http"):
            continue
        out.append({"code": code, "title_ru": name})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("OK", len(out), "rows ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
