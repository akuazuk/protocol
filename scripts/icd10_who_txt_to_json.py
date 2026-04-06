#!/usr/bin/env python3
"""
Сборка компактного JSON из WHO ICD-10 2016 meta (icd102016syst_codes.txt).
Использование: python3 scripts/icd10_who_txt_to_json.py
Выход: data/icd_reference/icd10_who_2016_terminal_codes.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/icd_reference/who_icd10_2016_en/icd102016syst_codes.txt"
OUT = ROOT / "data/icd_reference/icd10_who_2016_terminal_codes.json"


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"Нет файла: {SRC} — распакуйте icd102016enMeta.zip в data/icd_reference/")
    rows: list[dict] = []
    with SRC.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 9:
                continue
            level = parts[0]
            terminal = parts[1] == "T"
            code_dotted = parts[5].strip() if len(parts) > 5 else ""
            title = parts[8].strip() if len(parts) > 8 else ""
            if not code_dotted or not title:
                continue
            rows.append(
                {
                    "level": level,
                    "terminal": terminal,
                    "code": code_dotted,
                    "title_en": title,
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"Записано {len(rows)} записей в {OUT}")


if __name__ == "__main__":
    main()
