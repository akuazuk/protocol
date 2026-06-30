#!/usr/bin/env python3
"""Рекалибровка возрастных priors онкориска из источника CI5 / РНПЦ (Фаза 4).

НЕ выдумывает числа: читает ваш экспорт и записывает data/onco_risk/priors_age_belarus.yaml,
который движок (clinical_knowledge/onco_risk.py) подхватывает автоматически (возраст-специфичный
baseline вместо общего). Пока файла нет - поведение движка не меняется.

Формат входа (CSV или JSON-список объектов), колонки/ключи:
    site            - идентификатор локализации (как в priors_belarus.yaml: colorectal, lung, ...)
    age_min, age_max- границы возрастной полосы (включительно)
    baseline_symptomatic  - pre-test вероятность (0..0.5)   [вариант A, рекомендуется]
        ИЛИ
    rate_per_100k   - годовая заболеваемость на 100 000     [вариант B, с флагом --from-rate;
                      baseline := rate_per_100k / 100000, помечается как crude-proxy]
    sex (опц.)      - any|male|female
    source (опц.)   - ссылка на источник строки

Примеры:
    python3 scripts/onco_priors_recalibrate.py --source ci5_belarus.csv            # dry-run
    python3 scripts/onco_priors_recalibrate.py --source ci5_belarus.csv --write
    python3 scripts/onco_priors_recalibrate.py --source rnpc.csv --from-rate --write
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from clinical_knowledge import onco_risk as orisk  # noqa: E402

OUT_PATH = ROOT / "data" / "onco_risk" / "priors_age_belarus.yaml"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit("JSON должен быть списком объектов.")
        return [dict(x) for x in data]
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _to_band(row: dict[str, Any], from_rate: bool) -> tuple[str, dict[str, Any]]:
    site = str(row.get("site") or "").strip()
    if not site:
        raise ValueError("пустой site")
    age_min = int(float(row["age_min"]))
    age_max = int(float(row["age_max"]))
    if age_min > age_max:
        raise ValueError(f"{site}: age_min>age_max ({age_min}>{age_max})")
    if from_rate:
        rate = float(row["rate_per_100k"])
        baseline = rate / 100000.0
        note = "crude-proxy from rate_per_100k"
    else:
        baseline = float(row["baseline_symptomatic"])
        note = None
    if not (0 < baseline < 0.5):
        raise ValueError(f"{site} [{age_min}-{age_max}]: baseline вне диапазона ({baseline})")
    band: dict[str, Any] = {
        "age_min": age_min,
        "age_max": age_max,
        "baseline_symptomatic": round(baseline, 6),
    }
    if row.get("sex"):
        band["sex"] = str(row["sex"]).strip()
    if row.get("source"):
        band["source"] = str(row["source"]).strip()
    if note:
        band["note"] = note
    return site, band


def main() -> int:
    ap = argparse.ArgumentParser(description="Рекалибровка возрастных priors онкориска.")
    ap.add_argument("--source", required=True, help="CSV или JSON с возрастными данными")
    ap.add_argument("--from-rate", action="store_true", help="baseline := rate_per_100k/100000")
    ap.add_argument("--write", action="store_true", help="записать файл (иначе dry-run)")
    ap.add_argument("--source-ref", default="", help="общая ссылка на источник (CI5/РНПЦ)")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_file():
        raise SystemExit(f"Файл не найден: {src}")

    rows = _read_rows(src)
    if not rows:
        raise SystemExit("Источник пуст.")

    known_sites = set((orisk._priors().get("sites") or {}).keys())
    sites: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    unknown: set[str] = set()

    for i, row in enumerate(rows, 1):
        try:
            site, band = _to_band(row, args.from_rate)
        except Exception as e:  # noqa: BLE001
            errors.append(f"строка {i}: {e}")
            continue
        if site not in known_sites:
            unknown.add(site)
        sites.setdefault(site, {"bands": []})["bands"].append(band)

    if errors:
        for e in errors:
            print("[ERROR]", e)
        raise SystemExit(f"Прервано: {len(errors)} ошибок во входных данных.")

    if unknown:
        print("[warn] сайты вне priors_belarus.yaml (будут проигнорированы движком):",
              ", ".join(sorted(unknown)))

    doc = {
        "version": _dt.date.today().isoformat(),
        "source": {"ref": args.source_ref or src.name,
                   "ingested_from": src.name,
                   "from_rate": bool(args.from_rate)},
        "sites": sites,
    }

    print(f"=== Рекалибровка: {len(sites)} локализаций, {sum(len(v['bands']) for v in sites.values())} полос ===")
    for site, v in sorted(sites.items()):
        bands = ", ".join(
            f"{b['age_min']}-{b['age_max']}:{b['baseline_symptomatic']}" for b in v["bands"]
        )
        print(f"  {site}: {bands}")

    if not args.write:
        print(f"\nDRY-RUN. Для записи в {OUT_PATH.relative_to(ROOT)} добавьте --write.")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        fh.write("# Возраст-специфичные baselines онкориска (Фаза 4). Сгенерировано onco_priors_recalibrate.py.\n")
        yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False)
    print(f"\nЗаписано: {OUT_PATH.relative_to(ROOT)}")
    print("Движок подхватит возрастные baselines автоматически. Перезапустите тесты валидации.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
