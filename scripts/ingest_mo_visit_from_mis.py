#!/usr/bin/env python3
"""Подгрузить один визит МИС в secure_cases + cases + warehouse МО.

Запуск только на GCE (не с Mac к БД):

  sudo docker cp scripts/ingest_mo_visit_from_mis.py protocol-web:/tmp/
  sudo docker cp /opt/protocol/.env.mis protocol-web:/tmp/.env.mis
  sudo docker exec -w /app -e MO_DATA_ROOT=/var/data/medical_exams protocol-web \\
    python /tmp/ingest_mo_visit_from_mis.py --visit-id 3468853
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _slot(parts: list[str], idx: int) -> str:
    return (parts[idx] if idx < len(parts) else "").strip()


def fetch_visit(visit_id: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    import pymysql

    pw = (os.environ.get("KRAVIRA_DB_PASSWORD") or "").strip()
    con = pymysql.connect(
        host=os.environ.get("KRAVIRA_DB_HOST") or "178.163.240.131",
        port=int(os.environ.get("KRAVIRA_DB_PORT") or 6330),
        user=os.environ.get("KRAVIRA_DB_USER") or "kravira_mc_user",
        password=pw,
        database=os.environ.get("KRAVIRA_DB_NAME") or "kravira_mc",
        charset="utf8mb4",
        connect_timeout=30,
        read_timeout=60,
    )
    cur = con.cursor()
    cur.execute(
        """
        SELECT visit_id, vdate, specialist_id, specialist_name, specialization,
               patient_id, patient_bdate, diagnos, filial, pay_type, serv_name
        FROM mis_data WHERE visit_id=%s LIMIT 1
        """,
        (int(visit_id),),
    )
    meta = cur.fetchone()
    if not meta:
        con.close()
        raise SystemExit(f"visit_not_found_in_mis_data:{visit_id}")
    cur.execute(
        """
        SELECT id, date, visit_id, patient_id, result
        FROM mis_protocol WHERE visit_id=%s ORDER BY id DESC LIMIT 5
        """,
        (int(visit_id),),
    )
    protocols = cur.fetchall()
    con.close()
    if not protocols:
        raise SystemExit(f"visit_not_found_in_mis_protocol:{visit_id}")

    rows: list[dict[str, str]] = []
    for prot in protocols:
        parts = str(prot[4] or "").split("::")
        visit_date = str(prot[1] or meta[1])[:10]
        bdate = str(meta[6] or "")[:10]
        age = ""
        if bdate and visit_date:
            try:
                born = date.fromisoformat(bdate)
                day = date.fromisoformat(visit_date)
                age = str((day - born).days // 365)
            except ValueError:
                age = ""
        rows.append(
            {
                "id": str(prot[0]),
                "date": visit_date,
                "visit_id": str(prot[2]),
                "patient_id": str(prot[3] or meta[5]),
                "doctor_id": str(meta[2] or ""),
                "doctor_fio": str(meta[3] or ""),
                "doctor_specialization": str(meta[4] or ""),
                "specialist_id_from_visit": str(meta[2] or ""),
                "filial": str(meta[8] or ""),
                "pay_type": str(meta[9] or ""),
                "service_names": str(meta[10] or ""),
                "patient_bdate": bdate,
                "patient_age_years": age,
                "mis_diagnos": str(meta[7] or ""),
                "complaints": _slot(parts, 3),
                "objective_status": _slot(parts, 4),
                "clinical_diagnosis": _slot(parts, 5),
                "exam_recommendations": _slot(parts, 6),
                "anamnesis_doctor": _slot(parts, 10),
                "exam_data": _slot(parts, 11),
                "manipulations": _slot(parts, 24),
                "treatment_recommendations": _slot(parts, 26),
                "parse_ok": "1",
                "visit_date": visit_date,
            }
        )
    header = {
        "visit_id": str(meta[0]),
        "visit_date": str(meta[1])[:10],
        "patient_id": str(meta[5]),
        "doctor": str(meta[3]),
        "specialty": str(meta[4]),
    }
    return header, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visit-id", required=True)
    ap.add_argument("--data-root", default=os.environ.get("MO_DATA_ROOT") or "/var/data/medical_exams")
    ap.add_argument("--env-file", default="/tmp/.env.mis")
    args = ap.parse_args()
    _load_env(Path(args.env_file))
    _load_env(Path("/opt/protocol/.env.mis"))

    header, rows = fetch_visit(str(args.visit_id))
    day = header["visit_date"]
    data_root = Path(args.data_root)
    secure = data_root / "secure_cases" / day[:4] / day[5:7]
    secure.mkdir(parents=True, exist_ok=True)
    csv_path = secure / f"mo_{day}.csv"

    # Merge into day CSV if exists.
    existing: dict[str, dict[str, str]] = {}
    fieldnames: list[str] = list(rows[0].keys())
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(dict.fromkeys([*(reader.fieldnames or []), *fieldnames]))
            for row in reader:
                key = str(row.get("id") or row.get("visit_id") or "")
                if key:
                    existing[key] = row
    for row in rows:
        existing[str(row["id"])] = {**existing.get(str(row["id"]), {}), **row}

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing.values():
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "run_mis_protocol_l1_batch.py",
            "--csv",
            str(csv_path),
            "--out-dir",
            str(secure),
            "--month",
            day,
            "--direct",
            "--deep-eval",
            "--resume",
            "--workers",
            os.environ.get("MO_DAILY_WORKERS") or "1",
        ]
        from scripts.run_mis_protocol_l1_batch import main as batch_main

        rc = batch_main()
    finally:
        sys.argv = old_argv
    if rc not in (0, None):
        return int(rc or 1)

    from scripts.recompute_mo_days import recompute_day

    result = recompute_day(
        date.fromisoformat(day),
        data_root=data_root,
        warehouse=data_root / "warehouse" / "mo_analytics.sqlite",
        write_reports=True,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "header": header,
                "csv": str(csv_path),
                "rows_in_day_csv": len(existing),
                "recompute": result,
                "ingested_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
