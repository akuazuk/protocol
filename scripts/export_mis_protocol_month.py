#!/usr/bin/env python3
"""Выгрузка mis_protocol за период: result разобран по столбцам схемы EPAM.

Пароль: ~/CURSOR/sql_epam/.env → KRAVIRA_DB_PASSWORD
Схема: epam/scheme_mis_protocols.docx / clinical_knowledge/mis_protocol_parse.py

Пример:
  python3 scripts/export_mis_protocol_month.py --month 2026-07
  python3 scripts/export_mis_protocol_month.py --from 2026-07-01 --to 2026-08-01
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SQL_EPAM = Path.home() / "CURSOR" / "sql_epam"
DEFAULT_OUT = ROOT / "data" / "mis_protocol"


def _load_parse_result():
    """Импорт без clinical_knowledge.__init__ (там pydantic и прочий стек)."""
    import importlib.util

    path = ROOT / "clinical_knowledge" / "mis_protocol_parse.py"
    spec = importlib.util.spec_from_file_location("mis_protocol_parse", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.parse_result


def _engine():
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv(SQL_EPAM / ".env")
    pw = (os.environ.get("KRAVIRA_DB_PASSWORD") or "").strip()
    if not pw:
        raise SystemExit("Нет KRAVIRA_DB_PASSWORD в ~/CURSOR/sql_epam/.env")
    url = (
        f"mysql+pymysql://kravira_mc_user:{pw}@178.163.240.131:6330/kravira_mc"
        "?charset=utf8mb4"
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 30,
            "read_timeout": int(os.environ.get("MIS_DB_READ_TIMEOUT", "600")),
        },
    )


def _month_bounds(ym: str) -> tuple[str, str]:
    start = datetime.strptime(ym + "-01", "%Y-%m-%d").date()
    if start.month == 12:
        end = date(start.year + 1, 1, 1)
    else:
        end = date(start.year, start.month + 1, 1)
    return start.isoformat(), end.isoformat()


def main() -> int:
    import pandas as pd
    from sqlalchemy import text

    parse_result = _load_parse_result()

    ap = argparse.ArgumentParser()
    ap.add_argument("--month", type=str, default="", help="YYYY-MM (по умолчанию текущий)")
    ap.add_argument("--from", dest="date_from", type=str, default="")
    ap.add_argument("--to", dest="date_to", type=str, default="", help="exclusive")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.date_from and args.date_to:
        d0, d1 = args.date_from, args.date_to
        tag = f"{d0}_{d1}"
    else:
        ym = args.month or date.today().strftime("%Y-%m")
        d0, d1 = _month_bounds(ym)
        tag = ym

    print(f"Fetching mis_protocol [{d0}, {d1}) …", flush=True)
    engine = _engine()
    q = text(
        "SELECT id, date, visit_id, patient_id, result "
        "FROM mis_protocol "
        "WHERE date >= :d0 AND date < :d1 "
        "ORDER BY id"
    )
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"d0": d0, "d1": d1})
    engine.dispose()
    print(f"rows: {len(df)}", flush=True)

    parsed = df["result"].map(parse_result)
    parsed_df = pd.DataFrame(list(parsed))
    out = pd.concat(
        [
            df[["id", "date", "visit_id", "patient_id"]].reset_index(drop=True),
            parsed_df.reset_index(drop=True),
        ],
        axis=1,
    )
    # Сырой result оставляем для отладки, но в конце.
    out["result_raw"] = df["result"].values

    # ФИО врача из mis_data по visit_id (specialist_name).
    print("Joining mis_data specialist_name by visit_id …", flush=True)
    vids = out["visit_id"].dropna().astype(int).unique().tolist()
    chunks: list[pd.DataFrame] = []
    engine = _engine()
    with engine.connect() as conn:
        for i in range(0, len(vids), 800):
            batch = vids[i : i + 800]
            ph = ",".join(str(int(x)) for x in batch)
            q_doc = text(
                f"""
                SELECT visit_id,
                       MIN(specialist_id) AS specialist_id_from_visit,
                       MIN(specialist_name) AS doctor_fio,
                       MIN(specialization) AS doctor_specialization
                FROM mis_data
                WHERE visit_id IN ({ph})
                GROUP BY visit_id
                """
            )
            chunks.append(pd.read_sql(q_doc, conn))
    engine.dispose()
    if chunks:
        doc = pd.concat(chunks, ignore_index=True)
        out = out.merge(doc, on="visit_id", how="left")
        # doctor_fio рядом с doctor_id
        cols = list(out.columns)
        for c in ("doctor_fio", "doctor_specialization", "specialist_id_from_visit"):
            if c in cols:
                cols.remove(c)
        if "doctor_id" in cols:
            i = cols.index("doctor_id") + 1
            cols[i:i] = ["doctor_fio", "doctor_specialization", "specialist_id_from_visit"]
        else:
            cols += ["doctor_fio", "doctor_specialization", "specialist_id_from_visit"]
        out = out[cols]
        fio_n = int(out["doctor_fio"].fillna("").astype(str).str.strip().ne("").sum())
        print(f"doctor_fio filled: {fio_n}/{len(out)} ({100 * fio_n / max(1, len(out)):.1f}%)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / f"mis_protocol_{tag}.parquet"
    csv_path = args.out_dir / f"mis_protocol_{tag}.csv"
    meta_path = args.out_dir / f"mis_protocol_{tag}.meta.json"

    out.to_parquet(parquet_path, index=False)
    # CSV без сырого result (тяжелее и с переносами) - для просмотра.
    view_cols = [c for c in out.columns if c not in ("result_raw", "diagnosis_structured_raw")]
    out[view_cols].to_csv(csv_path, index=False, encoding="utf-8")

    import json

    meta = {
        "date_from": d0,
        "date_to_exclusive": d1,
        "rows": int(len(out)),
        "columns": list(out.columns),
        "doctor_fio_filled": int(out["doctor_fio"].fillna("").astype(str).str.strip().ne("").sum())
        if "doctor_fio" in out.columns
        else 0,
        "doctor_fio_source": "mis_data.specialist_name via visit_id",
        "parquet": str(parquet_path.relative_to(ROOT)),
        "csv": str(csv_path.relative_to(ROOT)),
        "source": "kravira_mc.mis_protocol + mis_data",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {parquet_path} ({parquet_path.stat().st_size // 1024} KB)")
    print(f"Wrote {csv_path} ({csv_path.stat().st_size // 1024} KB)")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
