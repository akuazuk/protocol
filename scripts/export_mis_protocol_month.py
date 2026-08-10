#!/usr/bin/env python3
"""Выгрузка mis_protocol за период: result разобран по столбцам схемы EPAM.

Пароль (канон E2): env `KRAVIRA_DB_PASSWORD` на GCE (`/opt/protocol/.env.mis`
или `.env.gcp-staging`). Fallback Mac: `~/CURSOR/sql_epam/.env`.
Опционально: `KRAVIRA_DB_HOST`, `KRAVIRA_DB_PORT`, `KRAVIRA_DB_USER`, `KRAVIRA_DB_NAME`.
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
from urllib.parse import quote_plus

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SQL_EPAM = Path.home() / "CURSOR" / "sql_epam"
DEFAULT_OUT = ROOT / "data" / "mis_protocol"

# Defaults = Kravira MariaDB (allowlist from GCE verified 2026-08-10).
_DEFAULT_HOST = "178.163.240.131"
_DEFAULT_PORT = "6330"
_DEFAULT_USER = "kravira_mc_user"
_DEFAULT_NAME = "kravira_mc"


def _load_parse_module():
    """Импорт без clinical_knowledge.__init__ (там pydantic и прочий стек)."""
    import importlib.util

    path = ROOT / "clinical_knowledge" / "mis_protocol_parse.py"
    spec = importlib.util.spec_from_file_location("mis_protocol_parse", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _engine():
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    # Process env first (GCE docker --env-file). Mac fallback only if empty.
    if not (os.environ.get("KRAVIRA_DB_PASSWORD") or "").strip():
        sql_env = SQL_EPAM / ".env"
        if sql_env.is_file():
            load_dotenv(sql_env)
    pw = (os.environ.get("KRAVIRA_DB_PASSWORD") or "").strip()
    if not pw:
        raise SystemExit(
            "Нет KRAVIRA_DB_PASSWORD (GCE env / --env-file или ~/CURSOR/sql_epam/.env)"
        )
    host = (os.environ.get("KRAVIRA_DB_HOST") or _DEFAULT_HOST).strip()
    port = (os.environ.get("KRAVIRA_DB_PORT") or _DEFAULT_PORT).strip()
    user = (os.environ.get("KRAVIRA_DB_USER") or _DEFAULT_USER).strip()
    name = (os.environ.get("KRAVIRA_DB_NAME") or _DEFAULT_NAME).strip()
    url = (
        f"mysql+pymysql://{quote_plus(user)}:{quote_plus(pw)}"
        f"@{host}:{port}/{name}?charset=utf8mb4"
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": int(os.environ.get("MIS_DB_CONNECT_TIMEOUT", "30")),
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
    from sqlalchemy import bindparam, text

    parse_mod = _load_parse_module()
    parse_result = parse_mod.parse_result
    classify_kz_kind = parse_mod.classify_kz_kind

    ap = argparse.ArgumentParser()
    ap.add_argument("--month", type=str, default="", help="YYYY-MM (по умолчанию текущий)")
    ap.add_argument("--from", dest="date_from", type=str, default="")
    ap.add_argument("--to", dest="date_to", type=str, default="", help="exclusive")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    # Всегда абсолютный out_dir - иначе relative_to(ROOT) падает на относительном пути.
    args.out_dir = args.out_dir.expanduser().resolve()

    if args.date_from and args.date_to:
        d0, d1 = args.date_from, args.date_to
        tag = f"{d0}_{d1}"
    else:
        ym = args.month or date.today().strftime("%Y-%m")
        d0, d1 = _month_bounds(ym)
        tag = ym

    print(f"Fetching mis_protocol [{d0}, {d1}) …", flush=True)
    engine = _engine()
    # В живой БД Kravira у mis_protocol только id/date/visit_id/patient_id/result
    # (колонки `type` из EPAM-схемы нет). doc_type заполняем NULL; разделение
    # КЗ/не-КЗ - через classify_kz_kind по специальности/содержимому.
    q = text(
        "SELECT id, date, visit_id, patient_id, result "
        "FROM mis_protocol "
        "WHERE date >= :d0 AND date < :d1 "
        "ORDER BY id"
    )
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"d0": d0, "d1": d1})
    engine.dispose()
    source_rows = int(len(df))
    print(f"rows: {len(df)}", flush=True)
    df["doc_type"] = pd.NA
    type_counts: dict = {"<absent_in_db>": int(len(df))}
    print(
        "doc_type: колонки type нет в БД - оставляем пустым; "
        "kz_kind считается эвристикой по спец./содержимому",
        flush=True,
    )

    parsed = df["result"].map(parse_result)
    parsed_df = pd.DataFrame(list(parsed))
    out = pd.concat(
        [
            df[["id", "date", "visit_id", "patient_id", "doc_type"]].reset_index(drop=True),
            parsed_df.reset_index(drop=True),
        ],
        axis=1,
    )
    # Сырой result оставляем для отладки, но в конце.
    out["result_raw"] = df["result"].values

    # Реквизиты визита из mis_data по visit_id (1 строка протокола = 1 визит).
    # Услуг на визит может быть несколько → code/serv_name склеиваем через « | ».
    print("Joining mis_data (doctor, pay, filial, services) by visit_id …", flush=True)
    vids = out["visit_id"].dropna().astype(int).unique().tolist()
    chunks: list[pd.DataFrame] = []
    engine = _engine()
    with engine.connect() as conn:
        for i in range(0, len(vids), 800):
            batch = vids[i : i + 800]
            q_doc = text(
                """
                SELECT visit_id,
                       MIN(specialist_id) AS specialist_id_from_visit,
                       MIN(specialist_name) AS doctor_fio,
                       MIN(specialization) AS doctor_specialization,
                       MIN(pay_type) AS pay_type,
                       MIN(filial) AS filial,
                       MIN(vdate) AS mis_vdate,
                       MIN(vtime) AS mis_vtime,
                       MIN(patient_bdate) AS patient_bdate,
                       GROUP_CONCAT(DISTINCT NULLIF(TRIM(diagnos), '')
                                    ORDER BY diagnos SEPARATOR ' | ') AS mis_diagnos,
                       GROUP_CONCAT(DISTINCT NULLIF(TRIM(code), '')
                                    ORDER BY code SEPARATOR ' | ') AS service_codes,
                       GROUP_CONCAT(DISTINCT NULLIF(TRIM(serv_name), '')
                                    ORDER BY serv_name SEPARATOR ' | ') AS service_names,
                       COUNT(*) AS service_row_count
                FROM mis_data
                WHERE visit_id IN :visit_ids
                GROUP BY visit_id
                """
            ).bindparams(bindparam("visit_ids", expanding=True))
            chunks.append(pd.read_sql(q_doc, conn, params={"visit_ids": [int(x) for x in batch]}))
    engine.dispose()
    if chunks:
        doc = pd.concat(chunks, ignore_index=True)
        drop_old = [
            "doctor_fio",
            "doctor_specialization",
            "specialist_id_from_visit",
            "pay_type",
            "filial",
            "mis_vdate",
            "mis_vtime",
            "patient_bdate",
            "mis_diagnos",
            "service_codes",
            "service_names",
            "service_row_count",
        ]
        out = out.drop(columns=[c for c in drop_old if c in out.columns], errors="ignore")
        out = out.merge(doc, on="visit_id", how="left")
        # порядок: после doctor_id
        cols = list(out.columns)
        extra = [
            "doctor_fio",
            "doctor_specialization",
            "specialist_id_from_visit",
            "pay_type",
            "filial",
            "mis_vdate",
            "mis_vtime",
            "patient_bdate",
            "mis_diagnos",
            "service_codes",
            "service_names",
            "service_row_count",
        ]
        for c in extra:
            if c in cols:
                cols.remove(c)
        if "doctor_id" in cols:
            i = cols.index("doctor_id") + 1
            cols[i:i] = extra
        else:
            cols += extra
        out = out[cols]
        fio_n = int(out["doctor_fio"].fillna("").astype(str).str.strip().ne("").sum())
        print(f"doctor_fio filled: {fio_n}/{len(out)} ({100 * fio_n / max(1, len(out)):.1f}%)", flush=True)
        print(f"rows after join (must equal protocol): {len(out)}", flush=True)

    # Специальность АВТОРА протокола по doctor_id (слот 7) - точнее лоссового MIN() по визиту.
    # Матчим mis_data.(visit_id, specialist_id) -> specialization; при совпадении переопределяем.
    try:
        print("Resolving author specialization by protocol doctor_id …", flush=True)
        author_map: dict[tuple[int, str], str] = {}
        engine = _engine()
        with engine.connect() as conn:
            for i in range(0, len(vids), 800):
                batch = vids[i : i + 800]
                q_sp = text(
                    """
                    SELECT visit_id, specialist_id, MIN(specialization) AS specialization
                    FROM mis_data
                    WHERE visit_id IN :visit_ids AND specialist_id IS NOT NULL
                    GROUP BY visit_id, specialist_id
                    """
                ).bindparams(bindparam("visit_ids", expanding=True))
                sp = pd.read_sql(q_sp, conn, params={"visit_ids": [int(x) for x in batch]})
                for _, r in sp.iterrows():
                    spv = str(r.get("specialization") or "").strip()
                    if not spv:
                        continue
                    author_map[(int(r["visit_id"]), str(r["specialist_id"]).strip())] = spv
        engine.dispose()

        def _norm_id(v) -> str:
            s = str(v or "").strip()
            if s.endswith(".0"):
                s = s[:-2]
            return s

        out["doctor_specialization_visit_min"] = out["doctor_specialization"]
        resolved = 0

        def _author_spec(row) -> str:
            nonlocal resolved
            vid = row.get("visit_id")
            did = _norm_id(row.get("doctor_id"))
            try:
                key = (int(vid), did)
            except (TypeError, ValueError):
                return row.get("doctor_specialization")
            spv = author_map.get(key)
            if spv:
                resolved += 1
                return spv
            return row.get("doctor_specialization")

        out["doctor_specialization"] = out.apply(_author_spec, axis=1)
        print(f"author specialization resolved for {resolved}/{len(out)} rows", flush=True)
    except Exception as e:  # noqa: BLE001 - экспортёр не должен падать из-за резолва
        print(f"WARN author specialization resolve skipped: {e}", flush=True)

    # --- Каноническая дата визита (ISO) + валидация согласованности источников ---
    # Три источника: mis_protocol.date (ISO), слот ::1 visit_date_text (ДД.ММ.ГГГГ),
    # mis_data.vdate (join). Даты должны совпадать; формат приводим к ISO.
    to_iso_date = parse_mod.to_iso_date
    d_db = out["date"].map(to_iso_date) if "date" in out.columns else ""
    d_slot = out["visit_date_text"].map(to_iso_date) if "visit_date_text" in out.columns else ""
    d_mis = out["mis_vdate"].map(to_iso_date) if "mis_vdate" in out.columns else ""
    out["visit_date_iso_db"] = d_db
    out["visit_date_iso_slot"] = d_slot
    out["visit_date_iso_mis"] = d_mis

    def _canon_date(row) -> str:
        # Приоритет: слот КЗ (ДД.ММ.ГГГГ, то что видит врач) → date БД → mis_data.
        for k in ("visit_date_iso_slot", "visit_date_iso_db", "visit_date_iso_mis"):
            v = str(row.get(k) or "").strip()
            if v:
                return v
        return ""

    out["visit_date"] = out.apply(_canon_date, axis=1)

    def _date_mismatch(row) -> str:
        vals = {
            str(row.get(k) or "").strip()
            for k in ("visit_date_iso_db", "visit_date_iso_slot", "visit_date_iso_mis")
            if str(row.get(k) or "").strip()
        }
        return "1" if len(vals) > 1 else "0"

    out["date_mismatch"] = out.apply(_date_mismatch, axis=1)
    n_mismatch = int((out["date_mismatch"] == "1").sum())
    print(f"date_mismatch (источники не совпали): {n_mismatch}/{len(out)}", flush=True)

    # --- Возраст пациента на дату визита (для доз/педиатрии) ---
    def _age_years(row) -> object:
        bd = to_iso_date(row.get("patient_bdate"))
        vd = str(row.get("visit_date") or "").strip()
        if not bd or not vd:
            return pd.NA
        try:
            b = datetime.strptime(bd, "%Y-%m-%d").date()
            v = datetime.strptime(vd, "%Y-%m-%d").date()
        except ValueError:
            return pd.NA
        years = v.year - b.year - ((v.month, v.day) < (b.month, b.day))
        return years if 0 <= years <= 120 else pd.NA

    if "patient_bdate" in out.columns:
        out["patient_age_years"] = out.apply(_age_years, axis=1)

    # --- Кросс-чек кода МКБ: inline (слот 22) ↔ mis_data.diagnos ---
    def _mis_first_code(v) -> str:
        return parse_mod.extract_mkb_code(str(v or "").split(" | ")[0])

    if "mis_diagnos" in out.columns:
        out["mkb_code_mis"] = out["mis_diagnos"].map(_mis_first_code)

        def _mkb_agree(row) -> str:
            a = str(row.get("mkb_code_main") or "").strip().upper()
            b = str(row.get("mkb_code_mis") or "").strip().upper()
            if not a or not b:
                return "unknown"
            if a == b:
                return "match"
            # Совпадение по 3-значной рубрике (K29 == K29.3) считаем частичным.
            if a.split(".")[0] == b.split(".")[0]:
                return "partial"
            return "mismatch"

        out["mkb_code_agreement"] = out.apply(_mkb_agree, axis=1)

    # Классификация каждой строки: КЗ / справка / диагностика / неклиническое / пустое.
    kinds = out.apply(lambda r: classify_kz_kind(r.to_dict()), axis=1)
    out["kz_kind"] = [k for k, _ in kinds]
    out["kz_exclude_reason"] = [reason for _, reason in kinds]
    # Флаги для отбора.
    # is_scored - строка идёт в оценку качества КЗ (консультации + медосмотры/справки).
    # is_clinical - специальность врачебная клиническая (не УЗИ/лаб./медсестра); при пустом
    #   содержании остаётся clinical, но не scored.
    out["is_scored"] = out["kz_kind"].isin(["kz", "certificate"]).map({True: "1", False: "0"})
    out["is_clinical"] = out["kz_kind"].isin(["kz", "certificate", "empty"]).map(
        {True: "1", False: "0"}
    )
    kz_kind_counts = out["kz_kind"].value_counts(dropna=False).to_dict()
    scored_kinds = {"kz", "certificate"}
    n_scored = int(out["kz_kind"].isin(scored_kinds).sum())
    print(
        f"kz_kind: {kz_kind_counts} | scored (kz+certificate): {n_scored}/{len(out)}",
        flush=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / f"mis_protocol_{tag}.parquet"
    csv_path = args.out_dir / f"mis_protocol_{tag}.csv"
    meta_path = args.out_dir / f"mis_protocol_{tag}.meta.json"

    out.to_parquet(parquet_path, index=False)
    # CSV без сырого result (тяжелее и с переносами) - для просмотра.
    view_cols = [c for c in out.columns if c not in ("result_raw", "diagnosis_structured_raw")]
    out[view_cols].to_csv(csv_path, index=False, encoding="utf-8")

    import json

    def _col_nonempty(col: str) -> int:
        if col not in out.columns:
            return 0
        return int(out[col].fillna("").astype(str).str.strip().ne("").sum())

    mkb_agree_counts = (
        out["mkb_code_agreement"].value_counts(dropna=False).to_dict()
        if "mkb_code_agreement" in out.columns
        else {}
    )
    parse_ok_n = (
        int((out["parse_ok"].astype(str) == "1").sum()) if "parse_ok" in out.columns else 0
    )
    age_n = (
        int(out["patient_age_years"].notna().sum())
        if "patient_age_years" in out.columns
        else 0
    )

    meta = {
        "date_from": d0,
        "date_to_exclusive": d1,
        "source_rows": source_rows,
        "rows": int(len(out)),
        "row_parity": source_rows == int(len(out)),
        "columns": list(out.columns),
        "date_validation": {
            "canonical_column": "visit_date (ISO)",
            "sources": ["visit_date_iso_slot (::1)", "visit_date_iso_db (date)", "visit_date_iso_mis (mis_data.vdate)"],
            "priority": "slot -> db -> mis_data",
            "date_mismatch_rows": int((out["date_mismatch"] == "1").sum())
            if "date_mismatch" in out.columns
            else 0,
            "visit_date_filled": _col_nonempty("visit_date"),
        },
        "mkb_validation": {
            "mkb_code_main_filled": _col_nonempty("mkb_code_main"),
            "mkb_code_mis_filled": _col_nonempty("mkb_code_mis"),
            "agreement_counts": {str(k): int(v) for k, v in mkb_agree_counts.items()},
            "note": "match=точное совпадение inline(слот22) и mis_data.diagnos; partial=по 3-знач. рубрике",
        },
        "parse_validation": {
            "parse_ok_rows": parse_ok_n,
            "note": "parse_ok=1 если слотов > max индекса схемы (строка не обрезана)",
        },
        "patient_age_filled": age_n,
        "doctor_fio_filled": int(out["doctor_fio"].fillna("").astype(str).str.strip().ne("").sum())
        if "doctor_fio" in out.columns
        else 0,
        "mis_data_join_fields": [
            "doctor_fio",
            "doctor_specialization",
            "pay_type",
            "filial",
            "mis_vdate",
            "mis_vtime",
            "patient_bdate",
            "mis_diagnos",
            "service_codes",
            "service_names",
            "service_row_count",
        ],
        "doctor_fio_source": "mis_data via visit_id (services aggregated with ' | ')",
        "doctor_specialization_source": "author by protocol doctor_id (fallback: mis_data MIN per visit)",
        "doc_type_distribution": {str(k): int(v) for k, v in (type_counts or {}).items()},
        "kz_kind_counts": {str(k): int(v) for k, v in (kz_kind_counts or {}).items()},
        "kz_scored_rows": int(n_scored),
        "kz_kind_rule": (
            "kz/certificate оцениваются; diagnostic (УЗИ/рентген/функц./эндоскопия/лаб.), "
            "non_clinical, empty - исключаются. См. classify_kz_kind."
        ),
        "parquet": str(parquet_path.resolve().relative_to(ROOT)),
        "csv": str(csv_path.resolve().relative_to(ROOT)),
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
