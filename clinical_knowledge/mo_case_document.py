"""Печатный документ МО: текст записи + оценка для сверки методистом.

Источник истины для текста - защищённый дневной срез (raw/quarantine parquet или
secure_cases CSV). Оценка и замечания - из витрины / case detail. PDF строится
через Chrome headless, если доступен; иначе отдаём print-ready HTML
(браузер: Печать → PDF).
"""
from __future__ import annotations

import csv
import html
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import re

ROOT = Path(__file__).resolve().parents[1]

DOCUMENT_KIND_LABELS = {
    "medical_exam": "Медицинский осмотр",
    "consultation": "Консультативное заключение",
    "certificate": "Справка",
    "diagnostic": "Диагностическое исследование",
    "non_clinical": "Неклинический документ",
    "empty": "Пустой документ",
    "unknown": "Не определён",
}

CLINICAL_FIELDS = (
    ("complaints", "Жалобы"),
    ("anamnesis_doctor", "Анамнез"),
    ("anamnesis_auto", "Анамнез (авто)"),
    ("objective_status", "Объективный статус"),
    ("exam_data", "Данные обследований"),
    ("manipulations", "Манипуляции"),
    ("clinical_diagnosis", "Диагноз клинический"),
    ("mis_diagnos", "Диагноз МИС"),
    ("exam_recommendations", "Рекомендации по обследованию"),
    ("treatment_recommendations", "Рекомендации по лечению"),
)

SCORED_KINDS = frozenset({"medical_exam", "consultation"})
_HEX_ID_RX = re.compile(r"^[a-f0-9]{32,64}$", re.IGNORECASE)
_ICD_CODE_RX = re.compile(r"^[A-Za-zА-Яа-я]\d{2}(?:\.\d{1,2})?$")


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _norm_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _looks_like_hash(value: Any) -> bool:
    return bool(_HEX_ID_RX.match(str(value or "").strip()))


def _safe_icd(value: Any) -> str:
    text = str(value or "").strip()
    if not text or _looks_like_hash(text):
        return ""
    return text if _ICD_CODE_RX.match(text) else ""


def _pick_icd(*values: Any) -> str:
    for value in values:
        code = _safe_icd(value)
        if code:
            return code
    return ""


def _medical_exam_roots() -> list[Path]:
    roots: list[Path] = []
    configured = (os.environ.get("MO_DATA_ROOT") or "").strip()
    if configured:
        roots.append(Path(configured))
    roots.append(ROOT / "data" / "medical_exams")
    var = Path("/var/data/medical_exams")
    if var.is_dir():
        roots.append(var)
    return roots


def _chrome_bin() -> str | None:
    for cand in (
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ):
        if cand and Path(cand).is_file():
            return cand
    return None


def score_reason(*, document_kind: str, overall_pct: Any, status: str = "") -> str:
    kind = str(document_kind or "unknown")
    if kind not in SCORED_KINDS:
        return f"Не оценивается: {DOCUMENT_KIND_LABELS.get(kind, kind)}"
    if overall_pct is None or overall_pct == "":
        if status == "scoring_error":
            return "Ошибка оценки - требуется повторный прогон"
        return "Оценка ещё не рассчитана"
    return "Оценено"


def _source_row_richness(row: Mapping[str, Any]) -> tuple[int, int]:
    """Scored documents first, then the row with the richest clinical content."""
    scored = int(str(row.get("document_kind") or "") in SCORED_KINDS)
    populated = sum(
        1
        for key, _label in CLINICAL_FIELDS
        if str(row.get(key) or "").strip().lower() not in {"", "nan", "none", "null"}
    )
    return scored, populated


def _read_source_records(path: Path) -> list[dict[str, Any]]:
    """Прочитать защищённый дневной срез без обязательной зависимости от pandas.

    Production-образ хранит безопасный fallback в CSV и не ставит pandas.
    Parquet остаётся опциональным богатым источником, когда pandas/pyarrow доступны.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            return []
        return pd.read_parquet(path).to_dict(orient="records")
    return []


def load_case_source_row(
    case_id: str,
    *,
    visit_date: str | None = None,
    mis_id: str | None = None,
) -> dict[str, Any] | None:
    """Найти клиническую строку: точная запись МИС, затем лучший документ визита."""
    needle = _norm_id(case_id)
    if not needle:
        return None
    day_hint = (visit_date or "")[:10]
    candidates: list[Path] = []
    for root in _medical_exam_roots():
        raw = root / "raw"
        secure = root / "secure_cases"
        if day_hint and len(day_hint) >= 10:
            year, month = day_hint[:4], day_hint[5:7]
            day_path = raw / year / month / f"mo_{day_hint}.parquet"
            if day_path.is_file():
                candidates.append(day_path)
            q_root = root / "quarantine" / year / month
            if q_root.is_dir():
                candidates.extend(sorted(q_root.glob(f"*/mo_{day_hint}.parquet"), reverse=True))
                candidates.extend(sorted(q_root.glob(f"{day_hint}*/mo_{day_hint}.parquet"), reverse=True))
            secure_path = secure / year / month / f"mo_{day_hint}.csv"
            if secure_path.is_file():
                candidates.append(secure_path)
        if raw.is_dir():
            candidates.extend(sorted(raw.rglob("mo_*.parquet"), reverse=True)[:40])
        if secure.is_dir():
            candidates.extend(sorted(secure.rglob("mo_*.csv"), reverse=True)[:40])
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            records = _read_source_records(path)
        except Exception:  # noqa: BLE001
            continue
        if not records:
            continue
        exact_needle = str(mis_id or "").strip()
        if exact_needle:
            for key in ("id", "mis_id"):
                if not any(key in row for row in records):
                    continue
                matched = [row for row in records if _norm_id(row.get(key)) == _norm_id(exact_needle)]
                if matched:
                    row = dict(matched[0])
                    row["_source_file"] = str(path)
                    if path.suffix.lower() == ".parquet":
                        row["_source_parquet"] = str(path)
                    return row
        visit_matches = []
        for key in ("visit_id", "id", "mis_id"):
            if not any(key in row for row in records):
                continue
            matched = [row for row in records if _norm_id(row.get(key)) == needle]
            if not matched:
                continue
            visit_matches.extend(matched)
            if key == "visit_id":
                break
        if visit_matches:
            row = max(visit_matches, key=_source_row_richness)
            row["_source_file"] = str(path)
            if path.suffix.lower() == ".parquet":
                row["_source_parquet"] = str(path)
            return row
    return None


def _warehouse_case_meta(case_id: str) -> dict[str, Any]:
    from clinical_knowledge.mo_backend import _read_connection
    from contextlib import closing

    needle = str(case_id or "").strip()
    with closing(_read_connection()) as conn:
        row = conn.execute(
            """SELECT c.*, d.doctor_fio,
                      COALESCE(d.specialty, c.specialty) AS doctor_specialty,
                      COALESCE(d.filial, c.filial) AS doctor_filial
               FROM fact_mo_case c
               LEFT JOIN dim_doctor d ON d.doctor_key = c.doctor_key
               WHERE c.visit_id = ? OR c.mis_id = ?
               ORDER BY CASE
                        WHEN c.document_kind='medical_exam' THEN 0
                        WHEN c.document_kind='consultation' THEN 1
                        ELSE 2
                      END
               LIMIT 1""",
            (needle, needle),
        ).fetchone()
        if not row:
            return {}
        item = dict(row)
        findings = conn.execute(
            """SELECT f.finding_code, f.severity, f.source_ref, f.evidence,
                      COALESCE(df.title_ru, f.finding_code) AS title_ru,
                      COALESCE(df.why_important_ru, '') AS why_important
               FROM fact_mo_finding f
               LEFT JOIN dim_finding df ON df.finding_code = f.finding_code
               WHERE f.mis_id = ?
               ORDER BY CASE f.severity WHEN 'P0' THEN 0 WHEN 'P1' THEN 1
                         WHEN 'P2' THEN 2 ELSE 3 END""",
            (item["mis_id"],),
        ).fetchall()
        item["findings"] = [dict(f) for f in findings]
        return item


def build_case_document_payload(
    case_id: str,
    *,
    month: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from clinical_knowledge.mo_backend import build_case_detail

    meta = _warehouse_case_meta(case_id)
    visit_date = str(meta.get("visit_date") or "")[:10]
    detail = dict(detail) if detail is not None else build_case_detail(
        case_id, month=month or (visit_date[:7] if visit_date else None)
    )
    record = dict(detail.get("record") or {}) if detail.get("ok") else {}
    preferred_id = str(meta.get("mis_id") or record.get("mis_id") or case_id)
    source = load_case_source_row(preferred_id, visit_date=visit_date or None)
    if not source:
        for probe in (case_id, meta.get("visit_id"), meta.get("mis_id")):
            if not probe:
                continue
            source = load_case_source_row(str(probe), visit_date=visit_date or None)
            if source:
                break
    source_kind = str((source or {}).get("document_kind") or "").strip()
    if source_kind == "kz":
        source_kind = "consultation"
    document_kind = str(
        source_kind
        or meta.get("document_kind")
        or record.get("document_kind")
        or "unknown"
    )
    overall = meta.get("overall_pct")
    if overall is None:
        overall = detail.get("deep_overall_pct")
    if overall is None:
        overall = record.get("overall_pct")
    findings = meta.get("findings") or detail.get("findings") or []
    clinical: dict[str, str] = {}
    src = source or {}
    for key, _label in CLINICAL_FIELDS:
        value = src.get(key)
        if value is None or str(value).strip() in {"", "nan", "None"}:
            value = record.get(key)
        text = str(value or "").strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            clinical[key] = text
    doctor = (
        meta.get("doctor_fio")
        or record.get("doctor_fio")
        or src.get("doctor_fio")
        or "Врач не указан"
    )
    diagnosis_code = _pick_icd(
        meta.get("diagnosis_code"),
        record.get("mkb_code_main"),
        record.get("diagnosis_code"),
        src.get("mkb_code_main"),
        src.get("diagnosis_code"),
    )
    payload = {
        "ok": True,
        "case_id": str(case_id),
        "mis_id": str(meta.get("mis_id") or record.get("mis_id") or src.get("id") or ""),
        "visit_id": str(meta.get("visit_id") or record.get("visit_id") or src.get("visit_id") or case_id),
        "visit_date": visit_date or str(record.get("date") or src.get("date") or "")[:10],
        "doctor_fio": doctor,
        "specialty": meta.get("doctor_specialty") or meta.get("specialty") or record.get("specialization") or "",
        "filial": meta.get("doctor_filial") or meta.get("filial") or record.get("filial") or "",
        "document_kind": document_kind,
        "document_kind_label": DOCUMENT_KIND_LABELS.get(document_kind, document_kind),
        "diagnosis_code": diagnosis_code,
        "overall_pct": overall,
        "score_reason": score_reason(
            document_kind=document_kind,
            overall_pct=overall,
            status=str(meta.get("status") or ""),
        ),
        "axes": detail.get("axes") or {},
        "findings": findings,
        "clinical": clinical,
        "has_source_text": bool(clinical),
        "source_state": "ready" if clinical else "missing",
        "source_format": (
            "secure_csv"
            if str((source or {}).get("_source_file") or "").lower().endswith(".csv")
            else ("parquet" if source else None)
        ),
        "source_parquet": (source or {}).get("_source_parquet"),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "detail_ok": bool(detail.get("ok")),
    }
    if not meta and not source and not detail.get("ok"):
        return {"ok": False, "error": "case_not_found", "case_id": case_id}
    return payload


def render_case_document_html(payload: Mapping[str, Any]) -> str:
    clinical_html = "".join(
        f"<section><h3>{_esc(label)}</h3><p>{_esc(payload.get('clinical', {}).get(key) or '—')}</p></section>"
        for key, label in CLINICAL_FIELDS
        if (payload.get("clinical") or {}).get(key)
    ) or "<p class='muted'>Клинический текст записи в локальном срезе не найден. Откройте исходник в МИС по визиту.</p>"
    findings = payload.get("findings") or []
    findings_html = "".join(
        "<li class='sev-{sev}'><b>{sev} · {title}</b><div>{why}</div>"
        "<div class='muted'>{src}</div></li>".format(
            sev=_esc(item.get("severity") or item.get("code") or "P?"),
            title=_esc(item.get("title_ru") or item.get("finding_code") or item.get("code") or "Замечание"),
            why=_esc(item.get("why_important") or item.get("evidence") or item.get("detail") or ""),
            src=_esc(item.get("source_ref") or ""),
        )
        for item in findings
    ) or "<li class='muted'>Критических замечаний нет.</li>"
    axes = payload.get("axes") or {}
    axis_html = "".join(
        f"<tr><td>{_esc(label)}</td><td>{_esc(axes.get(key) if axes.get(key) is not None else '—')}</td></tr>"
        for key, label in (
            ("documentation", "Оформление"),
            ("clinical_concordance", "Согласованность"),
            ("safety", "Безопасность"),
            ("regulatory", "Регуляторика"),
        )
    )
    score = payload.get("overall_pct")
    score_text = f"{round(float(score))}%" if isinstance(score, (int, float)) else "нет оценки"
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>МО { _esc(payload.get('visit_date')) } · визит { _esc(payload.get('visit_id')) }</title>
<style>
:root{{--ink:#1f2a37;--muted:#66788a;--line:#d9e2ec;--bg:#f5f7fa;--card:#fff;--accent:#0f766e;--rose:#b42318}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 "Avenir Next",Avenir,Segoe UI,sans-serif}}
main{{max-width:920px;margin:0 auto;padding:28px}}header{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:18px}}
h1{{margin:0;font-size:26px}}h2{{margin:22px 0 10px;font-size:18px}}h3{{margin:0 0 6px;font-size:14px;color:var(--accent)}}
.muted{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;margin-top:12px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.kpi span{{display:block;color:var(--muted);font-size:12px}}.kpi b{{font-size:20px}}
.banner{{background:#fff7ed;border:1px solid #fdba74;border-radius:12px;padding:12px 14px;margin:12px 0}}
.critical{{background:#fff1f0;border-color:#fda29b}}ul{{padding-left:18px}}li{{margin:8px 0}}
.sev-P0{{color:var(--rose)}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid var(--line);padding:8px;text-align:left}}
.actions{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}
.button{{appearance:none;border:0;border-radius:999px;padding:10px 16px;background:var(--accent);color:#fff;font-weight:600;cursor:pointer;text-decoration:none}}
.button.secondary{{background:#e7eef5;color:var(--ink)}}
@media print{{.actions{{display:none}}body{{background:#fff}}main{{padding:0}}.card{{box-shadow:none}}}}
</style></head><body><main>
<header>
  <div>
    <h1>{_esc(payload.get('document_kind_label') or 'Медицинская запись')}</h1>
    <p class="muted">{_esc(payload.get('visit_date'))} · визит {_esc(payload.get('visit_id'))} · запись {_esc(payload.get('mis_id'))}</p>
  </div>
  <div class="kpi"><span>Оценка</span><b>{_esc(score_text)}</b></div>
</header>
<div class="actions">
  <button class="button" type="button" onclick="window.print()">Печать / PDF</button>
  <a class="button secondary" href="/doctor/review?source=mo&amp;case={_esc(payload.get('visit_id'))}">Открыть в разборе</a>
</div>
{"<div class='banner'>" + _esc(payload.get('score_reason')) + "</div>" if payload.get("overall_pct") is None else ""}
{"<div class='banner critical'>Есть замечания P0/P1 - сверьте текст записи с оценкой ниже.</div>" if any(str(f.get('severity')) in {'P0','P1'} for f in findings) else ""}
<div class="card kpis">
  <div class="kpi"><span>Врач</span><b>{_esc(payload.get('doctor_fio'))}</b></div>
  <div class="kpi"><span>Специальность</span><b>{_esc(payload.get('specialty') or '—')}</b></div>
  <div class="kpi"><span>Филиал</span><b>{_esc(payload.get('filial') or '—')}</b></div>
  <div class="kpi"><span>МКБ</span><b>{_esc(payload.get('diagnosis_code') or '—')}</b></div>
</div>
<div class="card"><h2>Текст МО</h2>{clinical_html}</div>
<div class="card"><h2>Оси оценки</h2><table>{axis_html or "<tr><td colspan='2' class='muted'>Оси недоступны</td></tr>"}</table></div>
<div class="card"><h2>Замечания</h2><ul>{findings_html}</ul></div>
<footer class="muted" style="margin-top:18px">Сформировано {_esc(payload.get('generated_at'))}. Случай {_esc(payload.get('case_id'))}. Для архива используйте «Печать / PDF».</footer>
</main></body></html>"""


def html_to_pdf_bytes(html_text: str) -> bytes | None:
    chrome = _chrome_bin()
    if not chrome:
        return None
    with tempfile.TemporaryDirectory(prefix="mo-case-pdf-") as tmp:
        html_path = Path(tmp) / "case.html"
        pdf_path = Path(tmp) / "case.pdf"
        html_path.write_text(html_text, encoding="utf-8")
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not pdf_path.is_file():
            return None
        return pdf_path.read_bytes()


def build_case_document_response(case_id: str, *, month: str | None = None, as_pdf: bool = False) -> dict[str, Any]:
    payload = build_case_document_payload(case_id, month=month)
    if not payload.get("ok"):
        return payload
    html_text = render_case_document_html(payload)
    result = {
        "ok": True,
        "payload": payload,
        "html": html_text,
        "pdf_bytes": None,
        "pdf_available": False,
    }
    if as_pdf:
        pdf = html_to_pdf_bytes(html_text)
        result["pdf_bytes"] = pdf
        result["pdf_available"] = pdf is not None
    return result
