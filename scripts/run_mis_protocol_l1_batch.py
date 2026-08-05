#!/usr/bin/env python3
"""Массовый L1-анализ mis_protocol (без LLM, ~$0 API).

Запуск на Render (рекомендуется, тёплый сервис + данные на /var/data):

  export PORT=10000   # уже есть в Web Shell
  PYTHONPATH=. python3 scripts/run_mis_protocol_l1_batch.py \\
    --csv /var/data/mis_protocol/mis_protocol_2026-07.csv \\
    --out-dir /var/data/mis_protocol \\
    --month 2026-07 --resume --workers 1

Локально (нужен доступ к API и CSV):

  PYTHONPATH=. python3 scripts/run_mis_protocol_l1_batch.py \\
    --csv data/mis_protocol/mis_protocol_2026-07.csv \\
    --base https://protocol-bimy.onrender.com \\
    --out-dir data/mis_protocol --month 2026-07 --limit 50

Артефакты:
  {out}/kz_l1_{month}_cases.jsonl   - построчные результаты (ПДн, не в git)
  {out}/kz_l1_{month}_summary.json  - агрегаты по врачам (можно в git)
  {out}/kz_l1_{month}_state.jsonl   - resume-состояние

См. docs/plans/2026-07-21-mis-kz-l1-batch-v1.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "clinical_knowledge").is_dir():
    # Скрипт могли скопировать в /tmp на Render
    ROOT = Path(os.environ.get("PROTOCOL_ROOT") or "/opt/render/project/src")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from clinical_knowledge.mis_protocol_parse import KZ_SCORED_KINDS, classify_kz_kind
except ImportError:  # pragma: no cover - fallback if copied alone on Render
    KZ_SCORED_KINDS = frozenset({"kz", "certificate"})

    def classify_kz_kind(row):  # type: ignore[misc]
        return ("kz", "")

try:
    from clinical_knowledge.reg55_criteria import evaluate_reg55, regulation_meta
except ImportError:  # pragma: no cover - fallback if copied alone on Render
    def evaluate_reg55(case):  # type: ignore[misc]
        return {"regulatory_compliance_pct": None, "failed": [], "critical_failed": [],
                "has_p0_defect": False, "passed": 0, "total": 0, "na": 0, "by_group": {}}

    def regulation_meta():  # type: ignore[misc]
        return {"regulation_id": "mz_2021_55", "regulation_title": "", "criteria_total": 0}

try:
    from clinical_knowledge.mis_pay_type import pay_type_label_ru, normalize_pay_type_code
except ImportError:  # pragma: no cover - fallback if copied alone on Render
    def normalize_pay_type_code(raw):  # type: ignore[misc]
        if raw is None:
            return ""
        s = str(raw).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return ""
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
        except (TypeError, ValueError):
            pass
        return s

    def pay_type_label_ru(raw):  # type: ignore[misc]
        code = normalize_pay_type_code(raw)
        return {
            "0": "Не указан",
            "2": "Наличный расчёт",
            "3": "Страхование (ДМС)",
            "12": "Справки и профосмотры",
        }.get(code, f"Код {code}" if code else "Не указан")

FIELD_BLOCKS = [
    ("Жалобы", "complaints"),
    ("Анамнез", "anamnesis_doctor"),
    ("Анамнез (авто)", "anamnesis_auto"),
    ("Объективный статус", "objective_status"),
    ("Данные обследования", "exam_data"),
    ("Манипуляции", "manipulations"),
    ("Диагноз", "clinical_diagnosis"),
    ("Диагнозы (список)", "diagnosis_list"),
    ("Рекомендации по обследованию", "exam_recommendations"),
    ("Рекомендации по лечению", "treatment_recommendations"),
    ("Диспансерное наблюдение", "dispensary_info"),
    ("Явка", "return_date"),
]


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_kz_text(row: dict) -> str:
    parts: list[str] = []
    # Дата консультации: в PDF КЗ формат ДД.ММ.ГГГГ («Дата и время проведения…»).
    # Колонка date из БД часто ISO (2026-07-13) - парсер раньше её не видел и ложно
    # ставил «Не распознана дата консультации». Предпочитаем visit_date_text (слот ::1),
    # иначе нормализуем ISO → ДД.ММ.ГГГГ.
    from clinical_knowledge.date_parser import format_date_dmy, parse_date

    # visit_date - каноническая ISO-дата из экспортёра (сверена по 3 источникам).
    date_raw = (
        row.get("visit_date") or row.get("visit_date_text") or row.get("date") or ""
    ).strip()
    if date_raw:
        parsed = parse_date(date_raw[:32])
        date_label = format_date_dmy(parsed) if parsed else date_raw[:19]
        parts.append(f"Дата консультации: {date_label}")
    fio = (row.get("doctor_fio") or "").strip()
    spec = (row.get("doctor_specialization") or "").strip()
    if fio or spec:
        parts.append(f"Врач: {fio}" + (f", {spec}" if spec else ""))
    filial = (row.get("filial") or "").strip()
    if filial:
        parts.append(f"Филиал: {filial}")
    for title, name in FIELD_BLOCKS:
        val = (row.get(name) or "").strip()
        if val and val.lower() not in ("on", "off", "0", "1"):
            parts.append(f"{title}:\n{val}")
    vitals = []
    for k, label in (
        ("temperature", "t°"),
        ("bp_1", "АД1"),
        ("bp_2", "АД2"),
        ("heart_rate", "ЧСС"),
        ("resp_rate", "ЧД"),
        ("bmi", "ИМТ"),
        ("weight", "вес"),
        ("height", "рост"),
    ):
        v = (row.get(k) or "").strip()
        if v and v.lower() not in ("on", "off"):
            vitals.append(f"{label} {v}")
    if vitals:
        parts.append("Витальные показатели: " + ", ".join(vitals))
    return "\n\n".join(parts).strip()


def _post_tier(base: str, text: str, consultation_id: str, *, timeout: int = 180) -> dict:
    body = json.dumps(
        {
            "tier": "L1",
            "text": text,
            "consultation_id": consultation_id,
            "methodist_mode": False,
            "category_slugs": "",
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/api/consult-review/tier",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _direct_tier(text: str, consultation_id: str) -> dict:
    """L1 без HTTP - обходит rate-limit Render (60 POST/мин на /api/*)."""
    from clinical_knowledge.consult_tiering import run_consult_by_tier

    return run_consult_by_tier(
        tier="L1",
        text=text,
        bundle=None,
        consultation_id=consultation_id,
        category_slugs="",
    )


def _proto_names(items) -> list[str]:
    out: list[str] = []
    for it in items or []:
        if isinstance(it, dict):
            p = it.get("path") or it.get("protocol_path") or it.get("title") or it.get("protocol_id") or ""
        else:
            p = str(it)
        if p:
            name = Path(str(p)).name if ("/" in str(p) or "\\" in str(p)) else str(p)
            out.append(name[:90])
    return out


def summarize_case(row: dict, result: dict, ms: int, text_len: int) -> dict:
    sa = result.get("structured_analysis") or {}
    comp = sa.get("compliance") if isinstance(sa, dict) else {}
    if not isinstance(comp, dict):
        comp = {}
    overall = result.get("overall_score")
    if overall is None:
        overall = comp.get("overall_score")
    status = result.get("overall_status") or comp.get("overall_status")
    matched = _proto_names(comp.get("matched_protocols")) or _proto_names(result.get("retrieval_paths"))
    alignment = comp.get("alignment_by_block") or {}
    block_scores: dict[str, float | None] = {}
    if isinstance(alignment, dict):
        for bid, val in alignment.items():
            if isinstance(val, dict):
                sc = val.get("score")
                if sc is None:
                    sc = val.get("alignment_score")
                block_scores[str(bid)] = sc
            elif isinstance(val, (int, float)):
                block_scores[str(bid)] = float(val)
    case = {
        "ts": _utc(),
        "mis_id": row.get("id"),
        "visit_id": row.get("visit_id"),
        "patient_id": str(row.get("patient_id") or "").strip(),
        "date": (row.get("date") or "")[:19],
        "doctor_fio": (row.get("doctor_fio") or "").strip() or " - ",
        "doctor_specialization": (row.get("doctor_specialization") or "").strip() or " - ",
        "filial": (row.get("filial") or "").strip() or " - ",
        "diagnosis_short": ((row.get("clinical_diagnosis") or "").strip() or (row.get("diagnosis_list") or "").strip())[:160],
        "text_len": text_len,
        "analysis_ms": ms,
        "overall_pct": overall,
        "status": status,
        "matched_protocols": matched[:3],
        "block_scores": block_scores,
        "llm_used": bool(result.get("llm_used")),
        "error": None,
    }
    n_kz = row.get("_n_kz_per_visit")
    try:
        n_kz_i = int(n_kz) if n_kz is not None else None
    except (TypeError, ValueError):
        n_kz_i = None
    enrich_case_from_csv_row(case, row, n_kz=n_kz_i)
    return case


def _aggregate_deep(cases: list[dict]) -> dict:
    """Агрегаты глубокой оценки по всем КЗ (для summary/дашборда)."""
    from collections import Counter

    deep = [c["deep"] for c in cases if isinstance(c.get("deep"), dict)]
    n = len(deep)
    if not n:
        return {"n": 0}
    axis_sums = {a: 0.0 for a in ("documentation", "clinical_concordance", "safety", "regulatory")}
    axis_cnt = {a: 0 for a in axis_sums}
    sev_total = Counter()
    status_dist = Counter()
    finding_codes = Counter()
    harm = 0
    for d in deep:
        for a in axis_sums:
            v = (d.get("axes") or {}).get(a)
            if isinstance(v, (int, float)):
                axis_sums[a] += v
                axis_cnt[a] += 1
        for s, cnt in (d.get("n_by_severity") or {}).items():
            sev_total[s] += int(cnt or 0)
        status_dist[d.get("status") or "na"] += 1
        if d.get("has_potential_harm"):
            harm += 1
        for f in d.get("findings") or []:
            if not f.get("passed"):
                finding_codes[f.get("code")] += 1
    return {
        "n": n,
        "axis_means": {a: (round(axis_sums[a] / axis_cnt[a], 1) if axis_cnt[a] else None) for a in axis_sums},
        "severity_totals": dict(sev_total),
        "status_distribution": dict(status_dist),
        "n_potential_harm": harm,
        "pct_potential_harm": round(100.0 * harm / n, 1),
        "top_findings": finding_codes.most_common(20),
    }


def load_done_ids(state_path: Path) -> set[str]:
    done: set[str] = set()
    if not state_path.is_file():
        return done
    for line in state_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = str(row.get("visit_id") or "")
        if vid and row.get("status") == "ok":
            done.add(vid)
    return done


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


BLOCK_LABEL_RU = {
    "diagnosis": "диагноз",
    "complaints": "жалобы",
    "anamnesis": "анамнез",
    "objective_status": "объективный статус",
    "exams": "обследования",
    "treatment": "лечение",
    "follow_up": "наблюдение / явка",
    "limitations": "ограничения",
}

STATUS_LABEL_RU = {
    "compliant": "соответствует",
    "mostly_compliant": "в основном соответствует",
    "partially_compliant": "частичное соответствие",
    "non_compliant": "не соответствует",
    "not_assessed": "не оценено",
    "insufficient_protocol_data": "мало данных протокола",
    "manual_review_required": "нужен ручной разбор",
}

# Блоки, которые часто пусты в MIS или слабо матчятся с КП - мягче в комментариях
# и исключаются из core_overall (рейтинг «ядра» записи).
_SYSTEMICALLY_WEAK_BLOCKS = frozenset({"exams", "treatment", "limitations"})

_EMPTY_FIELD_TOKENS = frozenset({"", "on", "off", "0", "1", "nan", "none", "null"})


def _field_nonempty(row: dict, *names: str) -> bool:
    for name in names:
        v = str(row.get(name) or "").strip().lower()
        if v and v not in _EMPTY_FIELD_TOKENS:
            return True
    return False


def clinical_text_len(row: dict) -> int:
    """Длина клинических полей - для выбора «богатого» КЗ при нескольких на визит."""
    return len(build_kz_text(row))


def fields_present_from_row(row: dict) -> dict[str, bool]:
    return {
        "complaints": _field_nonempty(row, "complaints"),
        "anamnesis": _field_nonempty(row, "anamnesis_doctor", "anamnesis_auto"),
        "objective_status": _field_nonempty(row, "objective_status"),
        "exams": _field_nonempty(row, "exam_recommendations", "exam_data"),
        "treatment": _field_nonempty(row, "treatment_recommendations"),
        "diagnosis": _field_nonempty(row, "clinical_diagnosis", "diagnosis_list"),
        "follow_up": _field_nonempty(row, "dispensary_info", "return_date"),
    }


def core_overall_from_blocks(block_scores: dict | None) -> float | None:
    """Среднее по блокам без exams/treatment/limitations (эвристика MIS L1)."""
    if not isinstance(block_scores, dict):
        return None
    vals: list[float] = []
    for bid, bv in block_scores.items():
        if str(bid) in _SYSTEMICALLY_WEAK_BLOCKS:
            continue
        if isinstance(bv, (int, float)):
            vals.append(float(bv))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def split_service_names(raw: str | None, *, limit: int = 8) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.replace(";", "|").split("|")]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if not p or p.lower() in _EMPTY_FIELD_TOKENS:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p[:120])
        if len(out) >= limit:
            break
    return out


def enrich_case_from_csv_row(case: dict, row: dict | None, *, n_kz: int | None = None) -> None:
    """Дописать pay/services/fill из CSV к кейсу (для агрегатов summary)."""
    if not row:
        if n_kz is not None:
            case["n_kz_per_visit"] = int(n_kz)
        if case.get("core_overall_pct") is None:
            case["core_overall_pct"] = core_overall_from_blocks(case.get("block_scores"))
        return
    pt = normalize_pay_type_code(row.get("pay_type"))
    case["pay_type"] = pt
    case["pay_type_label"] = pay_type_label_ru(pt)
    services = split_service_names(row.get("service_names"))
    case["service_names"] = services
    case["service_primary"] = services[0] if services else ""
    case["fields_present"] = fields_present_from_row(row)
    if n_kz is not None:
        case["n_kz_per_visit"] = int(n_kz)
    elif case.get("n_kz_per_visit") is None:
        case["n_kz_per_visit"] = 1
    case["core_overall_pct"] = core_overall_from_blocks(case.get("block_scores"))


def select_rows_for_l1(rows: list[dict]) -> list[dict]:
    """Один ряд на visit_id: самое богатое КЗ; проставляем n_kz_per_visit.

    Правило: при нескольких mis_protocol на одном визите берём строку с
    максимальной длиной клинического текста (build_kz_text), при равенстве -
    больший mis id (новее).
    """
    by_vid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        vid = str(row.get("visit_id") or "").strip()
        if not vid:
            continue
        by_vid[vid].append(row)
    selected: list[dict] = []
    for vid, group in by_vid.items():
        best = max(
            group,
            key=lambda r: (
                clinical_text_len(r),
                int(str(r.get("id") or "0").strip() or "0")
                if str(r.get("id") or "").strip().isdigit()
                else 0,
            ),
        )
        best = dict(best)
        best["_n_kz_per_visit"] = len(group)
        selected.append(best)
    selected.sort(key=lambda r: str(r.get("date") or ""))
    return selected


def kz_kind_of(row: dict) -> str:
    """kz_kind строки: из готового столбца CSV или классификатором на месте."""
    kind = str(row.get("kz_kind") or "").strip()
    return kind or classify_kz_kind(row)[0]


def split_kz_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Разделить строки на оцениваемые медицинские документы и исключённые.

    Для нового МО-контура приоритетен ``mo_score_eligible``. В старых выгрузках
    сохраняется совместимое правило ``kz_kind``.
    """
    scored: list[dict] = []
    by_kind: Counter = Counter()
    by_spec: dict[str, Counter] = defaultdict(Counter)
    n_corrupt = 0
    for row in rows:
        kind = kz_kind_of(row)
        document_kind = str(row.get("document_kind") or "").strip()
        if not document_kind or document_kind.lower() == "nan":
            try:
                from clinical_knowledge.mo_daily import classify_document_kind

                document_kind, _reason = classify_document_kind(row)
                row["document_kind"] = document_kind
                row["mo_score_eligible"] = document_kind in {"medical_exam", "consultation"}
            except Exception:  # noqa: BLE001
                document_kind = ""
        by_kind[kind] += 1
        # Битая ::-строка (обрезана - слотов меньше схемы) в оценку не идёт: parse_ok=='0'.
        # Это не «плохой КЗ», а некорректно выгруженная строка. При отсутствии столбца
        # parse_ok (старые CSV) считаем строку валидной.
        if str(row.get("parse_ok", "1")).strip() == "0":
            n_corrupt += 1
            continue
        mo_eligible_raw = str(row.get("mo_score_eligible") or "").strip().lower()
        if mo_eligible_raw:
            mo_eligible = mo_eligible_raw in {"1", "true", "yes", "on"}
            if mo_eligible:
                scored.append(row)
                continue
            excluded_kind = document_kind or kind
            spec = (row.get("doctor_specialization") or "").strip() or " - "
            by_spec[excluded_kind][spec] += 1
        elif kind in KZ_SCORED_KINDS:
            scored.append(row)
        else:
            spec = (row.get("doctor_specialization") or "").strip() or " - "
            by_spec[kind][spec] += 1
    breakdown = {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_excluded": len(rows) - len(scored),
        "n_corrupt_parse": n_corrupt,
        "by_kind": dict(by_kind),
        "excluded_top_specialties": {
            kind: dict(sorted(spec.items(), key=lambda x: -x[1])[:8])
            for kind, spec in by_spec.items()
        },
        "rule_ru": (
            "В МО-контуре применяется явный признак mo_score_eligible; для старых "
            "выгрузок совместимо используются kz/certificate. Диагностические, "
            "неклинические, пустые и битые ::-строки исключаются. Неполный клинический "
            "документ не считается мусором и получает соответствующую низкую оценку."
        ),
    }
    return scored, breakdown


def dedupe_cases_by_visit(cases: list[dict]) -> list[dict]:
    """Оставляем лучший успешный кейс на visit_id (длиннее текст / новее ts)."""
    by_vid: dict[str, dict] = {}
    for c in cases:
        vid = str(c.get("visit_id") or "")
        if not vid:
            continue
        prev = by_vid.get(vid)
        if prev is None:
            by_vid[vid] = c
            continue
        prev_ok = not prev.get("error") and prev.get("overall_pct") is not None
        cur_ok = not c.get("error") and c.get("overall_pct") is not None
        if cur_ok and not prev_ok:
            by_vid[vid] = c
            continue
        if prev_ok and not cur_ok:
            continue
        prev_len = int(prev.get("text_len") or 0)
        cur_len = int(c.get("text_len") or 0)
        if cur_len > prev_len:
            by_vid[vid] = c
        elif cur_len == prev_len and (c.get("ts") or "") >= (prev.get("ts") or ""):
            by_vid[vid] = c
    return list(by_vid.values())


def comment_for_visit(case: dict) -> str:
    """Краткий комментарий методисту: что просело в L1 по этому визиту."""
    parts: list[str] = []
    status = str(case.get("status") or "").strip()
    if status and status not in {"compliant", "mostly_compliant"}:
        parts.append(STATUS_LABEL_RU.get(status, status))

    present = case.get("fields_present") or {}
    blocks = case.get("block_scores") or {}
    scored: list[tuple[str, float]] = []
    for bid, bv in blocks.items():
        if not isinstance(bv, (int, float)):
            continue
        scored.append((str(bid), float(bv)))
    scored.sort(key=lambda x: x[1])

    weak: list[str] = []
    for bid, val in scored:
        field_ok = present.get(bid)
        if bid in _SYSTEMICALLY_WEAK_BLOCKS:
            if field_ok is False:
                if val <= 5 and len(weak) < 3:
                    weak.append(f"«{BLOCK_LABEL_RU.get(bid, bid)}» не заполнено в КЗ")
                continue
            if val > 5:
                continue
            weak.append(f"«{BLOCK_LABEL_RU.get(bid, bid)}» почти пустой ({val:.0f}%)")
        elif val < 55:
            weak.append(f"«{BLOCK_LABEL_RU.get(bid, bid)}» слабо ({val:.0f}%)")
        if len(weak) >= 3:
            break
    if not weak and scored:
        bid, val = scored[0]
        weak.append(f"самый слабый блок - {BLOCK_LABEL_RU.get(bid, bid)} ({val:.0f}%)")
    parts.extend(weak)

    text_len = case.get("text_len")
    try:
        tl = int(text_len) if text_len is not None else None
    except (TypeError, ValueError):
        tl = None
    if tl is not None and tl < 350:
        parts.append(f"очень короткий текст КЗ ({tl} симв.)")

    n_kz = case.get("n_kz_per_visit")
    try:
        if n_kz is not None and int(n_kz) > 1:
            parts.append(f"на визите {int(n_kz)} КЗ (взят самый полный)")
    except (TypeError, ValueError):
        pass

    if not parts:
        overall = case.get("overall_pct")
        parts.append(f"низкий overall ({overall}%) без явных провалов по блокам")
    return "; ".join(parts)


def build_worst_visits(
    cases: list[dict],
    *,
    doctor_avgs: dict[str, float],
    limit: int = 50,
) -> list[dict]:
    """Топ худших визитов по L1 overall среди всех врачей."""
    rows: list[dict] = []
    for c in cases:
        if c.get("error") or c.get("overall_pct") is None:
            continue
        fio = (c.get("doctor_fio") or "").strip() or " - "
        try:
            overall = float(c["overall_pct"])
        except (TypeError, ValueError):
            continue
        comment = comment_for_visit(c)
        r55 = evaluate_reg55(c)
        rows.append(
            {
                "visit_id": str(c.get("visit_id") or ""),
                "patient_id": str(c.get("patient_id") or "").strip(),
                "date": (c.get("date") or "")[:19],
                "doctor_fio": fio,
                "doctor_specialization": (c.get("doctor_specialization") or "").strip() or " - ",
                "filial": (c.get("filial") or "").strip() or " - ",
                "overall_pct": round(overall, 1),
                "l1_overall_pct": round(overall, 1),
                "doctor_avg_overall_pct": doctor_avgs.get(fio),
                "status": c.get("status"),
                "l1_status": c.get("status"),
                "diagnosis_short": (c.get("diagnosis_short") or "")[:160],
                "comment": comment,
                "l1_comment": comment,
                "block_scores": c.get("block_scores") or {},
                "core_overall_pct": c.get("core_overall_pct"),
                "pay_type": c.get("pay_type") or "",
                "pay_type_label": c.get("pay_type_label") or pay_type_label_ru(c.get("pay_type")),
                "n_kz_per_visit": c.get("n_kz_per_visit") or 1,
                "service_primary": (c.get("service_primary") or "")[:120],
                "l2_overall_pct": c.get("l2_overall_pct"),
                "l2_status": c.get("l2_status"),
                "l2_comment": c.get("l2_comment"),
                "l2_error": c.get("l2_error"),
                "reg55_pct": r55.get("regulatory_compliance_pct"),
                "reg55_failed": r55.get("failed") or [],
                "reg55_has_p0": bool(r55.get("has_p0_defect")),
            }
        )
    rows.sort(
        key=lambda r: (
            r.get("overall_pct") if r.get("overall_pct") is not None else 999,
            r.get("date") or "",
            r.get("visit_id") or "",
        )
    )
    return rows[: max(0, int(limit))]


def load_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_csv_by_visit(csv_path: Path) -> dict[str, dict]:
    """visit_id → лучшая (самая полная) строка КЗ + _n_kz_per_visit.

    Только строки-КЗ (kz/certificate): диагностика/пустые не должны подтягивать
    patient_id и не идут в L2.
    """
    raw = load_csv_rows(csv_path)
    if not raw:
        return {}
    scored, _ = split_kz_rows(raw)
    selected = select_rows_for_l1(scored)
    return {str(r.get("visit_id") or "").strip(): r for r in selected if r.get("visit_id")}


def attach_patient_ids(cases: list[dict], csv_by_visit: dict[str, dict]) -> None:
    for c in cases:
        row = csv_by_visit.get(str(c.get("visit_id") or ""))
        if row and not str(c.get("patient_id") or "").strip():
            c["patient_id"] = str(row.get("patient_id") or "").strip()
        n_kz = None
        if row is not None:
            try:
                n_kz = int(row.get("_n_kz_per_visit") or 1)
            except (TypeError, ValueError):
                n_kz = 1
        enrich_case_from_csv_row(c, row, n_kz=n_kz)


def extract_l2_fields(result: dict) -> dict:
    sa = result.get("structured_analysis") or {}
    comp = sa.get("compliance") if isinstance(sa, dict) else {}
    if not isinstance(comp, dict):
        comp = {}
    overall = result.get("overall_score")
    if overall is None:
        overall = comp.get("overall_score")
    status = result.get("overall_status") or comp.get("overall_status")
    parts: list[str] = []
    if result.get("render_l2_limited"):
        parts.append("L2 limited (как L1 на Render)")
    summary = result.get("summary_ru") or result.get("review_summary_ru")
    review = result.get("review")
    if not summary and isinstance(review, dict):
        summary = review.get("summary_ru")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip()[:280])
    issues = comp.get("critical_issues") or comp.get("issues") or []
    for item in issues[:3]:
        if isinstance(item, dict):
            txt = str(item.get("message_ru") or item.get("text") or item.get("issue") or "").strip()
        else:
            txt = str(item).strip()
        if txt:
            parts.append(txt[:160])
    alignment = comp.get("alignment_by_block") or {}
    weak: list[str] = []
    if isinstance(alignment, dict):
        scored = []
        for bid, val in alignment.items():
            if isinstance(val, dict):
                sc = val.get("score")
                if sc is None:
                    sc = val.get("alignment_score")
            elif isinstance(val, (int, float)):
                sc = float(val)
            else:
                continue
            if isinstance(sc, (int, float)):
                scored.append((str(bid), float(sc)))
        scored.sort(key=lambda x: x[1])
        for bid, sc in scored[:3]:
            if sc < 55:
                weak.append(f"{BLOCK_LABEL_RU.get(bid, bid)} {sc:.0f}%")
    if weak:
        parts.append("слабые блоки: " + ", ".join(weak))
    if not parts and overall is not None:
        parts.append(f"L2 overall {overall}%")
    try:
        overall_f = round(float(overall), 1) if overall is not None else None
    except (TypeError, ValueError):
        overall_f = None
    return {
        "l2_overall_pct": overall_f,
        "l2_status": status,
        "l2_comment": "; ".join(parts)[:500] if parts else None,
        "l2_error": None,
        "l2_mode": result.get("l2_mode") or result.get("review_tier") or "L2",
    }


def enrich_worst_visits_l2(
    worst: list[dict],
    *,
    csv_by_visit: dict[str, dict],
    sleep_s: float = 0.0,
) -> list[dict]:
    """Прогон L2 (fast pipeline) для списка worst_visits; мутирует и возвращает список."""
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # На Render L2-fast даёт сверку с протоколом; skip_llm обходится через fast=1.
    os.environ.setdefault("CONSULT_L2_FAST", "1")
    os.environ.setdefault("CONSULT_RENDER_L2_SKIP_LLM", "0")

    from rag_server import _consult_review_from_tier_or_pipeline

    for i, item in enumerate(worst, 1):
        vid = str(item.get("visit_id") or "")
        row = csv_by_visit.get(vid)
        if not row:
            item["l2_error"] = "visit_not_in_csv"
            item["l2_comment"] = "нет строки CSV для L2"
            print(f"l2 {i}/{len(worst)} visit={vid} SKIP no csv", flush=True)
            continue
        if not str(item.get("patient_id") or "").strip():
            item["patient_id"] = str(row.get("patient_id") or "").strip()
        text = build_kz_text(row)
        t0 = time.perf_counter()
        try:
            result = _consult_review_from_tier_or_pipeline(
                tier="L2",
                text=text,
                bundle=None,
                consultation_id=f"mis-l2-{vid}",
                category_slugs="",
                require_rag_for_l2=False,
                l2_narrative=False,
            )
            fields = extract_l2_fields(result)
            item.update(fields)
            ms = int((time.perf_counter() - t0) * 1000)
            print(
                f"l2 {i}/{len(worst)} visit={vid} l2={item.get('l2_overall_pct')} "
                f"l1={item.get('l1_overall_pct')} ms={ms}",
                flush=True,
            )
        except Exception as e:
            item["l2_error"] = str(e)[:300]
            item["l2_comment"] = f"ошибка L2: {e}"[:300]
            item["l2_overall_pct"] = None
            item["l2_status"] = None
            print(f"l2 {i}/{len(worst)} visit={vid} FAIL {e}", flush=True)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return worst


def build_summary(
    cases: list[dict],
    *,
    month: str,
    source: str,
    excluded_breakdown: dict | None = None,
) -> dict:
    cases = dedupe_cases_by_visit(cases)
    by_doctor: dict[str, list] = defaultdict(list)
    by_spec: dict[str, list] = defaultdict(list)
    by_filial: dict[str, list] = defaultdict(list)
    by_pay: dict[str, list] = defaultdict(list)
    by_service: dict[str, list] = defaultdict(list)
    status_c: Counter = Counter()
    hist = Counter({"0-49": 0, "50-59": 0, "60-69": 0, "70-79": 0, "80-89": 0, "90-100": 0})
    block_sums: dict[str, list[float]] = defaultdict(list)
    block_sums_when_present: dict[str, list[float]] = defaultdict(list)
    field_fill_n: Counter = Counter()
    field_fill_ok: Counter = Counter()
    errors = 0
    scores: list[float] = []
    core_scores: list[float] = []
    reg55_scores: list[float] = []
    reg55_failed_c: Counter = Counter()
    reg55_titles: dict[str, dict] = {}
    reg55_p0_n = 0
    multi_kz_n = 0
    multi_kz_extra = 0

    for c in cases:
        if c.get("error"):
            errors += 1
            continue
        fio = c.get("doctor_fio") or " - "
        spec = c.get("doctor_specialization") or " - "
        filial = c.get("filial") or " - "
        by_doctor[fio].append(c)
        by_spec[spec].append(c)
        by_filial[filial].append(c)
        pay_label = (c.get("pay_type_label") or pay_type_label_ru(c.get("pay_type")) or "Не указан")
        by_pay[pay_label].append(c)
        for svc in (c.get("service_names") or ([c["service_primary"]] if c.get("service_primary") else [])):
            if svc:
                by_service[str(svc)[:120]].append(c)
        try:
            n_kz = int(c.get("n_kz_per_visit") or 1)
        except (TypeError, ValueError):
            n_kz = 1
        if n_kz > 1:
            multi_kz_n += 1
            multi_kz_extra += n_kz - 1
        present = c.get("fields_present") or {}
        for bid, ok in present.items():
            field_fill_n[str(bid)] += 1
            if ok:
                field_fill_ok[str(bid)] += 1
        r55 = evaluate_reg55(c)
        r55_pct = r55.get("regulatory_compliance_pct")
        if isinstance(r55_pct, (int, float)):
            reg55_scores.append(float(r55_pct))
        if r55.get("has_p0_defect"):
            reg55_p0_n += 1
        for f in r55.get("failed") or []:
            fid = str(f.get("id") or "")
            if fid:
                reg55_failed_c[fid] += 1
                reg55_titles.setdefault(fid, f)
        st = str(c.get("status") or "unknown")
        status_c[st] += 1
        sc = c.get("overall_pct")
        if sc is None:
            continue
        try:
            v = float(sc)
        except (TypeError, ValueError):
            continue
        scores.append(v)
        core = c.get("core_overall_pct")
        if core is None:
            core = core_overall_from_blocks(c.get("block_scores"))
            c["core_overall_pct"] = core
        if isinstance(core, (int, float)):
            core_scores.append(float(core))
        if v < 50:
            hist["0-49"] += 1
        elif v < 60:
            hist["50-59"] += 1
        elif v < 70:
            hist["60-69"] += 1
        elif v < 80:
            hist["70-79"] += 1
        elif v < 90:
            hist["80-89"] += 1
        else:
            hist["90-100"] += 1
        for bid, bv in (c.get("block_scores") or {}).items():
            if isinstance(bv, (int, float)):
                block_sums[str(bid)].append(float(bv))
                if present.get(str(bid), True):
                    block_sums_when_present[str(bid)].append(float(bv))

    def _agg_group(items: list[dict], *, key_name: str, key_val: str) -> dict:
        ok = [c for c in items if c.get("overall_pct") is not None]
        vals = [float(c["overall_pct"]) for c in ok]
        cores = [
            float(c["core_overall_pct"])
            for c in ok
            if isinstance(c.get("core_overall_pct"), (int, float))
        ]
        return {
            key_name: key_val,
            "n": len(items),
            "avg_overall_pct": round(sum(vals) / len(vals), 1) if vals else None,
            "avg_core_overall_pct": round(sum(cores) / len(cores), 1) if cores else None,
            "min_overall_pct": round(min(vals), 1) if vals else None,
            "max_overall_pct": round(max(vals), 1) if vals else None,
            "mostly_compliant_n": sum(1 for c in items if str(c.get("status") or "").startswith("mostly")),
            "partial_n": sum(1 for c in items if "partial" in str(c.get("status") or "")),
        }

    doctors = []
    for fio, items in by_doctor.items():
        row = _agg_group(items, key_name="doctor_fio", key_val=fio)
        row["specialization"] = (items[0].get("doctor_specialization") if items else None) or " - "
        row["filial"] = (items[0].get("filial") if items else None) or " - "
        doctors.append(row)
    doctors.sort(key=lambda r: (-(r["avg_overall_pct"] or 0), -r["n"], r["doctor_fio"]))

    specialties = [
        _agg_group(items, key_name="specialization", key_val=k)
        for k, items in by_spec.items()
    ]
    specialties.sort(key=lambda r: (-(r["avg_overall_pct"] or 0), -r["n"]))

    filials = [
        _agg_group(items, key_name="filial", key_val=k)
        for k, items in by_filial.items()
    ]
    filials.sort(key=lambda r: (-(r["avg_overall_pct"] or 0), -r["n"]))

    pay_types = [
        _agg_group(items, key_name="pay_type_label", key_val=k)
        for k, items in by_pay.items()
    ]
    for row in pay_types:
        # reverse-lookup code from first case
        sample = next((c for c in by_pay[row["pay_type_label"]] if c.get("pay_type") is not None), None)
        row["pay_type"] = (sample or {}).get("pay_type") or ""
    pay_types.sort(key=lambda r: (-(r["n"] or 0), -(r["avg_overall_pct"] or 0)))

    services = [
        _agg_group(items, key_name="service_name", key_val=k)
        for k, items in by_service.items()
    ]
    services.sort(key=lambda r: (-(r["n"] or 0), -(r["avg_overall_pct"] or 0)))
    top_services = services[:25]

    block_avg = {
        k: round(sum(v) / len(v), 1) for k, v in sorted(block_sums.items()) if v
    }
    block_avg_when_filled = {
        k: round(sum(v) / len(v), 1)
        for k, v in sorted(block_sums_when_present.items())
        if v
    }
    field_fill_rate = {
        k: round(100.0 * field_fill_ok[k] / field_fill_n[k], 1)
        for k in sorted(field_fill_n)
        if field_fill_n[k]
    }

    scored_doctors = [
        d for d in doctors if d.get("avg_overall_pct") is not None and int(d.get("n") or 0) >= 3
    ]
    scored_doctors_asc = sorted(
        scored_doctors,
        key=lambda r: (r["avg_overall_pct"], -r["n"], r["doctor_fio"]),
    )
    bottom_doctors = scored_doctors_asc[:15]
    doctor_avgs = {
        str(d["doctor_fio"]): float(d["avg_overall_pct"])
        for d in scored_doctors
        if d.get("avg_overall_pct") is not None
    }
    worst_visits = build_worst_visits(
        cases,
        doctor_avgs=doctor_avgs,
        limit=50,
    )

    # Очередь LLM: worst 50 + bottom doctors sample (до 80 уникальных visit_id)
    queue_vids: list[str] = []
    seen_q: set[str] = set()
    for w in worst_visits:
        vid = str(w.get("visit_id") or "")
        if vid and vid not in seen_q:
            seen_q.add(vid)
            queue_vids.append(vid)
    bottom_names = {str(d["doctor_fio"]) for d in bottom_doctors}
    bottom_cases = sorted(
        [
            c
            for c in cases
            if (c.get("doctor_fio") or "") in bottom_names
            and c.get("overall_pct") is not None
            and not c.get("error")
        ],
        key=lambda c: float(c["overall_pct"]),
    )
    for c in bottom_cases:
        vid = str(c.get("visit_id") or "")
        if not vid or vid in seen_q:
            continue
        seen_q.add(vid)
        queue_vids.append(vid)
        if len(queue_vids) >= 80:
            break
    llm_review_queue = {
        "n": len(queue_vids),
        "visit_ids": queue_vids,
        "rule_ru": (
            "До 80 визитов: сначала топ-50 worst overall, затем худшие КЗ врачей "
            "из bottom_doctors (n>=3). Для UI / пакетного LLM."
        ),
    }

    return {
        "month": month,
        "tier": "L1",
        "llm_used": False,
        "generated_at": _utc(),
        "source_csv": source,
        "n_cases": len(cases),
        "n_ok": len(cases) - errors,
        "n_errors": errors,
        "n_multi_kz_visits": multi_kz_n,
        "n_multi_kz_extra_rows": multi_kz_extra,
        "avg_overall_pct": round(sum(scores) / len(scores), 1) if scores else None,
        "median_overall_pct": round(sorted(scores)[len(scores) // 2], 1) if scores else None,
        "avg_core_overall_pct": round(sum(core_scores) / len(core_scores), 1) if core_scores else None,
        "score_histogram": dict(hist),
        "status_counts": dict(status_c),
        "block_avg": block_avg,
        "block_avg_when_filled": block_avg_when_filled,
        "field_fill_rate": field_fill_rate,
        "avg_regulatory_compliance_pct": (
            round(sum(reg55_scores) / len(reg55_scores), 1) if reg55_scores else None
        ),
        "reg55_p0_defect_n": reg55_p0_n,
        "reg55_scored_n": len(reg55_scores),
        "reg55_top_failed": [
            {
                "id": fid,
                "title": (reg55_titles.get(fid) or {}).get("title"),
                "point": (reg55_titles.get(fid) or {}).get("point"),
                "severity": (reg55_titles.get(fid) or {}).get("severity"),
                "n": cnt,
                "pct": round(100.0 * cnt / len(cases), 1) if cases else None,
            }
            for fid, cnt in reg55_failed_c.most_common()
        ],
        "reg55_meta": regulation_meta(),
        "doctors": doctors,
        "specialties": specialties,
        "filials": filials,
        "pay_types": pay_types,
        "top_services": top_services,
        "top_doctors": [d for d in doctors if d.get("avg_overall_pct") is not None][:15],
        "bottom_doctors": bottom_doctors,
        "worst_visits": worst_visits,
        "worst_visits_meta": {
            "limit": 50,
            "scope": "all_doctors",
            "min_doctor_n": 0,
            "rule_ru": (
                "50 визитов с самым низким L1 overall среди всех врачей "
                "(не только bottom_doctors); рядом колонки L2 после --enrich-l2-worst. "
                "При нескольких КЗ на визит выбран самый полный текст."
            ),
        },
        "llm_review_queue": llm_review_queue,
        "excluded_breakdown": excluded_breakdown or {},
        "gemini_reviews": [],
        "gemini_meta": {
            "model_preferred": "gemini-2.5-pro",
            "note_ru": "Выборочный LLM-разбор качества КЗ из UI.",
        },
        "notes": [
            "L1 = structured без RAG/LLM; стоимость API ~$0.",
            "*_print поля MIS = флаги on/off; в текст брались клинические столбцы.",
            "Полный jsonl с кейсами хранится только на /var/data (ПДн).",
            "worst_visits: топ-50 слабых визитов overall + patient_id; L2/LLM - отдельные поля.",
            "excluded_breakdown: диагностика (УЗИ и пр.)/пустые/неклинич. вне оценки (classify_kz_kind).",
            "core_overall = среднее блоков без exams/treatment/limitations (эвристика MIS).",
            "exams/treatment часто проседают из-за пустых полей или жёсткого матча с КП.",
            "pay_type: 0 не указан, 2 наличный, 3 ДМС, 12 справки/профосмотры.",
        ],
    }



def load_cases_from_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--base", default="", help="API base; default http://127.0.0.1:$PORT")
    ap.add_argument(
        "--direct",
        action="store_true",
        help="Вызывать L1 in-process (без HTTP) - рекомендуется на Render, нет 429",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--rebuild-summary-only", action="store_true")
    ap.add_argument(
        "--enrich-l2-worst",
        action="store_true",
        help="После сборки summary прогнать worst_visits через L2-fast и перезаписать summary",
    )
    ap.add_argument(
        "--reset-fails",
        action="store_true",
        help="Перед запуском убрать fail из state (повторить 429/ошибки)",
    )
    ap.add_argument(
        "--deep-eval",
        action="store_true",
        help="Глубокая оценка (kz_deep_eval): оси A/B/C/D, детекторы диагноза/лечения, findings, risk-gate",
    )
    ap.add_argument(
        "--deep-only",
        action="store_true",
        help="Только глубокая оценка без L1 (быстро, без тяжёлого стека) - для датасета дашборда/gold",
    )
    args = ap.parse_args()
    if args.deep_only:
        args.deep_eval = True

    base = (args.base or "").strip() or f"http://127.0.0.1:{os.environ.get('PORT', '10000')}"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / f"kz_l1_{args.month}_cases.jsonl"
    state_path = out_dir / f"kz_l1_{args.month}_state.jsonl"
    summary_path = out_dir / f"kz_l1_{args.month}_summary.json"
    gemini_path = out_dir / f"kz_l1_{args.month}_gemini_reviews.json"

    def _write_summary(summary: dict) -> None:
        if gemini_path.is_file():
            try:
                gem = json.loads(gemini_path.read_text(encoding="utf-8"))
                if isinstance(gem, dict) and isinstance(gem.get("reviews"), list):
                    summary["gemini_reviews"] = gem.get("reviews") or []
                    summary["gemini_meta"] = {
                        **(summary.get("gemini_meta") or {}),
                        **(gem.get("meta") or {}),
                    }
            except (OSError, json.JSONDecodeError):
                pass
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        queue = summary.get("llm_review_queue") or {}
        queue_path = out_dir / f"kz_l1_{args.month}_llm_queue.json"
        queue_path.write_text(
            json.dumps(
                {
                    "month": args.month,
                    "generated_at": summary.get("generated_at"),
                    **queue,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.rebuild_summary_only or args.enrich_l2_worst:
        cases = load_cases_from_jsonl(cases_path)
        _, kz_breakdown = split_kz_rows(load_csv_rows(args.csv))
        csv_by_visit = load_csv_by_visit(args.csv)
        # cases.jsonl мог быть собран старым кодом (все визиты, вкл. диагностику/пустые).
        # Оставляем только КЗ-визиты (те, что прошли split_kz_rows и есть в csv_by_visit).
        kz_vids = set(csv_by_visit.keys())
        if kz_vids:
            before = len(cases)
            cases = [c for c in cases if str(c.get("visit_id") or "") in kz_vids]
            print(f"kz filter cases: {before} -> {len(cases)} (по КЗ-визитам из CSV)", flush=True)
        attach_patient_ids(cases, csv_by_visit)
        # preserve prior L2 fields if rebuilding without re-enrich
        prior_l2: dict[str, dict] = {}
        if summary_path.is_file() and not args.enrich_l2_worst:
            try:
                prev = json.loads(summary_path.read_text(encoding="utf-8"))
                for w in prev.get("worst_visits") or []:
                    vid = str(w.get("visit_id") or "")
                    if vid and w.get("l2_overall_pct") is not None:
                        prior_l2[vid] = {
                            "l2_overall_pct": w.get("l2_overall_pct"),
                            "l2_status": w.get("l2_status"),
                            "l2_comment": w.get("l2_comment"),
                            "l2_error": w.get("l2_error"),
                            "l2_mode": w.get("l2_mode"),
                        }
            except (OSError, json.JSONDecodeError):
                prior_l2 = {}
        summary = build_summary(
            cases, month=args.month, source=str(args.csv), excluded_breakdown=kz_breakdown
        )
        if prior_l2:
            for w in summary.get("worst_visits") or []:
                extra = prior_l2.get(str(w.get("visit_id") or ""))
                if extra:
                    w.update(extra)
        if args.enrich_l2_worst:
            enrich_worst_visits_l2(
                summary.get("worst_visits") or [],
                csv_by_visit=csv_by_visit,
                sleep_s=float(args.sleep or 0),
            )
            summary["worst_visits_meta"] = {
                **(summary.get("worst_visits_meta") or {}),
                "l2_enriched_at": _utc(),
                "l2_enriched_n": sum(
                    1
                    for w in (summary.get("worst_visits") or [])
                    if w.get("l2_overall_pct") is not None
                ),
            }
        _write_summary(summary)
        print(
            f"rebuilt summary raw={len(cases)} unique={summary.get('n_cases')} "
            f"worst_visits={len(summary.get('worst_visits') or [])} "
            f"l2_ok={sum(1 for w in (summary.get('worst_visits') or []) if w.get('l2_overall_pct') is not None)} "
            f"-> {summary_path}"
        )
        return 0

    if args.reset_fails and state_path.is_file():
        kept = []
        for line in state_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok":
                kept.append(row)
        state_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
            encoding="utf-8",
        )
        print(f"reset fails: kept ok={len(kept)} in state", flush=True)

    if args.deep_only:
        os.chdir(ROOT)
        print("mode=deep-only (без L1)", flush=True)
    elif args.direct:
        os.chdir(ROOT)
        print(f"mode=direct ROOT={ROOT}", flush=True)
    else:
        try:
            with urllib.request.urlopen(f"{base.rstrip('/')}/health/live", timeout=15) as r:
                print(f"health={r.status} base={base}", flush=True)
        except Exception as e:
            print(f"ERROR: API unavailable at {base}: {e}", file=sys.stderr)
            return 2

    done = load_done_ids(state_path) if args.resume else set()
    raw_rows: list[dict] = []
    with args.csv.open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < args.offset:
                continue
            raw_rows.append(row)
    scored_rows, kz_breakdown = split_kz_rows(raw_rows)
    rows = select_rows_for_l1(scored_rows)
    if args.limit:
        rows = rows[: args.limit]
    print(
        f"csv_rows={len(raw_rows)} scored_rows={len(scored_rows)} "
        f"excluded={kz_breakdown.get('n_excluded')} by_kind={kz_breakdown.get('by_kind')} "
        f"unique_visits={len(rows)} "
        f"multi_kz={sum(1 for r in rows if int(r.get('_n_kz_per_visit') or 1) > 1)}",
        flush=True,
    )

    todo = []
    for row in rows:
        vid = str(row.get("visit_id") or row.get("id") or "")
        if args.resume and vid in done:
            continue
        todo.append(row)
    print(f"rows_in_slice={len(rows)} todo={len(todo)} already_done={len(done)} workers={args.workers}", flush=True)

    ok = fail = 0

    def _one(row: dict) -> dict:
        vid = str(row.get("visit_id") or row.get("id") or "")
        text = build_kz_text(row)
        t0 = time.perf_counter()
        try:
            if args.deep_only:
                case = summarize_case(row, {}, 0, len(text))
            else:
                if args.direct:
                    result = _direct_tier(text, f"mis-{vid}")
                else:
                    result = _post_tier(base, text, f"mis-{vid}")
                ms = int((time.perf_counter() - t0) * 1000)
                case = summarize_case(row, result, ms, len(text))
            if args.deep_eval:
                try:
                    from clinical_knowledge.kz_deep_eval import (
                        evaluate_kz_deep,
                        load_drug_ctx,
                        resolve_protocol_ctx,
                    )

                    deep_case = {**row, **case}
                    proto = resolve_protocol_ctx(deep_case)
                    deep = evaluate_kz_deep(deep_case, protocol_ctx=proto, drug_ctx=load_drug_ctx())
                    case["deep"] = {
                        "axes": deep["axes"],
                        "overall_pct": deep["overall_pct"],
                        "status": deep["overall_status"],
                        "n_findings": deep["n_findings"],
                        "n_by_severity": deep["n_by_severity"],
                        "has_potential_harm": deep["has_potential_harm"],
                        "protocol_used": deep["protocol_used"],
                        "findings": deep["findings"][:20],
                        # E3: shadow concordance для warehouse / очереди «Вчера»
                        "shadow_findings": (deep.get("shadow_findings") or [])[:20],
                    }
                    # Аддитивно: scorer v3 в shadow-режиме (не переключает prod score/gate).
                    try:
                        from clinical_knowledge.kz_evaluation_engine import (
                            evaluate_kz_v3,
                            resolve_mode,
                        )

                        mode = resolve_mode()
                        if mode.enabled:
                            v3 = evaluate_kz_v3(
                                deep_case, protocol_ctx=proto, drug_ctx=load_drug_ctx(),
                                legacy={
                                    "deep_overall_pct": deep.get("overall_pct"),
                                    "deep_status": deep.get("overall_status"),
                                    "l1_overall_pct": case.get("overall_pct"),
                                },
                                mode=mode,
                            )
                            case["evaluation_v3"] = v3.to_public_dict()
                    except Exception as e:  # noqa: BLE001
                        case["evaluation_v3_error"] = str(e)[:200]
                    # V4 is the single primary score. Deep remains the deterministic
                    # fallback and v3 is retained for the 30-day comparison window.
                    try:
                        from clinical_knowledge.kz_evaluation_v4 import evaluate_kz_v4

                        v4 = evaluate_kz_v4(
                            deep_case,
                            protocol_ctx=proto,
                            drug_ctx=load_drug_ctx(),
                            legacy={
                                "deep_overall_pct": deep.get("overall_pct"),
                                "deep_status": deep.get("overall_status"),
                                "l1_overall_pct": case.get("overall_pct"),
                                "v3_score_pct": (
                                    case.get("evaluation_v3") or {}
                                ).get("score_pct"),
                            },
                        )
                        case["evaluation_v4"] = v4.to_public_dict()
                        case["overall_pct_v3"] = deep.get("overall_pct")
                        case["scorer_version"] = v4.scorer_version
                        if v4.mode.primary:
                            case["overall_pct"] = v4.score_pct
                            case["status"] = v4.status
                    except Exception as e:  # noqa: BLE001
                        case["evaluation_v4_error"] = str(e)[:200]
                        case["scorer_version"] = "deep-v2-fallback"
                except Exception as e:  # noqa: BLE001
                    case["deep_error"] = str(e)[:200]
            return case
        except Exception as e:
            ms = int((time.perf_counter() - t0) * 1000)
            return {
                "ts": _utc(),
                "mis_id": row.get("id"),
                "visit_id": row.get("visit_id"),
                "doctor_fio": (row.get("doctor_fio") or "").strip() or " - ",
                "doctor_specialization": (row.get("doctor_specialization") or "").strip() or " - ",
                "filial": (row.get("filial") or "").strip() or " - ",
                "diagnosis_short": ((row.get("clinical_diagnosis") or "").strip())[:160],
                "analysis_ms": ms,
                "error": str(e)[:300],
            }

    workers = max(1, min(4, int(args.workers)))
    processed = 0
    if workers == 1:
        for row in todo:
            case = _one(row)
            append_jsonl(cases_path, case)
            vid = str(case.get("visit_id") or "")
            if case.get("error"):
                fail += 1
                append_jsonl(state_path, {"visit_id": vid, "status": "fail", "detail": case.get("error")})
            else:
                ok += 1
                append_jsonl(state_path, {"visit_id": vid, "status": "ok"})
            processed += 1
            if processed % 50 == 0 or processed == len(todo):
                print(f"progress {processed}/{len(todo)} ok={ok} fail={fail}", flush=True)
            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_one, row): row for row in todo}
            for fut in as_completed(futs):
                case = fut.result()
                append_jsonl(cases_path, case)
                vid = str(case.get("visit_id") or "")
                if case.get("error"):
                    fail += 1
                    append_jsonl(state_path, {"visit_id": vid, "status": "fail", "detail": case.get("error")})
                else:
                    ok += 1
                    append_jsonl(state_path, {"visit_id": vid, "status": "ok"})
                processed += 1
                if processed % 50 == 0 or processed == len(todo):
                    print(f"progress {processed}/{len(todo)} ok={ok} fail={fail}", flush=True)

    all_cases = load_cases_from_jsonl(cases_path)
    csv_by_visit = load_csv_by_visit(args.csv)
    attach_patient_ids(all_cases, csv_by_visit)
    summary = build_summary(
        all_cases, month=args.month, source=str(args.csv), excluded_breakdown=kz_breakdown
    )
    if args.deep_eval:
        summary["deep_eval"] = _aggregate_deep(all_cases)
    _write_summary(summary)
    print(
        f"DONE ok={ok} fail={fail} total_unique={summary.get('n_cases')} "
        f"avg={summary.get('avg_overall_pct')} core={summary.get('avg_core_overall_pct')} "
        f"pay_slices={len(summary.get('pay_types') or [])} "
        f"worst_visits={len(summary.get('worst_visits') or [])} -> {summary_path}",
        flush=True,
    )
    return 0 if fail == 0 or ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
