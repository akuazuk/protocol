"""Безопасные примитивы ежедневного МО-pipeline.

Модуль не подключается к БД и не управляет VPN при импорте. Все внешние действия
выполняет orchestrator через внедряемый subprocess runner.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import hmac
import html
import json
import math
import os
import sqlite3
import statistics
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

MINSK = ZoneInfo("Europe/Minsk")
DOCUMENT_KINDS = frozenset(
    {"medical_exam", "consultation", "certificate", "diagnostic", "non_clinical", "empty", "unknown"}
)
SCORED_DOCUMENT_KINDS = frozenset({"medical_exam", "consultation"})
EMPTY_TOKENS = frozenset({"", "0", "1", "on", "off", "nan", "none", "null"})
PII_FIELDS = frozenset(
    {
        "patient_id",
        "visit_id",
        "doctor_fio",
        "result_raw",
        "complaints",
        "anamnesis_doctor",
        "anamnesis_auto",
        "objective_status",
        "exam_data",
        "clinical_diagnosis",
        "diagnosis_list",
        "exam_recommendations",
        "treatment_recommendations",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def minsk_today(now: datetime | None = None) -> date:
    current = now or datetime.now(MINSK)
    if current.tzinfo is None:
        current = current.replace(tzinfo=MINSK)
    return current.astimezone(MINSK).date()


def resolve_run_date(value: str, *, now: datetime | None = None) -> date:
    value = value.strip().lower()
    if value == "yesterday":
        return minsk_today(now) - timedelta(days=1)
    parsed = date.fromisoformat(value)
    if parsed >= minsk_today(now):
        raise ValueError("Дата МО должна быть раньше сегодняшней даты Europe/Minsk")
    return parsed


def source_window(day: date) -> tuple[str, str]:
    return day.isoformat(), (day + timedelta(days=1)).isoformat()


def month_bounds(day: date) -> tuple[date, date]:
    start = day.replace(day=1)
    end = date(day.year + (day.month == 12), 1 if day.month == 12 else day.month + 1, 1)
    return start, end


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if type(value).__module__.startswith(("pandas", "numpy")):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, content: str) -> None:
    atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default) + "\n",
    )


@contextmanager
def exclusive_lock(path: Path, *, blocking: bool = False) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream.fileno(), flags)
        except BlockingIOError as exc:
            raise RuntimeError(f"МО-pipeline уже запущен: {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_document_kind(row: Mapping[str, Any], rules: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """Детерминированная taxonomy для продуктового контура МО."""
    rules = rules or {}
    text_fields = (
        "complaints",
        "anamnesis_doctor",
        "anamnesis_auto",
        "objective_status",
        "exam_data",
        "clinical_diagnosis",
        "diagnosis_list",
        "exam_recommendations",
        "treatment_recommendations",
    )
    has_content = any(str(row.get(key) or "").strip().lower() not in EMPTY_TOKENS for key in text_fields)
    if not has_content:
        return "empty", "нет клинического содержания"

    haystack = " ".join(
        str(row.get(key) or "").lower()
        for key in ("doctor_specialization", "service_codes", "service_names", "clinical_diagnosis")
    )
    custom = rules.get("keywords") if isinstance(rules.get("keywords"), Mapping) else {}
    keywords = {
        "diagnostic": ("узи", "ультразвук", "рентген", "эндоскоп", "лаборатор", "диагност"),
        "non_clinical": ("медсестр", "медицинская сестра", "регистратор", "логопед"),
        "certificate": ("справк", "выписк"),
        "medical_exam": ("профосмотр", "медосмотр", "предрейсов", "периодическ", "предварительн"),
    }
    for kind, defaults in keywords.items():
        configured = custom.get(kind, defaults)
        if any(str(token).lower() in haystack for token in configured):
            return kind, f"признак taxonomy: {kind}"
    pay_type = str(row.get("pay_type") or "").strip().removesuffix(".0")
    if pay_type in {str(v) for v in rules.get("medical_exam_pay_types", ["12"])}:
        return "medical_exam", f"тип оплаты {pay_type}"

    legacy = str(row.get("kz_kind") or "").strip()
    if legacy == "diagnostic":
        return "diagnostic", "совместимая классификация kz_kind"
    if legacy == "non_clinical":
        return "non_clinical", "совместимая классификация kz_kind"
    if legacy == "certificate":
        return "certificate", "справка без подтверждённого признака медосмотра"
    if legacy == "kz":
        return "consultation", "консультативная запись"
    return "unknown", "недостаточно признаков для taxonomy"


def add_document_taxonomy(frame: Any, rules: Mapping[str, Any] | None = None) -> Any:
    classified = frame.apply(lambda row: classify_document_kind(row.to_dict(), rules), axis=1)
    result = frame.copy()
    result["document_kind"] = [item[0] for item in classified]
    result["document_kind_reason"] = [item[1] for item in classified]
    result["mo_score_eligible"] = result["document_kind"].isin(SCORED_DOCUMENT_KINDS)
    return result


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    actual: float | int | None = None
    threshold: float | int | None = None


@dataclass
class QualityResult:
    rows: int
    blocking: list[QualityIssue] = field(default_factory=list)
    warnings: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "rows": self.rows,
            "blocking": [asdict(item) for item in self.blocking],
            "warnings": [asdict(item) for item in self.warnings],
            "metrics": self.metrics,
        }


def _filled_pct(frame: Any, column: str) -> float:
    if column not in frame.columns or len(frame) == 0:
        return 100.0 if len(frame) == 0 else 0.0
    values = frame[column].fillna("").astype(str).str.strip().str.lower()
    return round(100.0 * (~values.isin(EMPTY_TOKENS)).sum() / len(frame), 3)


def validate_export(
    frame: Any,
    *,
    day: date,
    source_rows: int | None = None,
    historical_same_weekday_counts: Sequence[int] = (),
) -> QualityResult:
    """Blocking/warning gates до scoring и публикации."""
    result = QualityResult(rows=int(len(frame)))
    metrics = result.metrics
    metrics["source_rows"] = source_rows
    metrics["raw_rows"] = len(frame)

    required = {"id", "visit_id", "visit_date", "parse_ok"}
    missing = sorted(required - set(frame.columns))
    if missing:
        result.blocking.append(QualityIssue("missing_columns", "blocking", ", ".join(missing)))
        return result
    if source_rows is not None and source_rows != len(frame):
        result.blocking.append(
            QualityIssue("row_parity", "blocking", "source row count не совпадает с raw", len(frame), source_rows)
        )

    duplicate_ids = int(frame["id"].duplicated(keep=False).sum())
    duplicate_pairs = int(frame.duplicated(subset=["id", "visit_id"], keep=False).sum())
    metrics.update({"duplicate_id_rows": duplicate_ids, "duplicate_pair_rows": duplicate_pairs})
    if duplicate_ids or duplicate_pairs:
        result.blocking.append(
            QualityIssue("duplicates", "blocking", "обнаружены дубли id или (id, visit_id)", max(duplicate_ids, duplicate_pairs), 0)
        )

    dates = frame["visit_date"].fillna("").astype(str).str[:10]
    outside = int((dates != day.isoformat()).sum())
    metrics["outside_window_rows"] = outside
    if outside:
        result.blocking.append(
            QualityIssue("date_window", "blocking", "даты за пределами календарного окна", outside, 0)
        )

    parse_pct = (
        100.0
        if len(frame) == 0 and source_rows == 0
        else round(100.0 * (frame["parse_ok"].astype(str) == "1").sum() / max(1, len(frame)), 3)
    )
    metrics["parse_ok_pct"] = parse_pct
    if parse_pct < 99.5:
        result.blocking.append(QualityIssue("parse_ok", "blocking", "parse_ok ниже порога", parse_pct, 99.5))
    date_filled = _filled_pct(frame, "visit_date")
    metrics["visit_date_filled_pct"] = date_filled
    if date_filled < 99.9:
        result.blocking.append(QualityIssue("visit_date_missing", "blocking", "не заполнена дата", date_filled, 99.9))

    mismatch_pct = 0.0
    if "date_mismatch" in frame:
        mismatch_pct = round(100.0 * (frame["date_mismatch"].astype(str) == "1").sum() / max(1, len(frame)), 3)
    metrics["date_mismatch_pct"] = mismatch_pct
    if mismatch_pct >= 0.2:
        result.warnings.append(QualityIssue("date_mismatch", "warning", "повышенное расхождение дат", mismatch_pct, 0.2))

    fill_columns = ("doctor_fio", "doctor_specialization", "filial", "patient_age_years", "mkb_code_main")
    for column in fill_columns:
        metrics[f"{column}_filled_pct"] = _filled_pct(frame, column)
    if metrics["doctor_fio_filled_pct"] < 98:
        result.warnings.append(
            QualityIssue("doctor_missing", "warning", "заполненность врача ниже порога", metrics["doctor_fio_filled_pct"], 98)
        )

    if historical_same_weekday_counts and len(frame) > 0:
        baseline = statistics.median(historical_same_weekday_counts)
        metrics["same_weekday_median_rows"] = baseline
        delta = abs(len(frame) - baseline) / baseline if baseline else 0.0
        metrics["volume_delta_pct"] = round(delta * 100, 2)
        if baseline and delta > 0.5:
            result.warnings.append(
                QualityIssue("volume_anomaly", "warning", "объём отличается от медианы более чем на 50%", metrics["volume_delta_pct"], 50)
            )
    return result


def _atomic_write_parquet(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(frame: Any, path: Path, *, excluded: Iterable[str] = ()) -> None:
    view = frame.drop(columns=[name for name in excluded if name in frame.columns], errors="ignore")
    atomic_write_text(path, view.to_csv(index=False))


def install_daily_partition(
    frame: Any,
    *,
    day: date,
    root: Path,
    quality: QualityResult,
    run_id: str,
    source_meta: Mapping[str, Any],
) -> tuple[Path, Path]:
    target_dir = root / "raw" / f"{day:%Y}" / f"{day:%m}"
    if quality.passed:
        partition = target_dir / f"mo_{day.isoformat()}.parquet"
        meta_path = target_dir / f"mo_{day.isoformat()}.meta.json"
    else:
        target_dir = root / "quarantine" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}" / run_id
        partition = target_dir / f"mo_{day.isoformat()}.parquet"
        meta_path = target_dir / f"mo_{day.isoformat()}.meta.json"
    _atomic_write_parquet(frame, partition)
    meta = {
        "run_id": run_id,
        "date": day.isoformat(),
        "window": source_window(day),
        "rows": len(frame),
        "sha256": sha256_file(partition),
        "quality": quality.to_dict(),
        "source": dict(source_meta),
        "installed_at": utc_now(),
    }
    atomic_write_json(meta_path, meta)
    return partition, meta_path


def merge_daily_partitions(
    daily_paths: Sequence[Path],
    *,
    month: str,
    out_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Пересобрать rolling month из daily partitions, upsert по id."""
    import pandas as pd

    frames = []
    for path in sorted(daily_paths):
        frame = pd.read_parquet(path)
        frame["_partition_source"] = path.name
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    before = len(merged)
    if before:
        if "id" not in merged:
            raise ValueError("daily partition не содержит id")
        merged["_content_hash"] = merged.apply(
            lambda row: hashlib.sha256(
                json.dumps(row.to_dict(), sort_keys=True, default=_json_default).encode("utf-8")
            ).hexdigest(),
            axis=1,
        )
        merged = merged.sort_values(["id", "_partition_source"]).drop_duplicates("id", keep="last")
        if merged["id"].duplicated().any():
            raise ValueError("upsert оставил дубли id")
        merged = merged.drop(columns=["_partition_source"])
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"mis_protocol_{month}.parquet"
    csv_path = out_dir / f"mis_protocol_{month}.csv"
    _atomic_write_parquet(merged, parquet_path)
    _atomic_write_csv(merged, csv_path, excluded=PII_FIELDS & {"result_raw"})
    details = {
        "month": month,
        "daily_partitions": len(daily_paths),
        "input_rows": before,
        "rows": len(merged),
        "upserted_duplicates": before - len(merged),
        "sha256": sha256_file(parquet_path),
    }
    atomic_write_json(out_dir / f"mis_protocol_{month}.merge.json", details)
    return parquet_path, csv_path, details


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _aggregate_group(rows: Sequence[Mapping[str, Any]], key: str, *, suppress_below: int) -> list[dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in rows:
        label = str(row.get(key) or "Не указано")
        counts[label] += 1
        score = _safe_number(row.get("overall_pct"))
        if score is not None:
            groups[label].append(score)
    output = []
    for label, count in counts.most_common():
        output.append(
            {
                key: label if count >= suppress_below else "Малая группа",
                "label": label if count >= suppress_below else "Малая группа",
                "n": count,
                "suppressed": count < suppress_below,
                "avg_score": round(statistics.fmean(groups[label]), 1)
                if count >= suppress_below and groups[label]
                else None,
                "needs_attention": sum(score < 70 for score in groups[label])
                if count >= suppress_below
                else None,
            }
        )
    return output

def build_daily_report(
    raw_rows: Sequence[Mapping[str, Any]],
    scored_cases: Sequence[Mapping[str, Any]],
    *,
    day: date,
    run_id: str,
    revision: int,
    quality: Mapping[str, Any],
    month_to_date: Mapping[str, Any] | None = None,
    comparisons: Mapping[str, Any] | None = None,
    suppress_below: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scores = [score for row in scored_cases if (score := _safe_number(row.get("overall_pct"))) is not None]
    kinds = Counter(str(row.get("document_kind") or "unknown") for row in raw_rows)
    failures = [row for row in scored_cases if row.get("error")]
    eligible_rows = [row for row in raw_rows if row.get("document_kind") in SCORED_DOCUMENT_KINDS]
    eligible_visits = {
        str(row.get("visit_id") or row.get("id") or "")
        for row in eligible_rows
        if row.get("visit_id") or row.get("id")
    }
    severe_case_ids: set[str] = set()
    action_queue = []
    for row in scored_cases:
        score = _safe_number(row.get("overall_pct"))
        deep = row.get("deep") if isinstance(row.get("deep"), Mapping) else {}
        severe = sum(int(v or 0) for key, v in (deep.get("n_by_severity") or {}).items() if key in {"P0", "P1"})
        if severe:
            severe_case_ids.add(str(row.get("mis_id") or row.get("visit_id") or id(row)))
        if severe or (score is not None and score < 60) or row.get("error"):
            action_queue.append(
                {
                    "mis_id": row.get("mis_id"),
                    "visit_id": row.get("visit_id"),
                    "priority": "P0/P1" if severe else "review",
                    "reason": "критические замечания" if severe else ("ошибка оценки" if row.get("error") else "низкая оценка"),
                    "score": score,
                    "doctor_fio": row.get("doctor_fio"),
                    "status": "new",
                    "assignee": None,
                    "due_date": (day + timedelta(days=1)).isoformat(),
                }
            )
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "date": day.isoformat(),
        "window": source_window(day),
        "revision": revision,
        "generated_at": utc_now(),
        "quality": dict(quality),
        "summary": {
            "source_rows": len(raw_rows),
            "eligible_rows": len(eligible_rows),
            "eligible_visits": len(eligible_visits),
            "excluded_rows": len(raw_rows) - len(eligible_rows),
            "document_kinds": dict(kinds),
            "scored": len(scored_cases) - len(failures),
            "scoring_errors": len(failures),
            "avg_score": round(statistics.fmean(scores), 1) if scores else None,
            "median_score": round(statistics.median(scores), 1) if scores else None,
            "needs_attention": sum(score < 70 for score in scores),
            "critical": sum(
                1
                for row in scored_cases
                if (
                    str(row.get("mis_id") or row.get("visit_id") or id(row)) in severe_case_ids
                    or (
                        (case_score := _safe_number(row.get("overall_pct"))) is not None
                        and case_score < 50
                    )
                )
            ),
            "critical_clinical": len(severe_case_ids),
            "critical_low_score": sum(score < 50 for score in scores),
        },
        "axes": _axes_summary(scored_cases),
        "organizations": {
            "specialties": _aggregate_group(scored_cases, "doctor_specialization", suppress_below=suppress_below),
            "filials": _aggregate_group(scored_cases, "filial", suppress_below=suppress_below),
            "doctors": _aggregate_group(scored_cases, "doctor_fio", suppress_below=suppress_below),
        },
        "comparisons": dict(comparisons or {}),
        "month_to_date": dict(month_to_date or {}),
        "action_queue": action_queue,
    }
    public = {
        "schema_version": report["schema_version"],
        "date": report["date"],
        "revision": report["revision"],
        "generated_at": report["generated_at"],
        "quality": report["quality"],
        "summary": report["summary"],
        "axes": report["axes"],
        "organizations": {
            key: [item for item in values if not item["suppressed"]]
            for key, values in report["organizations"].items()
            if key != "doctors"
        },
        "comparisons": report["comparisons"],
        "month_to_date": report["month_to_date"],
        "freshness": {"status": "current", "source_date": report["date"]},
    }
    assert not (PII_FIELDS & set(public))
    return report, public


def _axes_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    values: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        deep = case.get("deep")
        if not isinstance(deep, Mapping):
            continue
        for axis, raw in (deep.get("axes") or {}).items():
            score = _safe_number(raw)
            if score is not None:
                values[str(axis)].append(score)
    return {axis: round(statistics.fmean(scores), 1) if scores else None for axis, scores in values.items()}


def write_daily_report(
    secure: Mapping[str, Any],
    public: Mapping[str, Any],
    *,
    day: date,
    root: Path,
) -> dict[str, Path]:
    report_dir = root / "reports" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
    public_daily = root / "public" / "daily" / f"{day.isoformat()}.json"
    atomic_write_json(report_dir / "report.json", secure)
    atomic_write_json(report_dir / "quality.json", secure.get("quality") or {})
    atomic_write_json(public_daily, public)
    atomic_write_json(root / "public" / "latest.json", public)
    atomic_write_json(root / "public" / "monthly" / f"{day:%Y-%m}.json", public.get("month_to_date") or {})

    queue = secure.get("action_queue") or []
    if queue:
        columns = sorted({key for row in queue for key in row})
        buffer = tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="")
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        writer.writerows(queue)
        buffer.seek(0)
        atomic_write_text(report_dir / "report.csv", buffer.read())
        buffer.close()
    else:
        atomic_write_text(report_dir / "report.csv", "priority,reason,status\n")
    atomic_write_text(report_dir / "report.html", _render_daily_report_html(secure, day))
    return {"report_dir": report_dir, "public_daily": public_daily}


def _render_daily_report_html(report: Mapping[str, Any], day: date) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value if value is not None else "нет данных"))

    summary = report.get("summary") or {}
    axes = report.get("axes") or {}
    quality = report.get("quality") or {}
    quality_core = quality.get("metrics") or {}
    organizations = report.get("organizations") or {}
    queue = report.get("action_queue") or []

    cards = [
        ("Записей из МИС", summary.get("source_rows", 0), "lavender"),
        ("Допущено к оценке", summary.get("eligible_rows", 0), "mint"),
        ("Средняя оценка", summary.get("avg_score"), "sky"),
        ("Требуют внимания", summary.get("needs_attention", 0), "peach"),
        ("Критические", summary.get("critical", 0), "rose"),
        ("Ошибки оценки", summary.get("scoring_errors", 0), "sand"),
    ]
    card_html = "".join(
        f'<article class="kpi {tone}"><span>{esc(label)}</span><strong>{esc(value)}</strong></article>'
        for label, value, tone in cards
    )
    axis_labels = {
        "documentation": "Документирование",
        "clinical_concordance": "Клиническая согласованность",
        "safety": "Безопасность",
        "regulatory": "Обязательные поля",
    }
    axis_html = "".join(
        '<div class="bar-row"><span>{}</span><div class="track"><i style="width:{}%"></i></div><b>{}</b></div>'.format(
            esc(axis_labels.get(str(key), key)),
            max(0, min(100, float(value or 0))),
            esc(value),
        )
        for key, value in axes.items()
    ) or '<p class="muted">Оценки по осям пока не сформированы.</p>'
    kinds_html = "".join(
        f"<tr><td>{esc(kind)}</td><td>{esc(count)}</td></tr>"
        for kind, count in (summary.get("document_kinds") or {}).items()
    )

    def organization_rows(key: str) -> str:
        return "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                esc(row.get("label")),
                esc(row.get("n")),
                esc(row.get("avg_score")),
                esc(row.get("needs_attention")),
            )
            for row in (organizations.get(key) or [])[:12]
            if not row.get("suppressed")
        )

    queue_html = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            esc(item.get("priority")),
            esc(item.get("doctor_fio")),
            esc(item.get("score")),
            esc(item.get("reason")),
            esc(item.get("status")),
        )
        for item in queue[:100]
    ) or '<tr><td colspan="5" class="muted">Случаев для срочного разбора нет.</td></tr>'
    issues = list(quality.get("blocking") or []) + list(quality.get("warnings") or [])
    quality_html = "".join(
        f'<li class="{esc(item.get("severity"))}"><b>{esc(item.get("code"))}</b>: {esc(item.get("message"))}</li>'
        for item in issues
    ) or "<li>Блокирующих ошибок и предупреждений нет.</li>"
    generated = report.get("generated_at") or ""
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Отчёт МО за {esc(day.strftime('%d.%m.%Y'))}</title>
<style>
:root{{--ink:#283342;--muted:#6f7d8e;--line:#e4e9ef;--bg:#f6f8fb;--card:#fff;--accent:#7479c9}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}}
main{{max-width:1280px;margin:auto;padding:32px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;margin-bottom:24px}}
h1{{font-size:30px;margin:0}}h2{{font-size:19px;margin:0 0 16px}}p{{margin:4px 0}}.muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}}.kpi,.panel{{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 5px 18px #3440540a}}
.kpi{{padding:18px;min-height:105px}}.kpi span{{display:block;color:var(--muted)}}.kpi strong{{display:block;font-size:27px;margin-top:10px}}
.lavender{{background:#f1effb}}.mint{{background:#eaf7f1}}.sky{{background:#eaf4fa}}.peach{{background:#fff1e8}}.rose{{background:#fbecef}}.sand{{background:#f8f3e8}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}}.panel{{padding:22px;overflow:auto}}
.bar-row{{display:grid;grid-template-columns:190px 1fr 44px;gap:10px;align-items:center;margin:12px 0}}.track{{height:10px;background:#edf0f5;border-radius:10px;overflow:hidden}}.track i{{display:block;height:100%;background:#8b91d7;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}}th{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}}
ul{{padding-left:20px}}li{{margin:7px 0}}li.blocking{{color:#a93245}}li.warning{{color:#9a681a}}
.wide{{margin-top:16px}}footer{{margin-top:18px;color:var(--muted);font-size:12px}}
@media(max-width:980px){{.grid{{grid-template-columns:repeat(3,1fr)}}.two{{grid-template-columns:1fr}}}}
@media(max-width:600px){{main{{padding:18px}}header{{display:block}}.grid{{grid-template-columns:1fr 1fr}}.bar-row{{grid-template-columns:130px 1fr 38px}}}}
</style></head><body><main>
<header><div><p class="muted">Ежедневный контроль медицинских осмотров</p><h1>Отчёт МО за {esc(day.strftime('%d.%m.%Y'))}</h1></div>
<div class="muted">Ревизия {esc(report.get('revision'))}<br>Сформирован {esc(generated)}</div></header>
<section class="grid">{card_html}</section>
<section class="two"><article class="panel"><h2>Оценка по направлениям</h2>{axis_html}</article>
<article class="panel"><h2>Качество данных</h2><p>Статус: <b>{'пройден' if quality.get('passed') else 'требует проверки'}</b></p>
<p class="muted">Распознано: {esc(quality_core.get('parse_ok_pct'))}% · Врач: {esc(quality_core.get('doctor_fio_filled_pct'))}%</p><ul>{quality_html}</ul></article></section>
<section class="two"><article class="panel"><h2>Типы документов</h2><table><thead><tr><th>Тип</th><th>Количество</th></tr></thead><tbody>{kinds_html}</tbody></table></article>
<article class="panel"><h2>Специальности</h2><table><thead><tr><th>Специальность</th><th>N</th><th>Средняя</th><th>Внимание</th></tr></thead><tbody>{organization_rows('specialties')}</tbody></table></article></section>
<section class="panel wide"><h2>Филиалы</h2><table><thead><tr><th>Филиал</th><th>N</th><th>Средняя</th><th>Внимание</th></tr></thead><tbody>{organization_rows('filials')}</tbody></table></section>
<section class="panel wide"><h2>Очередь разбора</h2><table><thead><tr><th>Приоритет</th><th>Врач</th><th>Оценка</th><th>Причина</th><th>Статус</th></tr></thead><tbody>{queue_html}</tbody></table></section>
<footer>Отчёт содержит защищённые сведения и предназначен только для внутреннего контура.</footer>
</main></body></html>"""


def initialize_warehouse(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS fact_mo_case (
              mis_id TEXT PRIMARY KEY, visit_id TEXT, visit_date TEXT NOT NULL,
              document_kind TEXT NOT NULL, overall_pct REAL, status TEXT,
              doctor_key TEXT, specialty TEXT, filial TEXT, content_hash TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fact_mo_finding (
              mis_id TEXT NOT NULL, finding_code TEXT NOT NULL, severity TEXT,
              passed INTEGER, evidence TEXT, source_ref TEXT,
              PRIMARY KEY (mis_id, finding_code)
            );
            CREATE TABLE IF NOT EXISTS fact_mo_score_axis (
              mis_id TEXT NOT NULL, axis TEXT NOT NULL, score REAL,
              PRIMARY KEY (mis_id, axis)
            );
            CREATE TABLE IF NOT EXISTS fact_mo_daily (
              visit_date TEXT PRIMARY KEY, source_rows INTEGER, scored_rows INTEGER,
              avg_score REAL, revision INTEGER, quality_status TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS dim_date (date_key TEXT PRIMARY KEY, year INTEGER, month INTEGER, weekday INTEGER);
            CREATE TABLE IF NOT EXISTS dim_doctor (doctor_key TEXT PRIMARY KEY, doctor_fio TEXT, specialty TEXT, filial TEXT);
            CREATE TABLE IF NOT EXISTS dim_specialty (specialty TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS dim_branch (filial TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS dim_diagnosis (diagnosis_code TEXT PRIMARY KEY, diagnosis_label TEXT);
            CREATE TABLE IF NOT EXISTS dim_service (service_code TEXT PRIMARY KEY, service_name TEXT);
            CREATE TABLE IF NOT EXISTS dim_document_kind (document_kind TEXT PRIMARY KEY, label TEXT);
            CREATE TABLE IF NOT EXISTS crm_case_state (
              mis_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'new', assignee TEXT,
              due_date TEXT, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS crm_case_event (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT, mis_id TEXT NOT NULL,
              event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS saved_view (
              view_id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
              filters_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_case_date ON fact_mo_case(visit_date);
            CREATE INDEX IF NOT EXISTS idx_case_org ON fact_mo_case(filial, specialty);
            """
        )
        db.executemany(
            "INSERT OR IGNORE INTO dim_document_kind(document_kind, label) VALUES (?, ?)",
            [(kind, kind.replace("_", " ")) for kind in sorted(DOCUMENT_KINDS)],
        )


def upsert_warehouse(path: Path, raw_rows: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> None:
    initialize_warehouse(path)
    case_by_id = {str(row.get("mis_id") or ""): row for row in cases}
    now = utc_now()
    with sqlite3.connect(path) as db:
        for raw in raw_rows:
            mis_id = str(raw.get("id") or "")
            if not mis_id:
                continue
            case = case_by_id.get(mis_id, {})
            doctor_fio = str(raw.get("doctor_fio") or "")
            doctor_key = hashlib.sha256(doctor_fio.encode("utf-8")).hexdigest()[:20] if doctor_fio else ""
            payload = json.dumps(raw, sort_keys=True, default=_json_default)
            db.execute(
                """INSERT INTO fact_mo_case VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(mis_id) DO UPDATE SET
                   visit_id=excluded.visit_id, visit_date=excluded.visit_date,
                   document_kind=excluded.document_kind, overall_pct=excluded.overall_pct,
                   status=excluded.status, doctor_key=excluded.doctor_key,
                   specialty=excluded.specialty, filial=excluded.filial,
                   content_hash=excluded.content_hash, updated_at=excluded.updated_at""",
                (
                    mis_id,
                    str(raw.get("visit_id") or ""),
                    str(raw.get("visit_date") or "")[:10],
                    str(raw.get("document_kind") or "unknown"),
                    _safe_number(case.get("overall_pct")),
                    str(case.get("status") or ""),
                    doctor_key,
                    str(raw.get("doctor_specialization") or ""),
                    str(raw.get("filial") or ""),
                    hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    now,
                ),
            )
            if doctor_key:
                db.execute(
                    "INSERT OR REPLACE INTO dim_doctor VALUES (?, ?, ?, ?)",
                    (doctor_key, doctor_fio, str(raw.get("doctor_specialization") or ""), str(raw.get("filial") or "")),
                )
        summary = report.get("summary") or {}
        db.execute(
            "INSERT OR REPLACE INTO fact_mo_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                report.get("date"),
                summary.get("source_rows"),
                summary.get("scored"),
                summary.get("avg_score"),
                report.get("revision"),
                "passed" if (report.get("quality") or {}).get("passed") else "blocked",
                now,
            ),
        )


def public_case_token(value: str, secret: str) -> str:
    if not secret:
        raise ValueError("HMAC secret обязателен")
    return hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def catch_up_dates(
    *,
    successful_dates: Iterable[str],
    yesterday: date,
    first_date: date | None = None,
    limit: int = 31,
) -> list[date]:
    success = {date.fromisoformat(item) for item in successful_dates}
    start = first_date or ((max(success) + timedelta(days=1)) if success else yesterday)
    output = []
    cursor = start
    while cursor <= yesterday and len(output) < limit:
        if cursor not in success:
            output.append(cursor)
        cursor += timedelta(days=1)
    return output
