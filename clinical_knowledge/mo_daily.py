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
import re
import sqlite3
import statistics
import tempfile
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

MINSK = ZoneInfo("Europe/Minsk")
# clinical_visit - единственный тип для клинической оценки (L1/deep/№55/LLM).
# consultation сохранён как legacy-алиас в витрине до recompute.
DOCUMENT_KINDS = frozenset(
    {
        "clinical_visit",
        "procedure_session",
        "medical_exam",
        "consultation",
        "certificate",
        "diagnostic",
        "non_clinical",
        "empty",
        "unknown",
    }
)
# clinical_visit - канон; consultation - legacy-алиас в витрине/secure до recompute.
SCORED_DOCUMENT_KINDS = frozenset({"clinical_visit", "consultation"})
# SQL-фрагмент для KPI/очередей (legacy consultation = клинический приём).
SCORED_KIND_SQL = "document_kind IN ('clinical_visit', 'consultation')"
EMPTY_TOKENS = frozenset({"", "0", "1", "on", "off", "nan", "none", "null"})
_PROCEDURE_COMPLAINT_RE = re.compile(
    r"(?:яв\w*\s+)?на\s+(?:промыван|процедур|манипуляц|инъекц|перевяз|физиотерап|массаж|дренаж)",
    re.IGNORECASE,
)
_STOMATOLOGY_RE = re.compile(
    r"стоматолог|зубн(?:ой|ая|ые|ого)|ортодонт|пародонтолог",
    re.IGNORECASE,
)
_PROCEDURE_SERVICE_TOKENS = (
    "промывание",
    "инъекц",
    "перевяз",
    "вакуумный дренаж",
    "физиотерап",
    "массаж",
    "блокад",
    "пункц",
    "инстилляц",
    "электрофорез",
    "дренаж по",
)
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

# Обвал объёма: ночной экспорт «вчера» иногда отдаёт десятки строк вместо сотен,
# и такой день нельзя пускать в raw/ и в оценку. Порог сравнивается с медианой
# того же дня недели; для праздников и разовых перезаборов допустим override
# MO_VOLUME_COLLAPSE_RATIO=0 (гейт выключается).
VOLUME_COLLAPSE_RATIO = 0.35
VOLUME_COLLAPSE_MIN_SAMPLES = 3
# Пустой join врача означает, что mis_data не доехала: оценивать день бессмысленно.
DOCTOR_JOIN_BLOCK_PCT = 50.0
DOCTOR_JOIN_MIN_ROWS = 20
# Доля допущенных к оценке записей, ниже которой день считается недоделанным.
SCORING_COVERAGE_TARGET_PCT = 99.0


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


def _lock_owner_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return int(raw[0].strip())
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def exclusive_lock(path: Path, *, blocking: bool = False) -> Iterator[None]:
    """Эксклюзивная блокировка пайплайна; мёртвый pid в файле не держит замок.

    `fcntl.flock` снимается ОС при смерти процесса, но файл с pid может остаться.
    Если flock занят, а pid в файле мёртв - снимаем stale-файл и пробуем ещё раз.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _acquire() -> Any:
        stream = path.open("a+", encoding="utf-8")
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream.fileno(), flags)
        except BlockingIOError:
            stream.close()
            raise
        stream.seek(0)
        stream.truncate()
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        return stream

    try:
        handle = _acquire()
    except BlockingIOError as first:
        owner = _lock_owner_pid(path)
        if owner is not None and not _pid_alive(owner):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                handle = _acquire()
            except BlockingIOError as second:
                raise RuntimeError(f"МО-pipeline уже запущен: {path}") from second
        else:
            detail = f" pid={owner}" if owner is not None else ""
            raise RuntimeError(f"МО-pipeline уже запущен: {path}{detail}") from first
    try:
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_nonempty(row: Mapping[str, Any], *keys: str) -> bool:
    for key in keys:
        value = str(row.get(key) or "").strip().lower()
        if value and value not in EMPTY_TOKENS:
            return True
    return False


def is_scored_document_kind(kind: str | None) -> bool:
    """Оцениваем clinical_visit и legacy consultation."""
    return str(kind or "").strip() in SCORED_DOCUMENT_KINDS


def scored_kind_sql(alias: str | None = "c") -> str:
    """SQL-условие для витрины: clinical_visit + legacy consultation."""
    col = f"{alias}.document_kind" if alias else "document_kind"
    return f"{col} IN ('clinical_visit', 'consultation')"


def _is_procedure_session(row: Mapping[str, Any]) -> bool:
    """Короткий процедурный визит: манипуляция + слабый осмотр / «на промывание»."""
    has_manip = _text_nonempty(row, "manipulations")
    complaints = str(row.get("complaints") or "").strip()
    has_objective = _text_nonempty(row, "objective_status")
    has_anamnesis = _text_nonempty(row, "anamnesis_doctor", "anamnesis_auto")
    has_rich_exam = has_objective and (bool(complaints) or has_anamnesis)
    if has_rich_exam:
        return False
    services = str(row.get("service_names") or "").lower()
    has_proc_service = any(token in services for token in _PROCEDURE_SERVICE_TOKENS)
    has_consult_service = "консультац" in services
    procedure_complaint = bool(complaints) and bool(_PROCEDURE_COMPLAINT_RE.search(complaints))
    thin_note = not has_objective and not has_anamnesis
    if has_manip and thin_note and (not complaints or procedure_complaint):
        return True
    if procedure_complaint and has_manip and not has_objective:
        return True
    if has_proc_service and not has_consult_service and thin_note:
        return True
    if has_manip and thin_note and has_proc_service:
        return True
    return False


def classify_document_kind(row: Mapping[str, Any], rules: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """Детерминированная taxonomy для продуктового контура МО.

    В клинический score идёт только ``clinical_visit``. Стоматология, процедуры,
    профосмотры, диагностика и справки помечаются, но не оцениваются.
    """
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
    has_clinical = any(
        str(row.get(key) or "").strip().lower() not in EMPTY_TOKENS for key in text_fields
    )
    has_manip = _text_nonempty(row, "manipulations")
    has_services = _text_nonempty(row, "service_names", "service_codes")
    if not has_clinical and not has_manip and not has_services:
        return "empty", "нет клинического содержания"

    spec = str(row.get("doctor_specialization") or "")
    services_l = str(row.get("service_names") or "").lower()
    haystack = " ".join(
        str(row.get(key) or "").lower()
        for key in (
            "doctor_specialization",
            "service_codes",
            "service_names",
            "clinical_diagnosis",
            "pay_type_label",
        )
    )
    custom = rules.get("keywords") if isinstance(rules.get("keywords"), Mapping) else {}

    if _STOMATOLOGY_RE.search(spec) or "стоматолог" in haystack:
        return "non_clinical", "стоматология вне клинической оценки МО"
    non_clinical_tokens = tuple(
        custom.get(
            "non_clinical",
            ("медсестр", "медицинская сестра", "регистратор", "логопед", "стоматолог"),
        )
    )
    if any(str(token).lower() in haystack for token in non_clinical_tokens):
        return "non_clinical", "признак taxonomy: non_clinical"

    if _is_procedure_session(row):
        return "procedure_session", "манипуляция / короткий процедурный визит"

    diagnostic_tokens = tuple(
        custom.get(
            "diagnostic",
            ("узи", "ультразвук", "рентген", "эндоскоп", "лаборатор", "диагност"),
        )
    )
    has_consult_service = "консультац" in services_l
    legacy = str(row.get("kz_kind") or "").strip()
    if legacy == "diagnostic" or (
        any(str(token).lower() in haystack for token in diagnostic_tokens) and not has_consult_service
    ):
        return "diagnostic", "диагностическое исследование"

    medical_exam_tokens = tuple(
        custom.get(
            "medical_exam",
            (
                "профосмотр",
                "медосмотр",
                "медицинский осмотр",
                "предрейсов",
                "периодическ",
                "предварительн",
            ),
        )
    )
    if any(str(token).lower() in haystack for token in medical_exam_tokens):
        return "medical_exam", "признак taxonomy: medical_exam"
    pay_type = str(row.get("pay_type") or "").strip().removesuffix(".0")
    if pay_type in {str(v) for v in rules.get("medical_exam_pay_types", ["12"])}:
        return "medical_exam", f"тип оплаты {pay_type} (профосмотр/справка)"

    certificate_tokens = tuple(custom.get("certificate", ("справк", "выписк")))
    if any(str(token).lower() in haystack for token in certificate_tokens):
        return "certificate", "справка без подтверждённого признака медосмотра"

    if legacy == "non_clinical":
        return "non_clinical", "совместимая классификация kz_kind"
    if legacy == "certificate":
        return "certificate", "справка без подтверждённого признака медосмотра"
    if legacy == "kz" or has_clinical:
        return "clinical_visit", "клинический приём врача"
    return "unknown", "недостаточно признаков для taxonomy"


def add_document_taxonomy(frame: Any, rules: Mapping[str, Any] | None = None) -> Any:
    classified = frame.apply(lambda row: classify_document_kind(row.to_dict(), rules), axis=1)
    result = frame.copy()
    result["document_kind"] = [item[0] for item in classified]
    result["document_kind_reason"] = [item[1] for item in classified]
    result["mo_score_eligible"] = result["document_kind"].map(is_scored_document_kind)
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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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
    doctor_filled = metrics["doctor_fio_filled_pct"]
    if doctor_filled < 98:
        result.warnings.append(
            QualityIssue("doctor_missing", "warning", "заполненность врача ниже порога", doctor_filled, 98)
        )
    if len(frame) >= DOCTOR_JOIN_MIN_ROWS and doctor_filled < DOCTOR_JOIN_BLOCK_PCT:
        result.blocking.append(
            QualityIssue(
                "doctor_join_broken",
                "blocking",
                "join с mis_data не дал врача у большинства строк",
                doctor_filled,
                DOCTOR_JOIN_BLOCK_PCT,
            )
        )

    if historical_same_weekday_counts and len(frame) > 0:
        baseline = statistics.median(historical_same_weekday_counts)
        metrics["same_weekday_median_rows"] = baseline
        metrics["same_weekday_samples"] = len(historical_same_weekday_counts)
        delta = abs(len(frame) - baseline) / baseline if baseline else 0.0
        metrics["volume_delta_pct"] = round(delta * 100, 2)
        ratio = len(frame) / baseline if baseline else 1.0
        metrics["volume_ratio_pct"] = round(ratio * 100, 2)
        collapse_ratio = _env_float("MO_VOLUME_COLLAPSE_RATIO", VOLUME_COLLAPSE_RATIO)
        enough_history = len(historical_same_weekday_counts) >= VOLUME_COLLAPSE_MIN_SAMPLES
        if baseline and collapse_ratio > 0 and enough_history and ratio < collapse_ratio:
            result.blocking.append(
                QualityIssue(
                    "volume_collapse",
                    "blocking",
                    "объём резко ниже медианы того же дня недели: похоже на неполный экспорт",
                    metrics["volume_ratio_pct"],
                    round(collapse_ratio * 100, 2),
                )
            )
        elif baseline and delta > 0.5:
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


def _row_keys(row: Mapping[str, Any]) -> set[str]:
    """Идентификаторы записи: в выгрузке это `id`, в оценках - `mis_id`."""
    keys: set[str] = set()
    for field_name in ("mis_id", "id"):
        value = row.get(field_name)
        if value not in (None, ""):
            keys.add(f"mis:{value}")
    visit = row.get("visit_id")
    if visit not in (None, ""):
        keys.add(f"visit:{visit}")
    return keys


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
        score = case_overall_pct(row)
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

def assess_completeness(
    raw_rows: Sequence[Mapping[str, Any]],
    scored_cases: Sequence[Mapping[str, Any]],
    *,
    llm_queue_pending: int = 0,
) -> dict[str, Any]:
    """День `partial`, если оценка не покрыла допущенные записи или есть ошибки.

    Очередь LLM сама по себе - advisory: при coverage >= цели и без scoring_errors
    день может быть `success`, а `llm_queue_pending` остаётся в advisory_reasons.
    """
    eligible = [row for row in raw_rows if is_scored_document_kind(row.get("document_kind"))]
    failures = [row for row in scored_cases if row.get("error")]
    # Считаем оценённым только случай с оценкой: пустой результат без ошибки тоже пробел.
    scored_ok = sum(
        1 for row in scored_cases if not row.get("error") and case_overall_pct(row) is not None
    )
    # Покрытие считаем по допущенным к оценке записям: иначе оценки чужих типов дают >100%.
    scored_keys = {
        key
        for row in scored_cases
        if not row.get("error") and case_overall_pct(row) is not None
        for key in _row_keys(row)
    }
    covered = sum(1 for row in eligible if _row_keys(row) & scored_keys)
    coverage = round(100.0 * covered / len(eligible), 2) if eligible else 100.0
    reasons: list[str] = []
    advisory_reasons: list[str] = []
    if eligible and coverage < SCORING_COVERAGE_TARGET_PCT:
        reasons.append("scoring_coverage")
    if failures:
        reasons.append("scoring_errors")
    pending = int(llm_queue_pending)
    if pending > 0:
        # Очередь LLM блокирует день только вместе с дырами в оценке.
        if reasons:
            reasons.append("llm_queue_pending")
        else:
            advisory_reasons.append("llm_queue_pending")
    return {
        "eligible_rows": len(eligible),
        "covered_rows": covered,
        "scored_ok": scored_ok,
        "scoring_errors": len(failures),
        "llm_queue_pending": pending,
        "coverage_pct": coverage,
        "target_coverage_pct": SCORING_COVERAGE_TARGET_PCT,
        "partial": bool(reasons),
        "reasons": reasons,
        "advisory_reasons": advisory_reasons,
    }


def apply_completeness_policy(completeness: Mapping[str, Any]) -> dict[str, Any]:
    """Нормализовать сохранённую completeness под текущую политику LLM-очереди."""
    result = dict(completeness)
    coverage = float(result.get("coverage_pct") or 0.0)
    target = float(result.get("target_coverage_pct") or SCORING_COVERAGE_TARGET_PCT)
    scoring_errors = int(result.get("scoring_errors") or 0)
    pending = int(result.get("llm_queue_pending") or 0)
    blocking = [
        str(code)
        for code in (result.get("reasons") or [])
        if str(code) != "llm_queue_pending"
    ]
    if coverage < target and "scoring_coverage" not in blocking:
        blocking.append("scoring_coverage")
    if scoring_errors > 0 and "scoring_errors" not in blocking:
        blocking.append("scoring_errors")
    advisory = [
        str(code)
        for code in (result.get("advisory_reasons") or [])
        if str(code) != "llm_queue_pending"
    ]
    if pending > 0:
        if blocking:
            blocking.append("llm_queue_pending")
        else:
            advisory.append("llm_queue_pending")
    result["reasons"] = blocking
    result["advisory_reasons"] = advisory
    result["partial"] = bool(blocking)
    result["llm_queue_pending"] = pending
    return result


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
    completeness: Mapping[str, Any] | None = None,
    suppress_below: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scores = [score for row in scored_cases if (score := case_overall_pct(row)) is not None]
    kinds = Counter(str(row.get("document_kind") or "unknown") for row in raw_rows)
    failures = [row for row in scored_cases if row.get("error")]
    eligible_rows = [row for row in raw_rows if is_scored_document_kind(row.get("document_kind"))]
    eligible_visits = {
        str(row.get("visit_id") or row.get("id") or "")
        for row in eligible_rows
        if row.get("visit_id") or row.get("id")
    }
    severe_case_ids: set[str] = set()
    action_queue = []
    for row in scored_cases:
        score = case_overall_pct(row)
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
    completeness_payload = dict(completeness or assess_completeness(raw_rows, scored_cases))
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "date": day.isoformat(),
        "window": source_window(day),
        "revision": revision,
        "generated_at": utc_now(),
        "quality": dict(quality),
        "completeness": completeness_payload,
        "partial": bool(completeness_payload.get("partial")),
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
                        (case_score := case_overall_pct(row)) is not None
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
        "completeness": report["completeness"],
        "partial": report["partial"],
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


def case_overall_pct(case: Mapping[str, Any]) -> float | None:
    """Единая шкала витрины - primary v4, затем детерминированный deep fallback.

    `overall_pct_v3` хранится отдельно для окна сравнения методик и никогда не
    подменяет primary score после успешного v4.
    """
    evaluation_v4 = (
        case.get("evaluation_v4") if isinstance(case.get("evaluation_v4"), Mapping) else {}
    )
    v4_primary = bool((evaluation_v4.get("mode") or {}).get("primary"))
    for value in (
        evaluation_v4.get("score_pct") if v4_primary else None,
        (case.get("deep") or {}).get("overall_pct") if isinstance(case.get("deep"), Mapping) else None,
        (case.get("evaluation_v3") or {}).get("score_pct")
        if isinstance(case.get("evaluation_v3"), Mapping)
        else None,
        case.get("overall_pct"),
        case.get("core_overall_pct"),
    ):
        score = _safe_number(value)
        if score is not None:
            return score
    return None


def case_status(case: Mapping[str, Any]) -> str:
    """Статус берём из того же источника, что и балл."""
    evaluation_v4 = (
        case.get("evaluation_v4") if isinstance(case.get("evaluation_v4"), Mapping) else {}
    )
    for value in (
        evaluation_v4.get("status")
        if (evaluation_v4.get("mode") or {}).get("primary")
        else None,
        (case.get("deep") or {}).get("status") if isinstance(case.get("deep"), Mapping) else None,
        case.get("status"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _axes_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    values: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        candidate = (
            case.get("evaluation_v4")
            if isinstance(case.get("evaluation_v4"), Mapping)
            else {}
        )
        evaluation = (
            candidate
            if (candidate.get("mode") or {}).get("primary")
            else case.get("deep")
        )
        if not isinstance(evaluation, Mapping):
            continue
        for axis, raw in (evaluation.get("axes") or {}).items():
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
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><a href=\"{}\">МО</a></td></tr>".format(
            esc(item.get("priority")),
            esc(item.get("doctor_fio")),
            esc(item.get("score")),
            esc(item.get("reason")),
            esc(item.get("status")),
            esc(
                f"/api/methodist/mo/cases/{item.get('visit_id') or item.get('mis_id')}/document"
            ),
        )
        for item in queue[:100]
    ) or '<tr><td colspan="6" class="muted">Случаев для срочного разбора нет.</td></tr>'
    issues = list(quality.get("blocking") or []) + list(quality.get("warnings") or [])
    quality_html = "".join(
        f'<li class="{esc(item.get("severity"))}"><b>{esc(item.get("code"))}</b>: {esc(item.get("message"))}</li>'
        for item in issues
    ) or "<li>Блокирующих ошибок и предупреждений нет.</li>"
    generated = report.get("generated_at") or ""
    completeness = report.get("completeness") or {}
    partial_reasons = {
        "scoring_coverage": "оценены не все допущенные записи",
        "scoring_errors": "часть записей завершилась ошибкой оценки",
        "llm_queue_pending": "остались записи в очереди LLM",
    }
    banner_html = ""
    if report.get("partial"):
        details = ", ".join(
            partial_reasons.get(str(code), str(code)) for code in (completeness.get("reasons") or [])
        )
        banner_html = (
            '<section class="banner"><b>День доделывается</b>'
            f"<span>Покрытие оценки {esc(completeness.get('coverage_pct'))}% из "
            f"{esc(completeness.get('target_coverage_pct'))}% целевых. {esc(details)}. "
            "Цифры ниже неполные, повторный прогон запланирован.</span></section>"
        )
    elif completeness.get("advisory_reasons"):
        details = ", ".join(
            partial_reasons.get(str(code), str(code))
            for code in (completeness.get("advisory_reasons") or [])
        )
        pending = completeness.get("llm_queue_pending")
        banner_html = (
            '<section class="banner"><b>День принят с замечанием</b>'
            f"<span>{esc(details)}"
            f"{(' (' + esc(pending) + ')') if pending not in (None, '', 0) else ''}. "
            "Оценка покрыта, очередь LLM не блокирует итог.</span></section>"
        )
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
.banner{{background:#fff6e5;border:1px solid #f0d9a8;border-radius:14px;padding:14px 18px;margin-bottom:16px}}.banner b{{display:block;color:#8a5a12}}.banner span{{color:#7a6438}}
@media(max-width:980px){{.grid{{grid-template-columns:repeat(3,1fr)}}.two{{grid-template-columns:1fr}}}}
@media(max-width:600px){{main{{padding:18px}}header{{display:block}}.grid{{grid-template-columns:1fr 1fr}}.bar-row{{grid-template-columns:130px 1fr 38px}}}}
</style></head><body><main>
<header><div><p class="muted">Ежедневный контроль медицинских осмотров</p><h1>Отчёт МО за {esc(day.strftime('%d.%m.%Y'))}</h1></div>
<div class="muted">Ревизия {esc(report.get('revision'))}<br>Сформирован {esc(generated)}</div></header>
{banner_html}
<section class="grid">{card_html}</section>
<section class="two"><article class="panel"><h2>Оценка по направлениям</h2>{axis_html}</article>
<article class="panel"><h2>Качество данных</h2><p>Статус: <b>{'пройден' if quality.get('passed') else 'требует проверки'}</b></p>
<p class="muted">Распознано: {esc(quality_core.get('parse_ok_pct'))}% · Врач: {esc(quality_core.get('doctor_fio_filled_pct'))}%</p><ul>{quality_html}</ul></article></section>
<section class="two"><article class="panel"><h2>Типы документов</h2><table><thead><tr><th>Тип</th><th>Количество</th></tr></thead><tbody>{kinds_html}</tbody></table></article>
<article class="panel"><h2>Специальности</h2><table><thead><tr><th>Специальность</th><th>N</th><th>Средняя</th><th>Внимание</th></tr></thead><tbody>{organization_rows('specialties')}</tbody></table></article></section>
<section class="panel wide"><h2>Филиалы</h2><table><thead><tr><th>Филиал</th><th>N</th><th>Средняя</th><th>Внимание</th></tr></thead><tbody>{organization_rows('filials')}</tbody></table></section>
<section class="panel wide"><h2>Очередь разбора</h2><table><thead><tr><th>Приоритет</th><th>Врач</th><th>Оценка</th><th>Причина</th><th>Статус</th><th>МО</th></tr></thead><tbody>{queue_html}</tbody></table></section>
<footer>Отчёт содержит защищённые сведения и предназначен только для внутреннего контура.</footer>
</main></body></html>"""


DOCUMENT_KIND_LABELS = {
    "clinical_visit": "Клинический приём",
    "procedure_session": "Манипуляция / процедура",
    "medical_exam": "Профосмотр / медосмотр",
    "consultation": "Клинический приём (legacy)",
    "certificate": "Справка",
    "diagnostic": "Диагностическое исследование",
    "non_clinical": "Неклинический документ",
    "empty": "Пустой документ",
    "unknown": "Не определён",
}

# Операционные таблицы кабинета методиста. Схема описана здесь один раз: и pipeline,
# и API работают с одним файлом витрины, а раньше два файла расходились по схеме.
# Ключ CRM - `case_id` = `visit_id` МИС: разбор ведётся по визиту, а не по строке протокола.
CRM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crm_case_state (
  case_id TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'new',
  assignee TEXT,
  tags_json TEXT NOT NULL DEFAULT '[]',
  due_date TEXT,
  finding_decisions_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL,
  updated_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crm_case_event (
  event_id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  actor TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS saved_view (
  view_id TEXT PRIMARY KEY,
  owner TEXT NOT NULL,
  scope TEXT NOT NULL,
  name TEXT NOT NULL,
  filters_json TEXT NOT NULL,
  config_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS export_job (
  job_id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
  kind TEXT NOT NULL, filters_json TEXT NOT NULL, result_path TEXT,
  created_at TEXT NOT NULL, expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS access_log (
  access_id TEXT PRIMARY KEY, actor TEXT NOT NULL, role TEXT NOT NULL,
  action TEXT NOT NULL, doctor_key TEXT, case_id TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crm_dispute_state (
  dispute_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, case_id TEXT NOT NULL,
  finding_code TEXT, status TEXT NOT NULL, reason TEXT NOT NULL,
  actor TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  resolved_by TEXT
);
CREATE TABLE IF NOT EXISTS crm_review_pack (
  pack_id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  visit_id TEXT NOT NULL,
  mis_id TEXT,
  patient_id TEXT,
  visit_date TEXT,
  doctor_fio TEXT,
  specialty TEXT,
  filial TEXT,
  clinical_json TEXT NOT NULL,
  system_json TEXT NOT NULL,
  decision_json TEXT NOT NULL,
  training_use INTEGER NOT NULL DEFAULT 1,
  actor TEXT,
  created_at TEXT NOT NULL,
  supersedes_pack_id TEXT
);
CREATE TABLE IF NOT EXISTS crm_expert_user (
  expert_id TEXT PRIMARY KEY,
  login TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  display_name TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crm_expert_session (
  session_id TEXT PRIMARY KEY,
  expert_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  last_seen_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_crm_state_status ON crm_case_state(status);
CREATE INDEX IF NOT EXISTS idx_crm_state_assignee ON crm_case_state(assignee);
CREATE INDEX IF NOT EXISTS idx_crm_event_case_time ON crm_case_event(case_id, created_at);
CREATE INDEX IF NOT EXISTS idx_saved_view_owner ON saved_view(owner, scope);
CREATE INDEX IF NOT EXISTS idx_access_log_time ON access_log(created_at);
CREATE INDEX IF NOT EXISTS idx_access_log_doctor ON access_log(doctor_key, created_at);
CREATE INDEX IF NOT EXISTS idx_dispute_case_status ON crm_dispute_state(case_id, status);
CREATE INDEX IF NOT EXISTS idx_crm_review_pack_case ON crm_review_pack(case_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_crm_review_pack_training ON crm_review_pack(training_use, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_crm_expert_session_expert ON crm_expert_session(expert_id, expires_at);
"""
CRM_TABLES = (
    "crm_case_state",
    "crm_case_event",
    "saved_view",
    "export_job",
    "access_log",
    "crm_dispute_state",
    "crm_review_pack",
    "crm_expert_user",
    "crm_expert_session",
)

# Главы МКБ-10: нужны для группировки диагнозов без внешнего справочника.
_ICD_CHAPTERS: tuple[tuple[str, str, str], ...] = (
    ("A00", "B99", "Инфекционные и паразитарные болезни"),
    ("C00", "D48", "Новообразования"),
    ("D50", "D89", "Болезни крови и иммунные нарушения"),
    ("E00", "E90", "Эндокринные болезни и нарушения обмена"),
    ("F00", "F99", "Психические расстройства"),
    ("G00", "G99", "Болезни нервной системы"),
    ("H00", "H59", "Болезни глаза"),
    ("H60", "H95", "Болезни уха"),
    ("I00", "I99", "Болезни системы кровообращения"),
    ("J00", "J99", "Болезни органов дыхания"),
    ("K00", "K93", "Болезни органов пищеварения"),
    ("L00", "L99", "Болезни кожи"),
    ("M00", "M99", "Болезни костно-мышечной системы"),
    ("N00", "N99", "Болезни мочеполовой системы"),
    ("O00", "O99", "Беременность, роды и послеродовой период"),
    ("P00", "P96", "Состояния перинатального периода"),
    ("Q00", "Q99", "Врождённые аномалии"),
    ("R00", "R99", "Симптомы и отклонения от нормы"),
    ("S00", "T98", "Травмы и отравления"),
    ("V01", "Y98", "Внешние причины"),
    ("Z00", "Z99", "Факторы, влияющие на здоровье"),
    ("U00", "U99", "Особые цели"),
)


def icd_chapter(code: str) -> str:
    """Глава МКБ-10 по коду; пустая строка, если код нераспознаваем."""
    normalized = str(code or "").strip().upper().replace(" ", "")
    if len(normalized) < 3 or not normalized[0].isalpha():
        return ""
    head = normalized[:3]
    for start, end, title in _ICD_CHAPTERS:
        if start <= head <= end:
            return title
    return ""


def _split_multi(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return []
    return [part.strip() for part in raw.split("|") if part.strip()]


def _upgrade_crm_schema(db: sqlite3.Connection) -> list[str]:
    """Привести CRM-таблицы к единой схеме `case_id`.

    Ранние сборки витрины успели создать `crm_*` и `saved_view` со своими колонками.
    Пустые таблицы пересоздаём, непустые переименовываем в `*_legacy`, чтобы данные
    методиста не исчезли молча.
    """
    renamed: list[str] = []
    for table, required in (("crm_case_state", "case_id"), ("crm_case_event", "actor"), ("saved_view", "scope")):
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        if not columns or required in columns:
            continue
        rows = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if rows:
            db.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
            renamed.append(f"{table}_legacy")
        else:
            db.execute(f"DROP TABLE {table}")
    return renamed


def _ensure_columns(db: sqlite3.Connection, table: str, columns: Mapping[str, str]) -> None:
    existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    for name, sql_type in columns.items():
        if name not in existing:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


def initialize_warehouse(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS fact_mo_case (
              mis_id TEXT PRIMARY KEY, visit_id TEXT, visit_date TEXT NOT NULL,
              document_kind TEXT NOT NULL, overall_pct REAL, overall_pct_v3 REAL,
              status TEXT, scorer_version TEXT, score_schema_version TEXT,
              llm_cost_usd REAL DEFAULT 0,
              doctor_key TEXT, doctor_id TEXT, specialty TEXT, filial TEXT,
              patient_key TEXT,
              diagnosis_code TEXT, diagnosis_text TEXT, icd_chapter TEXT,
              mkb_code_main_source TEXT, mkb_code_main_slot TEXT,
              history_prior_n INTEGER DEFAULT 0, history_tier TEXT,
              content_hash TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fact_mo_patient_history_cache (
              patient_key TEXT PRIMARY KEY,
              lookback_days INTEGER,
              as_of_date TEXT NOT NULL,
              n_visits INTEGER NOT NULL,
              summary_json TEXT NOT NULL,
              visit_index_json TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fact_mo_finding (
              mis_id TEXT NOT NULL, finding_code TEXT NOT NULL, severity TEXT,
              passed INTEGER, evidence TEXT, source_ref TEXT, axis TEXT,
              title_ru TEXT, detail_ru TEXT, trust_level TEXT,
              penalty_applied INTEGER DEFAULT 0, needs_human INTEGER DEFAULT 0,
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
            CREATE TABLE IF NOT EXISTS fact_mo_doctor_daily (
              visit_date TEXT NOT NULL, doctor_key TEXT NOT NULL, specialty TEXT, filial TEXT,
              cases INTEGER, scored INTEGER, avg_score REAL, needs_attention INTEGER,
              critical INTEGER, updated_at TEXT NOT NULL,
              PRIMARY KEY (visit_date, doctor_key)
            );
            CREATE TABLE IF NOT EXISTS fact_mo_template_pair (
              pair_id TEXT PRIMARY KEY, doctor_key TEXT NOT NULL,
              case_id_a TEXT NOT NULL, case_id_b TEXT NOT NULL,
              similarity REAL NOT NULL, algorithm TEXT NOT NULL,
              threshold REAL NOT NULL, provenance_json TEXT NOT NULL,
              detected_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fact_mo_visit (
              visit_id TEXT PRIMARY KEY, visit_date TEXT NOT NULL,
              records INTEGER NOT NULL, scored_records INTEGER NOT NULL,
              overall_pct REAL, worst_severity TEXT, scorer_version TEXT,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fact_llm_usage (
              usage_id TEXT PRIMARY KEY, run_id TEXT, usage_date TEXT NOT NULL,
              tier TEXT, model TEXT, case_id TEXT, prompt_tokens INTEGER,
              completion_tokens INTEGER, cost_usd REAL, latency_ms INTEGER,
              status TEXT, retry_count INTEGER DEFAULT 0, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dim_finding (
              finding_code TEXT PRIMARY KEY, title_ru TEXT NOT NULL,
              why_important_ru TEXT NOT NULL, source_ref TEXT,
              axis TEXT, default_severity TEXT, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dim_date (date_key TEXT PRIMARY KEY, year INTEGER, month INTEGER, weekday INTEGER);
            CREATE TABLE IF NOT EXISTS dim_doctor (doctor_key TEXT PRIMARY KEY, doctor_fio TEXT, specialty TEXT, filial TEXT);
            CREATE TABLE IF NOT EXISTS dim_specialty (specialty TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS dim_branch (filial TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS dim_diagnosis (diagnosis_code TEXT PRIMARY KEY, diagnosis_label TEXT, chapter TEXT);
            CREATE TABLE IF NOT EXISTS dim_service (service_code TEXT PRIMARY KEY, service_name TEXT);
            CREATE TABLE IF NOT EXISTS dim_document_kind (document_kind TEXT PRIMARY KEY, label TEXT);
            CREATE INDEX IF NOT EXISTS idx_case_date ON fact_mo_case(visit_date);
            CREATE INDEX IF NOT EXISTS idx_case_org ON fact_mo_case(filial, specialty);
            CREATE INDEX IF NOT EXISTS idx_case_doctor ON fact_mo_case(doctor_key, visit_date);
            CREATE INDEX IF NOT EXISTS idx_case_date_document ON fact_mo_case(visit_date, document_kind);
            CREATE INDEX IF NOT EXISTS idx_case_date_specialty ON fact_mo_case(visit_date, specialty);
            CREATE INDEX IF NOT EXISTS idx_case_date_filial ON fact_mo_case(visit_date, filial);
            CREATE INDEX IF NOT EXISTS idx_case_status_date ON fact_mo_case(status, visit_date);
            CREATE INDEX IF NOT EXISTS idx_finding_code ON fact_mo_finding(finding_code, severity);
            CREATE INDEX IF NOT EXISTS idx_finding_severity_case ON fact_mo_finding(severity, mis_id);
            CREATE INDEX IF NOT EXISTS idx_axis_axis ON fact_mo_score_axis(axis);
            CREATE INDEX IF NOT EXISTS idx_doctor_daily_date ON fact_mo_doctor_daily(visit_date);
            CREATE INDEX IF NOT EXISTS idx_template_pair_doctor
              ON fact_mo_template_pair(doctor_key, detected_at);
            CREATE INDEX IF NOT EXISTS idx_visit_date ON fact_mo_visit(visit_date);
            CREATE INDEX IF NOT EXISTS idx_llm_usage_date ON fact_llm_usage(usage_date, tier);
            CREATE INDEX IF NOT EXISTS idx_llm_usage_case ON fact_llm_usage(case_id);
            CREATE INDEX IF NOT EXISTS idx_daily_date_quality ON fact_mo_daily(visit_date, quality_status);
            """
        )
        _upgrade_crm_schema(db)
        db.executescript(CRM_SCHEMA_SQL)
        _ensure_columns(
            db,
            "fact_mo_daily",
            {
                "eligible_rows": "INTEGER",
                "partial": "INTEGER",
                "coverage_pct": "REAL",
                "avg_documentation": "REAL",
                "avg_clinical_concordance": "REAL",
                "avg_safety": "REAL",
                "avg_regulatory": "REAL",
                "needs_attention": "INTEGER",
                "critical": "INTEGER",
            },
        )
        _ensure_columns(
            db,
            "fact_mo_case",
            {
                "diagnosis_code": "TEXT",
                "diagnosis_text": "TEXT",
                "icd_chapter": "TEXT",
                "overall_pct_v3": "REAL",
                "scorer_version": "TEXT",
                "score_schema_version": "TEXT",
                "llm_cost_usd": "REAL DEFAULT 0",
                "mkb_code_main_source": "TEXT",
                "mkb_code_main_slot": "TEXT",
                "patient_key": "TEXT",
                "doctor_id": "TEXT",
                "history_prior_n": "INTEGER DEFAULT 0",
                "history_tier": "TEXT",
                "zone1_pct": "REAL",
                "zone2a_pct": "REAL",
                "zone2b_pct": "REAL",
                "zone1_band": "TEXT",
                "zone2a_band": "TEXT",
                "zone2b_band": "TEXT",
                "zone2b_kp_status": "TEXT",
                "attention_primary": "TEXT",
                "attention_reason_ru": "TEXT",
                "rubric_json": "TEXT",
                "rubric_pct": "REAL",
                "layer_engine": "TEXT",
                "layer_updated_at": "TEXT",
            },
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_case_attention "
            "ON fact_mo_case(attention_primary, visit_date)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_case_zone_bands "
            "ON fact_mo_case(zone1_band, zone2a_band, zone2b_band, visit_date)"
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS fact_mo_patient_history_cache (
              patient_key TEXT PRIMARY KEY,
              lookback_days INTEGER,
              as_of_date TEXT NOT NULL,
              n_visits INTEGER NOT NULL,
              summary_json TEXT NOT NULL,
              visit_index_json TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )"""
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_case_patient_date ON fact_mo_case(patient_key, visit_date)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_case_patient_doctor "
            "ON fact_mo_case(patient_key, doctor_key, visit_date)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_case_patient_specialty "
            "ON fact_mo_case(patient_key, specialty, visit_date)"
        )
        _ensure_columns(
            db,
            "fact_mo_finding",
            {
                "axis": "TEXT",
                "title_ru": "TEXT",
                "detail_ru": "TEXT",
                "trust_level": "TEXT",
                "penalty_applied": "INTEGER DEFAULT 0",
                "needs_human": "INTEGER DEFAULT 0",
                "is_shadow": "INTEGER DEFAULT 0",
                "linked_fields_json": "TEXT",
                "link_hint_ru": "TEXT",
            },
        )
        _ensure_columns(db, "dim_diagnosis", {"chapter": "TEXT"})
        _ensure_columns(db, "dim_service", {"service_group": "TEXT"})
        db.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_case_date_chapter
              ON fact_mo_case(visit_date, icd_chapter);
            CREATE INDEX IF NOT EXISTS idx_case_specialty_chapter
              ON fact_mo_case(specialty, icd_chapter);
            """
        )
        db.executemany(
            "INSERT OR IGNORE INTO dim_document_kind(document_kind, label) VALUES (?, ?)",
            [(kind, DOCUMENT_KIND_LABELS.get(kind, kind.replace("_", " "))) for kind in sorted(DOCUMENT_KINDS)],
        )
        # saved_view и export_job наполняет только кабинет: системные пресеты живут в UI,
        # иначе они попадают в личный список методиста.
        db.execute("PRAGMA user_version=2")


def day_status(report: Mapping[str, Any]) -> str:
    """`blocked` -> данные не приняты, `partial` -> день доделывается, `passed` -> готов."""
    if not (report.get("quality") or {}).get("passed"):
        return "blocked"
    return "partial" if report.get("partial") else "passed"


def doctor_key_for(doctor_fio: Any) -> str:
    """Псевдонимный ключ врача: витрина группирует по нему, ФИО живёт в dim_doctor.

    Хеш берём от ФИО как есть (только без краевых пробелов), чтобы ключи совпадали
    с уже записанными строками витрины и история не разъехалась на два врача.
    """
    normalized = str(doctor_fio or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20] if normalized else ""


def patient_key_for(patient_id: Any) -> str:
    """Hash patient_id для склада (сырой id в fact_mo_case не пишем)."""
    normalized = str(patient_id or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20] if normalized else ""


_VERSIONISH_ORG = re.compile(
    r"^(?:v?\d+(?:\.\d+){1,3}(?:-?[a-z0-9._]+)?|deep-v\d|not-applicable)",
    re.I,
)


def sanitize_mo_org_label(
    value: Any,
    *,
    scorer_version: str = "",
    schema_version: str = "",
) -> str:
    """Убрать из specialty/filial мусор вроде scorer_version (`v4.0.0`) / schema (`4.0`)."""
    text = str(value or "").strip()
    if not text or text in {"-", "\u2014", "\u2013", "Не указано", "не указано"}:
        return ""
    if scorer_version and text == str(scorer_version).strip():
        return ""
    if schema_version and text == str(schema_version).strip():
        return ""
    if _VERSIONISH_ORG.match(text):
        return ""
    low = text.lower()
    if "scorer" in low or low.startswith("deep-"):
        return ""
    return text


_TEMPLATE_TEXT_FIELDS = (
    "complaints",
    "anamnesis_doctor",
    "objective_status",
    "exam_data",
    "clinical_diagnosis",
    "exam_recommendations",
    "treatment_recommendations",
)


def _template_shingles(row: Mapping[str, Any], size: int = 5) -> frozenset[str]:
    """Нормализованные точные шинглы без сохранения клинического текста."""
    text = " ".join(str(row.get(field) or "") for field in _TEMPLATE_TEXT_FIELDS).lower()
    tokens = re.findall(r"[a-zа-яё0-9]+", text)
    if len(tokens) < size:
        return frozenset(tokens)
    return frozenset(" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def detect_template_copies(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.85,
    max_cases_per_doctor: int = 200,
) -> list[dict[str, Any]]:
    """Точный Jaccard по 5-шинглам, ограниченный врачом и числом случаев.

    Сравниваются только разные случаи разных пациентов. Результат содержит
    идентификаторы, сходство и хеши провенанса, но не клинический текст.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("template_threshold_out_of_range")
    grouped: dict[str, list[tuple[str, str, frozenset[str], str]]] = defaultdict(list)
    for row in rows:
        doctor_key = doctor_key_for(row.get("doctor_fio"))
        case_id = str(row.get("id") or row.get("mis_id") or "").strip()
        patient_id = str(row.get("patient_id") or "").strip()
        shingles = _template_shingles(row)
        # Короткие клише вроде «без особенностей» не являются надёжным сигналом
        # копирования целой записи и дают слишком много ложных совпадений.
        if not doctor_key or not case_id or not patient_id or len(shingles) < 25:
            continue
        digest = hashlib.sha256("\n".join(sorted(shingles)).encode("utf-8")).hexdigest()
        grouped[doctor_key].append((case_id, patient_id, shingles, digest))
    pairs: list[dict[str, Any]] = []
    for doctor_key, documents in grouped.items():
        documents = documents[:max_cases_per_doctor]
        for index, left in enumerate(documents):
            for right in documents[index + 1 :]:
                if left[0] == right[0] or left[1] == right[1]:
                    continue
                union = left[2] | right[2]
                similarity = len(left[2] & right[2]) / len(union) if union else 0.0
                if similarity < threshold:
                    continue
                case_a, case_b = sorted((left[0], right[0]))
                pairs.append(
                    {
                        "pair_id": hashlib.sha256(
                            f"{doctor_key}:{case_a}:{case_b}".encode("utf-8")
                        ).hexdigest()[:32],
                        "doctor_key": doctor_key,
                        "case_id_a": case_a,
                        "case_id_b": case_b,
                        "similarity": round(similarity, 6),
                        "algorithm": "exact_jaccard_5_shingles_v1",
                        "threshold": threshold,
                        "provenance": {
                            "shingle_hash_a": left[3],
                            "shingle_hash_b": right[3],
                            "different_patient_gate": True,
                        },
                    }
                )
    return pairs


def _case_shadow_findings(case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Shadow concordance из deep (не из evaluation_v4 - там их нет)."""
    deep = case.get("deep") if isinstance(case.get("deep"), Mapping) else {}
    items = deep.get("shadow_findings") if isinstance(deep, Mapping) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _case_primary_findings(case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Primary findings: предпочитаем v4, иначе deep."""
    v4 = case.get("evaluation_v4") if isinstance(case.get("evaluation_v4"), Mapping) else None
    deep = case.get("deep") if isinstance(case.get("deep"), Mapping) else {}
    source = v4 if isinstance(v4, Mapping) else deep
    items = (source or {}).get("findings") if isinstance(source, Mapping) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _finding_is_severe_attention(finding: Mapping[str, Any]) -> bool:
    return (not finding.get("passed")) and str(finding.get("severity") or "") in {"P0", "P1"}


def upsert_warehouse(
    path: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> dict[str, int]:
    """Заполнить витрину за один день: факты, оси, дефекты, все dim и CRM-заготовки.

    Возвращает счётчики записанных строк по таблицам - по ним видно, что день
    доехал целиком, а не только в `fact_mo_case`.
    """
    initialize_warehouse(path)
    case_by_id = {str(row.get("mis_id")): row for row in cases if row.get("mis_id") not in (None, "")}
    # Оценка идёт по визиту: несколько КЗ одного визита сводятся в один случай
    # (`n_kz_per_visit`). Поэтому документ без своего mis_id наследует оценку визита,
    # но только когда на визит есть ровно один случай.
    by_visit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cases:
        if row.get("visit_id") not in (None, ""):
            by_visit[str(row.get("visit_id"))].append(row)
    case_by_visit = {visit: rows[0] for visit, rows in by_visit.items() if len(rows) == 1}
    day_key = str(report.get("date") or "")[:10]
    now = utc_now()
    written: Counter[str] = Counter()
    doctor_daily: dict[str, dict[str, Any]] = {}
    eligible_rows_count = 0
    eligible_scores: list[float] = []
    eligible_case_ids: set[str] = set()
    eligible_critical_ids: set[str] = set()
    eligible_attention_ids: set[str] = set()
    eligible_axis_scores: dict[str, list[float]] = defaultdict(list)
    with sqlite3.connect(path) as db:
        seen_ids: list[str] = []
        for raw in raw_rows:
            mis_id = str(raw.get("id") or "")
            if not mis_id:
                continue
            seen_ids.append(mis_id)
            case = case_by_id.get(mis_id) or case_by_visit.get(str(raw.get("visit_id") or "")) or {}
            doctor_fio = str(raw.get("doctor_fio") or "").strip()
            doctor_key = doctor_key_for(doctor_fio)
            doctor_id = str(
                raw.get("doctor_id")
                or raw.get("specialist_id_from_visit")
                or case.get("doctor_id")
                or ""
            ).strip()
            patient_key = patient_key_for(
                raw.get("patient_id") or case.get("patient_id") or ""
            )
            visit_date = str(raw.get("visit_date") or "")[:10] or day_key
            score = case_overall_pct(case)
            evaluation_v4 = (
                case.get("evaluation_v4")
                if isinstance(case.get("evaluation_v4"), Mapping)
                else {}
            )
            deep_case = case.get("deep") if isinstance(case.get("deep"), Mapping) else {}
            previous_score = _safe_number(case.get("overall_pct_v3"))
            if previous_score is None:
                previous_score = _safe_number(deep_case.get("overall_pct"))
            scorer_version = str(
                evaluation_v4.get("scorer_version")
                or case.get("scorer_version")
                or ("deep-v2-fallback" if deep_case else "")
            )
            if not scorer_version:
                scorer_version = "not-applicable-v4"
            score_schema_version = str(evaluation_v4.get("schema_version") or "")
            specialty = sanitize_mo_org_label(
                raw.get("doctor_specialization"),
                scorer_version=scorer_version,
                schema_version=score_schema_version,
            )
            filial = sanitize_mo_org_label(
                raw.get("filial"),
                scorer_version=scorer_version,
                schema_version=score_schema_version,
            )
            llm_cost = _safe_number(case.get("llm_cost_usd")) or 0.0
            eligible_document = is_scored_document_kind(raw.get("document_kind"))
            diagnosis_codes: list[str] = []
            diagnosis_code = ""
            mkb_code_main_source = "empty"
            mkb_code_main_slot = ""
            diagnosis_chapter = ""
            diagnosis_text = ""
            history_prior_n = 0
            history_tier = ""
            zone_cols: dict[str, Any] = {
                "zone1_pct": None,
                "zone2a_pct": None,
                "zone2b_pct": None,
                "zone1_band": None,
                "zone2a_band": None,
                "zone2b_band": None,
                "zone2b_kp_status": None,
                "attention_primary": "none",
                "attention_reason_ru": "",
                "rubric_json": None,
                "rubric_pct": None,
                "layer_engine": None,
                "layer_updated_at": None,
            }
            if not eligible_document:
                # Не клинический приём: без МКБ/истории/баллов (в таблице не показываем).
                score = None
                previous_score = None
            else:
                # Soft-fill МКБ для KPI/UI: явный слот кода → иначе слоты диагноза.
                # Agreement не трогаем (raw.mkb_code_agreement на исходном слоте экспорта).
                from clinical_knowledge.mo_icd_resolve import soft_fill_mkb_for_warehouse

                resolve_case = dict(raw)
                if isinstance(case, Mapping):
                    for key in (
                        "clinical_diagnosis",
                        "diagnosis_main_text",
                        "diagnosis_short",
                        "diagnosis_text",
                        "mis_diagnos",
                        "mis_diagnosis",
                        "diagnosis_mis",
                    ):
                        if not resolve_case.get(key) and case.get(key):
                            resolve_case[key] = case.get(key)
                mkb_fill = soft_fill_mkb_for_warehouse(resolve_case)
                diagnosis_codes = list(mkb_fill.get("codes") or [])
                diagnosis_code = str(mkb_fill.get("code") or "")
                mkb_code_main_source = str(mkb_fill.get("source") or "empty")
                mkb_code_main_slot = str(mkb_fill.get("slot_code") or "")
                diagnosis_chapter = icd_chapter(diagnosis_code)
                from clinical_knowledge.mo_patient_history_bundle import (
                    attach_bundle_to_case,
                    short_diagnosis_text_for_warehouse,
                    upsert_history_cache,
                )

                diagnosis_text = short_diagnosis_text_for_warehouse(resolve_case)
                if patient_key:
                    hist_case = {
                        "patient_key": patient_key,
                        "patient_id": raw.get("patient_id") or "",
                        "visit_date": visit_date,
                        "doctor_id": doctor_id,
                        "doctor_key": doctor_key,
                        "doctor_fio": doctor_fio,
                        "specialty": specialty,
                        "diagnosis_code": diagnosis_code,
                        "mis_id": mis_id,
                        "visit_id": str(raw.get("visit_id") or ""),
                    }
                    try:
                        hist_bundle = attach_bundle_to_case(hist_case, warehouse=db)
                        history_prior_n = int((hist_bundle.get("summary") or {}).get("n_visits") or 0)
                        history_tier = str(hist_bundle.get("tier") or "")
                        upsert_history_cache(
                            db,
                            patient_key=patient_key,
                            as_of_date=visit_date,
                            bundle=hist_bundle,
                        )
                        if isinstance(case, dict):
                            case["_patient_history"] = hist_bundle
                    except Exception:  # noqa: BLE001
                        history_prior_n = 0
                        history_tier = ""
                eligible_rows_count += 1
                if score is not None:
                    eligible_scores.append(score)
                selected_case_id = str(case.get("mis_id") or "")
                if selected_case_id:
                    eligible_case_ids.add(selected_case_id)
                finding_source = evaluation_v4 or deep_case
                primary_findings = (
                    list(finding_source.get("findings") or [])
                    if isinstance(finding_source, Mapping)
                    else []
                )
                shadow_findings = _case_shadow_findings(case) if isinstance(case, Mapping) else []
                if any(
                    str(finding.get("severity") or "") == "P0"
                    for finding in primary_findings
                    if isinstance(finding, Mapping)
                ):
                    eligible_critical_ids.add(selected_case_id or mis_id)
                if evaluation_v4.get("attention_required") or any(
                    _finding_is_severe_attention(finding)
                    for finding in (*primary_findings, *shadow_findings)
                    if isinstance(finding, Mapping)
                ):
                    eligible_attention_ids.add(selected_case_id or mis_id)
                try:
                    from clinical_knowledge.mo_zone_scores import (
                        clinical_slots_from_mapping,
                        compute_mo_zone_scores,
                        zones_scores_enabled,
                        warehouse_zone_columns,
                    )

                    if zones_scores_enabled():
                        block_scores = {}
                        if isinstance(case, Mapping):
                            if isinstance(case.get("block_scores"), Mapping):
                                block_scores = dict(case.get("block_scores") or {})
                            elif isinstance(evaluation_v4.get("block_scores"), Mapping):
                                block_scores = dict(evaluation_v4.get("block_scores") or {})
                        suggest = None
                        if isinstance(case, Mapping) and isinstance(case.get("protocol_suggest"), Mapping):
                            suggest = case.get("protocol_suggest")
                        zones = compute_mo_zone_scores(
                            {
                                "clinical": clinical_slots_from_mapping(raw, case if isinstance(case, Mapping) else None),
                                "meta": {
                                    "visit_date": visit_date,
                                    "visit_time": raw.get("visit_time") or (case.get("visit_time") if isinstance(case, Mapping) else None),
                                    "diagnosis_code": diagnosis_code,
                                    "mkb_code_main": diagnosis_code,
                                    "diagnosis_short": diagnosis_text,
                                },
                                "block_scores": block_scores,
                                "findings": [*primary_findings, *shadow_findings],
                                "patient_history": (
                                    case.get("_patient_history")
                                    if isinstance(case, Mapping)
                                    else None
                                ),
                                "protocol_suggest": suggest,
                                "document_kind": raw.get("document_kind"),
                                "score_eligible": True,
                            }
                        )
                        zone_cols.update(warehouse_zone_columns(zones))
                except Exception:  # noqa: BLE001
                    pass
            payload = json.dumps(raw, sort_keys=True, default=_json_default)
            db.execute(
                """INSERT INTO fact_mo_case
                   (mis_id, visit_id, visit_date, document_kind, overall_pct,
                    overall_pct_v3, status, scorer_version, score_schema_version,
                    llm_cost_usd,
                    doctor_key, doctor_id, specialty, filial, patient_key,
                    diagnosis_code, diagnosis_text, icd_chapter,
                    mkb_code_main_source, mkb_code_main_slot,
                    history_prior_n, history_tier,
                    zone1_pct, zone2a_pct, zone2b_pct,
                    zone1_band, zone2a_band, zone2b_band, zone2b_kp_status,
                    attention_primary, attention_reason_ru,
                    rubric_json, rubric_pct, layer_engine, layer_updated_at,
                    content_hash, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(mis_id) DO UPDATE SET
                   visit_id=excluded.visit_id, visit_date=excluded.visit_date,
                   document_kind=excluded.document_kind, overall_pct=excluded.overall_pct,
                   overall_pct_v3=excluded.overall_pct_v3,
                   status=excluded.status, scorer_version=excluded.scorer_version,
                   score_schema_version=excluded.score_schema_version,
                   llm_cost_usd=excluded.llm_cost_usd,
                   doctor_key=excluded.doctor_key, doctor_id=excluded.doctor_id,
                   specialty=excluded.specialty, filial=excluded.filial,
                   patient_key=excluded.patient_key,
                   diagnosis_code=excluded.diagnosis_code,
                   diagnosis_text=excluded.diagnosis_text,
                   icd_chapter=excluded.icd_chapter,
                   mkb_code_main_source=excluded.mkb_code_main_source,
                   mkb_code_main_slot=excluded.mkb_code_main_slot,
                   history_prior_n=excluded.history_prior_n,
                   history_tier=excluded.history_tier,
                   zone1_pct=excluded.zone1_pct, zone2a_pct=excluded.zone2a_pct,
                   zone2b_pct=excluded.zone2b_pct,
                   zone1_band=excluded.zone1_band, zone2a_band=excluded.zone2a_band,
                   zone2b_band=excluded.zone2b_band,
                   zone2b_kp_status=excluded.zone2b_kp_status,
                   attention_primary=excluded.attention_primary,
                   attention_reason_ru=excluded.attention_reason_ru,
                   rubric_json=excluded.rubric_json, rubric_pct=excluded.rubric_pct,
                   layer_engine=excluded.layer_engine,
                   layer_updated_at=excluded.layer_updated_at,
                   content_hash=excluded.content_hash, updated_at=excluded.updated_at""",
                (
                    mis_id,
                    str(raw.get("visit_id") or ""),
                    visit_date,
                    str(raw.get("document_kind") or "unknown"),
                    score,
                    previous_score,
                    case_status(case),
                    scorer_version,
                    score_schema_version,
                    llm_cost,
                    doctor_key,
                    doctor_id,
                    specialty,
                    filial,
                    patient_key,
                    diagnosis_code,
                    diagnosis_text,
                    diagnosis_chapter,
                    mkb_code_main_source,
                    mkb_code_main_slot,
                    history_prior_n,
                    history_tier,
                    zone_cols.get("zone1_pct"),
                    zone_cols.get("zone2a_pct"),
                    zone_cols.get("zone2b_pct"),
                    zone_cols.get("zone1_band"),
                    zone_cols.get("zone2a_band"),
                    zone_cols.get("zone2b_band"),
                    zone_cols.get("zone2b_kp_status"),
                    zone_cols.get("attention_primary"),
                    zone_cols.get("attention_reason_ru"),
                    zone_cols.get("rubric_json"),
                    zone_cols.get("rubric_pct"),
                    zone_cols.get("layer_engine"),
                    zone_cols.get("layer_updated_at"),
                    hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    now,
                ),
            )
            written["fact_mo_case"] += 1

            if doctor_key:
                db.execute(
                    "INSERT OR REPLACE INTO dim_doctor VALUES (?, ?, ?, ?)",
                    (doctor_key, doctor_fio, specialty, filial),
                )
                written["dim_doctor"] += 1
                if eligible_document:
                    bucket = doctor_daily.setdefault(
                        doctor_key,
                        {
                            "specialty": specialty,
                            "filial": filial,
                            "cases": 0,
                            "scores": [],
                            "attention": 0,
                            "critical": 0,
                        },
                    )
                    bucket["cases"] += 1
                    if score is not None:
                        bucket["scores"].append(score)
                    if (str(case.get("mis_id") or "") or mis_id) in eligible_attention_ids:
                        bucket["attention"] += 1
            if specialty:
                db.execute("INSERT OR IGNORE INTO dim_specialty VALUES (?)", (specialty,))
                written["dim_specialty"] += 1
            if filial:
                db.execute("INSERT OR IGNORE INTO dim_branch VALUES (?)", (filial,))
                written["dim_branch"] += 1
            for code in diagnosis_codes:
                db.execute(
                    "INSERT OR IGNORE INTO dim_diagnosis(diagnosis_code, diagnosis_label, chapter) VALUES (?, ?, ?)",
                    (code, "", icd_chapter(code)),
                )
                written["dim_diagnosis"] += 1
            service_codes = _split_multi(raw.get("service_codes"))
            service_names = _split_multi(raw.get("service_names"))
            for index, code in enumerate(service_codes):
                name = service_names[index] if index < len(service_names) else ""
                db.execute(
                    "INSERT OR IGNORE INTO dim_service(service_code, service_name, service_group) VALUES (?, ?, ?)",
                    (code, name, name.split(",")[0][:80]),
                )
                written["dim_service"] += 1

        if day_key:
            moment = date.fromisoformat(day_key)
            db.execute(
                "INSERT OR REPLACE INTO dim_date VALUES (?, ?, ?, ?)",
                (day_key, moment.year, moment.month, moment.weekday()),
            )
            written["dim_date"] += 1

        # Повторный прогон дня: записи, исчезнувшие из МИС, не должны оставаться в витрине.
        if day_key and seen_ids:
            placeholders = ",".join("?" for _ in seen_ids)
            stale = [
                row[0]
                for row in db.execute(
                    f"SELECT mis_id FROM fact_mo_case WHERE visit_date = ? AND mis_id NOT IN ({placeholders})",
                    (day_key, *seen_ids),
                )
            ]
            for mis_id in stale:
                db.execute("DELETE FROM fact_mo_case WHERE mis_id = ?", (mis_id,))
                db.execute("DELETE FROM fact_mo_finding WHERE mis_id = ?", (mis_id,))
                db.execute("DELETE FROM fact_mo_score_axis WHERE mis_id = ?", (mis_id,))
            written["deleted_stale_cases"] = len(stale)

        for case in cases:
            mis_id = str(case.get("mis_id") or "")
            if not mis_id:
                continue
            deep = (
                case.get("evaluation_v4")
                if isinstance(case.get("evaluation_v4"), Mapping)
                else case.get("deep")
            )
            deep = deep if isinstance(deep, Mapping) else {}
            db.execute("DELETE FROM fact_mo_finding WHERE mis_id = ?", (mis_id,))
            db.execute("DELETE FROM fact_mo_score_axis WHERE mis_id = ?", (mis_id,))
            for axis, value in (deep.get("axes") or {}).items():
                axis_score = _safe_number(value)
                if axis_score is None:
                    continue
                db.execute(
                    "INSERT OR REPLACE INTO fact_mo_score_axis VALUES (?, ?, ?)",
                    (mis_id, str(axis), axis_score),
                )
                written["fact_mo_score_axis"] += 1
                if mis_id in eligible_case_ids:
                    eligible_axis_scores[str(axis)].append(axis_score)
            for finding in deep.get("findings") or []:
                if not isinstance(finding, Mapping):
                    continue
                code = str(finding.get("code") or "").strip()
                if not code:
                    continue
                db.execute(
                    """INSERT OR REPLACE INTO fact_mo_finding
                       (mis_id, finding_code, severity, passed, evidence, source_ref,
                        axis, title_ru, detail_ru, trust_level, penalty_applied,
                        needs_human, is_shadow, linked_fields_json, link_hint_ru)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        mis_id,
                        code,
                        str(finding.get("severity") or ""),
                        1 if finding.get("passed") else 0,
                        str(finding.get("evidence") or "")[:2000],
                        str(finding.get("source_ref") or ""),
                        str(finding.get("axis") or ""),
                        str(finding.get("title_ru") or code),
                        str(finding.get("detail_ru") or ""),
                        str(finding.get("trust_level") or "D"),
                        1 if finding.get("penalty_applied") else 0,
                        1 if finding.get("needs_human") else 0,
                        1 if finding.get("shadow") or finding.get("is_shadow") else 0,
                        json.dumps(finding.get("linked_fields") or [], ensure_ascii=False),
                        str(finding.get("link_hint_ru") or ""),
                    ),
                )
                title = str(finding.get("title_ru") or code)
                detail = str(finding.get("detail_ru") or "").strip()
                why = detail or (
                    f"«{title}»: автоматическое замечание по правилам оценки МО. "
                    "Откройте цитату и источник, затем подтвердите или отклоните."
                )
                db.execute(
                    """INSERT INTO dim_finding
                       (finding_code,title_ru,why_important_ru,source_ref,axis,
                        default_severity,updated_at)
                       VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(finding_code) DO UPDATE SET
                       title_ru=excluded.title_ru,
                       why_important_ru=excluded.why_important_ru,
                       source_ref=excluded.source_ref,axis=excluded.axis,
                       default_severity=excluded.default_severity,
                       updated_at=excluded.updated_at""",
                    (
                        code,
                        title,
                        why,
                        str(finding.get("source_ref") or ""),
                        str(finding.get("axis") or ""),
                        str(finding.get("severity") or ""),
                        now,
                    ),
                )
                written["dim_finding"] += 1
                written["fact_mo_finding"] += 1
            # E3: shadow concordance - в витрину и очередь, без влияния на overall/axes.
            for finding in _case_shadow_findings(case):
                code = str(finding.get("code") or "").strip()
                if not code:
                    continue
                # Не затирать primary с тем же code.
                exists = db.execute(
                    "SELECT 1 FROM fact_mo_finding WHERE mis_id=? AND finding_code=?",
                    (mis_id, code),
                ).fetchone()
                if exists:
                    continue
                linked = finding.get("linked_fields") or []
                db.execute(
                    """INSERT OR REPLACE INTO fact_mo_finding
                       (mis_id, finding_code, severity, passed, evidence, source_ref,
                        axis, title_ru, detail_ru, trust_level, penalty_applied,
                        needs_human, is_shadow, linked_fields_json, link_hint_ru)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                    (
                        mis_id,
                        code,
                        str(finding.get("severity") or ""),
                        1 if finding.get("passed") else 0,
                        str(finding.get("evidence") or "")[:2000],
                        str(finding.get("source_ref") or ""),
                        str(finding.get("axis") or ""),
                        str(finding.get("title_ru") or code),
                        str(finding.get("detail_ru") or ""),
                        str(finding.get("trust_level") or "D"),
                        0,
                        1 if finding.get("needs_human") else 0,
                        json.dumps(linked if isinstance(linked, list) else [], ensure_ascii=False),
                        str(finding.get("link_hint_ru") or ""),
                    ),
                )
                title = str(finding.get("title_ru") or code)
                detail = str(finding.get("detail_ru") or "").strip()
                why = detail or f"Shadow-согласованность: «{title}»."
                db.execute(
                    """INSERT INTO dim_finding
                       (finding_code,title_ru,why_important_ru,source_ref,axis,
                        default_severity,updated_at)
                       VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(finding_code) DO UPDATE SET
                       title_ru=excluded.title_ru,
                       why_important_ru=excluded.why_important_ru,
                       source_ref=excluded.source_ref,axis=excluded.axis,
                       default_severity=excluded.default_severity,
                       updated_at=excluded.updated_at""",
                    (
                        code,
                        title,
                        why,
                        str(finding.get("source_ref") or ""),
                        str(finding.get("axis") or ""),
                        str(finding.get("severity") or ""),
                        now,
                    ),
                )
                written["dim_finding"] += 1
                written["fact_mo_finding"] += 1
                written["fact_mo_finding_shadow"] += 1
            severity = deep.get("n_by_severity") or {}
            severe = sum(int(severity.get(level) or 0) for level in ("P0", "P1"))
            if not severity:
                severe = sum(
                    1
                    for finding in deep.get("findings") or []
                    if isinstance(finding, Mapping)
                    and not finding.get("passed")
                    and str(finding.get("severity") or "") in {"P0", "P1"}
                )
            # Shadow P1 тоже поднимает critical в doctor_daily (очередь методиста).
            shadow_p1 = sum(
                1
                for finding in _case_shadow_findings(case)
                if _finding_is_severe_attention(finding)
            )
            if shadow_p1 and not severe:
                severe = shadow_p1
            bucket = doctor_daily.get(doctor_key_for(case.get("doctor_fio")))
            if bucket is not None and severe:
                bucket["critical"] += 1

        template_pairs = detect_template_copies(raw_rows)
        if seen_ids:
            placeholders = ",".join("?" for _ in seen_ids)
            db.execute(
                f"""DELETE FROM fact_mo_finding
                    WHERE finding_code='E_template_copy'
                      AND mis_id IN ({placeholders})""",
                seen_ids,
            )
            db.execute(
                f"DELETE FROM fact_mo_template_pair WHERE case_id_a IN ({placeholders}) "
                f"OR case_id_b IN ({placeholders})",
                (*seen_ids, *seen_ids),
            )
        for pair in template_pairs:
            db.execute(
                """INSERT OR REPLACE INTO fact_mo_template_pair
                   (pair_id,doctor_key,case_id_a,case_id_b,similarity,algorithm,
                    threshold,provenance_json,detected_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    pair["pair_id"],
                    pair["doctor_key"],
                    pair["case_id_a"],
                    pair["case_id_b"],
                    pair["similarity"],
                    pair["algorithm"],
                    pair["threshold"],
                    json.dumps(pair["provenance"], ensure_ascii=False),
                    now,
                ),
            )
            written["fact_mo_template_pair"] += 1
            for case_id, other_id in (
                (pair["case_id_a"], pair["case_id_b"]),
                (pair["case_id_b"], pair["case_id_a"]),
            ):
                db.execute(
                    """INSERT OR REPLACE INTO fact_mo_finding
                       (mis_id,finding_code,severity,passed,evidence,source_ref)
                       VALUES(?,?,?,?,?,?)""",
                    (
                        case_id,
                        "E_template_copy",
                        "P2",
                        0,
                        "",
                        f"template_pair:{pair['pair_id']}:{other_id}",
                    ),
                )
                written["fact_mo_finding"] += 1
                db.execute(
                    """INSERT OR IGNORE INTO dim_finding
                       (finding_code,title_ru,why_important_ru,source_ref,axis,
                        default_severity,updated_at)
                       VALUES ('E_template_copy','Подозрение на копирование шаблона между случаями',
                        'Текст МО почти совпадает с другим случаем. Проверьте, что жалобы, статус, диагноз и план индивидуализированы, а не скопированы из шаблона.',
                        'advisory:exact_jaccard_5_shingles_v1','documentation','P2',?)""",
                    (now,),
                )

        db.execute(
            """INSERT OR IGNORE INTO dim_finding
               (finding_code,title_ru,why_important_ru,source_ref,axis,
                default_severity,updated_at)
               SELECT finding_code,
                      COALESCE(MAX(NULLIF(title_ru,'')),finding_code),
                      'Автоматическое замечание по правилам оценки МО: откройте цитату и источник, затем подтвердите или отклоните.',
                      COALESCE(MAX(source_ref),''),
                      COALESCE(MAX(axis),''),
                      COALESCE(MAX(severity),''),
                      ?
               FROM fact_mo_finding GROUP BY finding_code""",
            (now,),
        )

        if day_key:
            db.execute("DELETE FROM fact_mo_visit WHERE visit_date = ?", (day_key,))
            db.execute(
                """INSERT OR REPLACE INTO fact_mo_visit
                   (visit_id,visit_date,records,scored_records,overall_pct,
                    worst_severity,scorer_version,updated_at)
                   SELECT c.visit_id,c.visit_date,c.records,c.scored_records,
                          c.overall_pct,
                          CASE
                            WHEN COALESCE(f.p0,0)>0 THEN 'P0'
                            WHEN COALESCE(f.p1,0)>0 THEN 'P1'
                            WHEN COALESCE(f.p2,0)>0 THEN 'P2'
                            WHEN COALESCE(f.p3,0)>0 THEN 'P3'
                            ELSE 'ok'
                          END,
                          c.scorer_version,?
                   FROM (
                     SELECT visit_id,visit_date,COUNT(*) records,
                            COUNT(overall_pct) scored_records,
                            ROUND(AVG(overall_pct),2) overall_pct,
                            MAX(scorer_version) scorer_version
                     FROM fact_mo_case
                     WHERE visit_date=? AND visit_id<>''
                     GROUP BY visit_id,visit_date
                   ) c
                   LEFT JOIN (
                     SELECT fc.visit_id,
                            SUM(CASE WHEN f.severity='P0' AND f.passed=0 THEN 1 ELSE 0 END) p0,
                            SUM(CASE WHEN f.severity='P1' AND f.passed=0 THEN 1 ELSE 0 END) p1,
                            SUM(CASE WHEN f.severity='P2' AND f.passed=0 THEN 1 ELSE 0 END) p2,
                            SUM(CASE WHEN f.severity='P3' AND f.passed=0 THEN 1 ELSE 0 END) p3
                     FROM fact_mo_case fc
                     JOIN fact_mo_finding f ON f.mis_id=fc.mis_id
                     WHERE fc.visit_date=?
                     GROUP BY fc.visit_id
                   ) f ON f.visit_id=c.visit_id""",
                (now, day_key, day_key),
            )
            written["fact_mo_visit"] = int(
                db.execute(
                    "SELECT COUNT(*) FROM fact_mo_visit WHERE visit_date=?", (day_key,)
                ).fetchone()[0]
            )

        if day_key:
            db.execute("DELETE FROM fact_mo_doctor_daily WHERE visit_date = ?", (day_key,))
            for doctor_key, bucket in doctor_daily.items():
                scores = bucket["scores"]
                db.execute(
                    "INSERT INTO fact_mo_doctor_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        day_key,
                        doctor_key,
                        bucket["specialty"],
                        bucket["filial"],
                        bucket["cases"],
                        len(scores),
                        round(statistics.fmean(scores), 1) if scores else None,
                        bucket["attention"],
                        bucket["critical"],
                        now,
                    ),
                )
                written["fact_mo_doctor_daily"] += 1

        summary = report.get("summary") or {}
        axes = report.get("axes") or {}
        eligible_rows = eligible_rows_count
        scored_rows = len(eligible_scores)
        avg_score = (
            round(statistics.fmean(eligible_scores), 1) if eligible_scores else None
        )
        needs_attention = len(eligible_attention_ids)
        daily_axes = {
            axis: round(statistics.fmean(values), 1)
            for axis, values in eligible_axis_scores.items()
            if values
        }
        critical = len(eligible_critical_ids)
        def resolved_axis(key: str) -> float | None:
            value = daily_axes.get(key)
            return _safe_number(value if value is not None else axes.get(key))

        db.execute(
            """INSERT INTO fact_mo_daily (
                 visit_date, source_rows, scored_rows, avg_score, revision, quality_status, updated_at,
                 eligible_rows, partial, coverage_pct, avg_documentation, avg_clinical_concordance,
                 avg_safety, avg_regulatory, needs_attention, critical
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(visit_date) DO UPDATE SET
                 source_rows=excluded.source_rows, scored_rows=excluded.scored_rows,
                 avg_score=excluded.avg_score, revision=excluded.revision,
                 quality_status=excluded.quality_status, updated_at=excluded.updated_at,
                 eligible_rows=excluded.eligible_rows, partial=excluded.partial,
                 coverage_pct=excluded.coverage_pct, avg_documentation=excluded.avg_documentation,
                 avg_clinical_concordance=excluded.avg_clinical_concordance,
                 avg_safety=excluded.avg_safety, avg_regulatory=excluded.avg_regulatory,
                 needs_attention=excluded.needs_attention, critical=excluded.critical""",
            (
                day_key,
                summary.get("source_rows"),
                scored_rows,
                avg_score,
                report.get("revision"),
                day_status(report),
                now,
                eligible_rows,
                1 if report.get("partial") else 0,
                round(100 * scored_rows / eligible_rows, 1) if eligible_rows else None,
                resolved_axis("documentation"),
                resolved_axis("clinical_concordance"),
                resolved_axis("safety"),
                resolved_axis("regulatory"),
                needs_attention,
                critical,
            ),
        )
        written["fact_mo_daily"] += 1

        # Заготовки CRM по визитам: статус методиста не перезаписываем, только добавляем новые.
        for item in report.get("action_queue") or []:
            case_id = str(item.get("visit_id") or item.get("mis_id") or "")
            if not case_id:
                continue
            cursor = db.execute(
                "INSERT OR IGNORE INTO crm_case_state"
                "(case_id, status, assignee, tags_json, due_date, finding_decisions_json, updated_at, updated_by)"
                " VALUES (?, 'new', ?, '[]', ?, '{}', ?, 'pipeline')",
                (case_id, item.get("assignee"), item.get("due_date"), now),
            )
            if cursor.rowcount:
                written["crm_case_state"] += 1
                db.execute(
                    "INSERT INTO crm_case_event(event_id, case_id, event_type, actor, payload_json, created_at)"
                    " VALUES (?, ?, 'queued', 'pipeline', ?, ?)",
                    (
                        uuid.uuid4().hex,
                        case_id,
                        json.dumps(
                            {"priority": item.get("priority"), "reason": item.get("reason"), "date": day_key},
                            ensure_ascii=False,
                        ),
                        now,
                    ),
                )
                written["crm_case_event"] += 1
    return dict(written)


def migrate_crm(source: Path, target: Path) -> dict[str, int]:
    """Перенести операционные таблицы из старого файла CRM в единую витрину.

    Источник открывается только на чтение и не меняется: старый файл остаётся
    как резервная копия. Повторный запуск ничего не дублирует.
    """
    if not source.is_file() or source.resolve() == target.resolve():
        return {}
    initialize_warehouse(target)
    moved: Counter[str] = Counter()
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as origin, sqlite3.connect(target) as destination:
        origin.row_factory = sqlite3.Row
        available = {row[0] for row in origin.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in CRM_TABLES:
            if table not in available:
                continue
            target_columns = [row[1] for row in destination.execute(f"PRAGMA table_info({table})")]
            source_columns = [row[1] for row in origin.execute(f"PRAGMA table_info({table})")]
            shared = [name for name in target_columns if name in source_columns]
            if not shared:
                continue
            marks = ",".join("?" for _ in shared)
            columns = ",".join(shared)
            for row in origin.execute(f"SELECT {columns} FROM {table}"):
                cursor = destination.execute(
                    f"INSERT OR IGNORE INTO {table}({columns}) VALUES ({marks})",
                    tuple(row[name] for name in shared),
                )
                moved[table] += cursor.rowcount
    return dict(moved)


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


def week_monday(day: date) -> date:
    """Понедельник календарной недели (Пн=0) для даты Europe/Minsk."""
    return day - timedelta(days=day.weekday())


def previous_week_dates(*, now: datetime | None = None) -> list[date]:
    """Прошлая полная неделя Пн-Вс относительно сегодня в Europe/Minsk."""
    this_monday = week_monday(minsk_today(now))
    prev_monday = this_monday - timedelta(days=7)
    return [prev_monday + timedelta(days=offset) for offset in range(7)]


def this_week_dates(*, now: datetime | None = None) -> list[date]:
    """Текущая неделя с понедельника по вчера (сегодня ещё не принимаем)."""
    today = minsk_today(now)
    yesterday = today - timedelta(days=1)
    monday = week_monday(today)
    if yesterday < monday:
        return []
    days: list[date] = []
    cursor = monday
    while cursor <= yesterday:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days
