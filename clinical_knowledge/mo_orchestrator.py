"""Orchestrator ежедневного МО-pipeline с внедряемыми внешними действиями."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from clinical_knowledge.mis_export import build_export_command, export_artifacts
from clinical_knowledge.mo_daily import (
    MINSK,
    PII_FIELDS,
    add_document_taxonomy,
    assess_completeness,
    atomic_write_text,
    atomic_write_json,
    build_daily_report,
    catch_up_dates,
    exclusive_lock,
    install_daily_partition,
    load_jsonl,
    merge_daily_partitions,
    minsk_today,
    previous_week_dates,
    resolve_run_date,
    sha256_file,
    this_week_dates,
    upsert_warehouse,
    utc_now,
    validate_export,
    write_daily_report,
)

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

# Сколько раз catch-up пытается доделать `partial` день, прежде чем оставить его как есть.
PARTIAL_MAX_ATTEMPTS = 4

DIGEST_TITLES = {
    "ok": "Готово",
    "partial": "Доделывается",
    "blocked": "Не принято",
    "failed": "Ошибка",
}


def build_digest(events: Sequence[Mapping[str, Any]], *, now: datetime | None = None) -> str:
    """Одно сообщение на прогон: сначала проблемы, потом успешные дни.

    Поток «по сообщению на каждый день» превращал Telegram в шум, из которого
    не видно, что именно требует внимания.
    """
    moment = (now or datetime.now(MINSK)).astimezone(MINSK)
    by_level: dict[str, list[Mapping[str, Any]]] = {level: [] for level in DIGEST_TITLES}
    for event in events:
        by_level.setdefault(str(event.get("level") or "failed"), []).append(event)
    lines = [f"МО, приём {moment:%d.%m %H:%M} Europe/Minsk"]
    for level in ("failed", "blocked", "partial", "ok"):
        items = by_level.get(level) or []
        if not items:
            continue
        if level == "ok":
            days = ", ".join(f"{date.fromisoformat(str(item['day'])):%d.%m}" for item in items)
            rows = sum(int(item.get("rows") or 0) for item in items)
            scored = sum(int(item.get("scored") or 0) for item in items)
            lines.append(f"{DIGEST_TITLES[level]} ({len(items)}): {days}; строк {rows}, оценено {scored}")
            continue
        for item in items:
            day = f"{date.fromisoformat(str(item['day'])):%d.%m}"
            lines.append(f"{DIGEST_TITLES.get(level, level)}: {day} - {item.get('detail') or 'без деталей'}")
    return "\n".join(lines)


def read_sql_epam_health(path: Path | None) -> dict[str, Any]:
    """Read-only health signal from the neighbouring sync; never runs its jobs."""
    if path is None:
        return {"status": "not_configured"}
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    stat = path.stat()
    signal: dict[str, Any] = {
        "status": "present",
        "path": str(path),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                signal["upstream_status"] = payload.get("status")
                signal["upstream_finished_at"] = payload.get("finished_at") or payload.get("updated_at")
        except (OSError, json.JSONDecodeError):
            signal["status"] = "unreadable"
    return signal


def default_command_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=float(os.environ.get("MO_COMMAND_TIMEOUT_SEC", "21600")),
    )


def _safe_command(command: Sequence[str]) -> str:
    sensitive = ("password", "token", "secret", "mysql+pymysql")
    return " ".join("<redacted>" if any(item in part.lower() for item in sensitive) else part for part in command)


def run_with_retry(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    attempts: int,
    base_delay_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> subprocess.CompletedProcess[str]:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return runner(command)
        except (subprocess.SubprocessError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                sleep(base_delay_seconds * (2 ** (attempt - 1)))
    assert last_error is not None
    raise RuntimeError(f"Команда не выполнена после {attempts} попыток: {_safe_command(command)}") from last_error


@dataclass
class VpnController:
    runner: CommandRunner
    script: Path = Path.home() / "CURSOR" / "bin" / "vanya_vpn.sh"
    wait_seconds: float = 2.0
    max_checks: int = 15
    sleep: Callable[[float], None] = time.sleep

    def status(self) -> str:
        result = self.runner((str(self.script), "status"))
        value = (result.stdout or "").strip().splitlines()
        return value[-1] if value else "Unknown"

    def wait_for(self, wanted: str) -> None:
        for _ in range(self.max_checks):
            if self.status().lower().startswith(wanted.lower()):
                return
            self.sleep(self.wait_seconds)
        raise RuntimeError(f"VPN не достиг состояния {wanted}")

    @contextmanager
    def sql_window(self) -> Iterator[None]:
        initial = self.status()
        restore_connected = initial.lower().startswith("connected")
        primary_error: BaseException | None = None
        try:
            self.runner((str(self.script), "ensure-off"))
            self.wait_for("Disconnected")
            yield
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                if restore_connected:
                    self.runner((str(self.script), "ensure-on"))
                    self.wait_for("Connected")
                elif not self.status().lower().startswith("disconnected"):
                    self.runner((str(self.script), "ensure-off"))
                    self.wait_for("Disconnected")
            except BaseException as restore_error:
                if primary_error is None:
                    raise
                raise ExceptionGroup(
                    "SQL-этап завершился ошибкой и исходное состояние VPN не восстановлено",
                    [primary_error, restore_error],
                ) from restore_error


class PipelineState:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": 1, "dates": {}, "runs": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Некорректный pipeline state")
        payload.setdefault("dates", {})
        payload.setdefault("runs", [])
        return payload

    @property
    def successful_dates(self) -> list[str]:
        return sorted(key for key, value in self.data["dates"].items() if value.get("status") == "success")

    def rework_dates(self, *, partial_max_attempts: int = PARTIAL_MAX_ATTEMPTS) -> list[str]:
        """Дни `partial`, которые ещё имеет смысл доделать: их доберут retry и hourly."""
        return sorted(
            key
            for key, value in self.data["dates"].items()
            if value.get("status") == "partial" and int(value.get("attempts") or 0) < partial_max_attempts
        )

    def settled_dates(self, *, partial_max_attempts: int = PARTIAL_MAX_ATTEMPTS) -> list[str]:
        """Дни, которые не надо перезабирать: успешные плюс partial с исчерпанными попытками.

        Так catch-up доделывает недооценённый день, но не крутит его вечно.
        """
        settled = []
        for key, value in self.data["dates"].items():
            status = value.get("status")
            if status == "success":
                settled.append(key)
            elif status == "partial" and int(value.get("attempts") or 0) >= partial_max_attempts:
                settled.append(key)
        return sorted(settled)

    def mark_stale_runs(self, *, max_age_hours: float = 2.0) -> list[str]:
        stale: list[str] = []
        now = datetime.now().astimezone()
        for day, record in self.data["dates"].items():
            if record.get("status") in {"success", "failed", "partial"}:
                continue
            raw = str(record.get("heartbeat") or record.get("started_at") or "")
            try:
                heartbeat = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if (now - heartbeat.astimezone()).total_seconds() > max_age_hours * 3600:
                record.update(
                    {
                        "status": "failed",
                        "stage": "failed",
                        "finished_at": utc_now(),
                        "error": "stale run exceeded heartbeat limit",
                    }
                )
                stale.append(day)
        if stale:
            self.save()
        return stale

    def start(self, day: date, run_id: str) -> None:
        now = utc_now()
        existing = self.data["dates"].get(day.isoformat()) or {}
        self.data["dates"][day.isoformat()] = {
            **existing,
            "status": "extracting",
            "stage": "extracting",
            "run_id": run_id,
            "started_at": now,
            "heartbeat": now,
            "attempts": int(existing.get("attempts") or 0) + 1,
        }
        self.data["runs"].append({"run_id": run_id, "date": day.isoformat(), "started_at": now})
        self.data["runs"] = self.data["runs"][-200:]
        self.save()

    def stage(self, day: date, name: str, **fields: Any) -> None:
        record = self.data["dates"].setdefault(day.isoformat(), {})
        record.update({"status": name, "stage": name, "heartbeat": utc_now(), **fields})
        self.save()

    def success(self, day: date, **fields: Any) -> None:
        self.stage(day, "success", finished_at=utc_now(), error=None, **fields)

    def fail(self, day: date, error: BaseException) -> None:
        self.stage(day, "failed", finished_at=utc_now(), error=f"{type(error).__name__}: {str(error)[:500]}")

    def revision(self, day: date, content_hash: str) -> int:
        record = self.data["dates"].setdefault(day.isoformat(), {})
        previous_hash = record.get("content_hash")
        current = int(record.get("revision") or 0)
        revision = current if current and previous_hash == content_hash else current + 1
        record.update({"content_hash": content_hash, "revision": revision})
        self.save()
        return revision

    def save(self) -> None:
        atomic_write_json(self.path, self.data)


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    data_root: Path

    @property
    def state(self) -> Path:
        return self.data_root / "state" / "pipeline.json"

    @property
    def lock(self) -> Path:
        return self.data_root / "state" / "pipeline.lock"

    @property
    def warehouse(self) -> Path:
        return self.data_root / "warehouse" / "mo_analytics.sqlite"

    def staging(self, run_id: str) -> Path:
        return self.data_root / "staging" / run_id

    def secure_month(self, day: date) -> Path:
        return self.data_root / "secure_cases" / f"{day:%Y}" / f"{day:%m}"

    def daily_partition(self, day: date) -> Path:
        return self.data_root / "raw" / f"{day:%Y}" / f"{day:%m}" / f"mo_{day.isoformat()}.parquet"


class MoDailyPipeline:
    def __init__(
        self,
        paths: PipelinePaths,
        *,
        runner: CommandRunner = default_command_runner,
        vpn: VpnController | None = None,
        dry_run: bool = False,
        notify: Callable[[str], bool] | None = None,
        rules_path: Path | None = None,
    ):
        self.paths = paths
        self.runner = runner
        self.vpn = vpn or VpnController(runner)
        self.dry_run = dry_run
        self.notify = notify or (lambda _message: False)
        self.rules_path = rules_path or paths.project_root / "config" / "mo_document_kind_rules.json"
        self.events: list[dict[str, Any]] = []
        # Результаты прогона доступны даже когда run() бросает из-за сбойного дня.
        self.last_results: list[dict[str, Any]] = []
        self._collecting = False

    def _emit(self, day: date, level: str, **fields: Any) -> None:
        event = {"day": day.isoformat(), "level": level, **fields}
        self.events.append(event)
        if not self._collecting:
            self.notify(build_digest([event]))

    def _rules(self) -> Mapping[str, Any]:
        if not self.rules_path.is_file():
            return {}
        return json.loads(self.rules_path.read_text(encoding="utf-8"))

    def plan_commands(self, day: date) -> list[Sequence[str]]:
        staging = self.paths.staging("<run-id>")
        export = build_export_command(self.paths.project_root, staging, day)
        tag = day.isoformat()
        secure = self.paths.secure_month(day)
        batch = (
            os.environ.get("PYTHON", os.sys.executable),
            str(self.paths.project_root / "scripts" / "run_mis_protocol_l1_batch.py"),
            "--csv",
            str(secure / f"mo_{tag}.csv"),
            "--out-dir",
            str(secure),
            "--month",
            tag,
            "--direct",
            "--deep-eval",
            "--resume",
            "--workers",
            os.environ.get("MO_DAILY_WORKERS", "1"),
        )
        return [export, batch]

    def run_day(self, day: date, *, force: bool = False) -> dict[str, Any]:
        if self.dry_run:
            return {
                "date": day.isoformat(),
                "dry_run": True,
                "force": force,
                "commands": [list(c) for c in self.plan_commands(day)],
            }

        import pandas as pd

        state = PipelineState(self.paths.state)
        state.mark_stale_runs()
        previous_content_hash = (state.data.get("dates", {}).get(day.isoformat()) or {}).get("content_hash")
        run_id = f"{day.isoformat()}-{uuid.uuid4().hex[:10]}"
        staging = self.paths.staging(run_id)
        staging.mkdir(parents=True, exist_ok=False)
        state.start(day, run_id)
        try:
            artifacts = export_artifacts(staging, day)
            export_command = build_export_command(self.paths.project_root, staging, day)
            with self.vpn.sql_window():
                run_with_retry(
                    self.runner,
                    export_command,
                    attempts=int(os.environ.get("MO_DB_RETRIES", "5")),
                    base_delay_seconds=float(os.environ.get("MO_DB_RETRY_DELAY_SEC", "5")),
                )
            if not artifacts.parquet.is_file() or not artifacts.meta.is_file():
                raise FileNotFoundError("Экспортёр не создал parquet/meta")
            source_meta = json.loads(artifacts.meta.read_text(encoding="utf-8"))
            frame = add_document_taxonomy(pd.read_parquet(artifacts.parquet), self._rules())
            history = self._same_weekday_counts(day)
            quality = validate_export(
                frame,
                day=day,
                source_rows=(
                    int(source_meta.get("source_rows"))
                    if source_meta.get("source_rows") is not None
                    else (int(source_meta.get("rows")) if source_meta.get("rows") is not None else None)
                ),
                historical_same_weekday_counts=history,
            )
            quality_payload = quality.to_dict()
            health_path = os.environ.get("MO_SQL_EPAM_STATUS_FILE", "").strip()
            quality_payload["external_health"] = {
                "sql_epam": read_sql_epam_health(Path(health_path).expanduser() if health_path else None)
            }
            state.stage(day, "validating", quality=quality_payload)
            partition, meta_path = install_daily_partition(
                frame,
                day=day,
                root=self.paths.data_root,
                quality=quality,
                run_id=run_id,
                source_meta=source_meta,
            )
            if not quality.passed:
                raise RuntimeError(f"data-quality gate blocked: {[item.code for item in quality.blocking]}")

            state.stage(day, "scoring", partition=str(partition))
            month = f"{day:%Y-%m}"
            raw_dir = self.paths.data_root / "raw" / f"{day:%Y}" / f"{day:%m}"
            daily_paths = sorted(raw_dir.glob(f"mo_{month}-*.parquet"))
            secure_dir = self.paths.secure_month(day)
            _, _, merge_info = merge_daily_partitions(daily_paths, month=month, out_dir=secure_dir)
            daily_csv = secure_dir / f"mo_{day.isoformat()}.csv"
            atomic_write_text(
                daily_csv,
                frame.drop(columns=["result_raw"], errors="ignore").to_csv(index=False),
            )
            content_hash = sha256_file(partition)
            if force or (previous_content_hash and previous_content_hash != content_hash):
                for suffix in ("cases.jsonl", "state.jsonl", "summary.json", "llm_queue.json"):
                    (secure_dir / f"kz_l1_{day.isoformat()}_{suffix}").unlink(missing_ok=True)
            batch_command = list(self.plan_commands(day)[1])
            batch_command[batch_command.index("--csv") + 1] = str(daily_csv)
            run_with_retry(self.runner, batch_command, attempts=2, base_delay_seconds=10)

            state.stage(day, "reporting", merge=merge_info)
            cases_path = secure_dir / f"kz_l1_{day.isoformat()}_cases.jsonl"
            cases = load_jsonl(cases_path)
            raw_rows = frame.to_dict(orient="records")
            revision = state.revision(day, content_hash)
            month_cases = []
            for path in sorted(secure_dir.glob(f"kz_l1_{month}-??_cases.jsonl")):
                month_cases.extend(load_jsonl(path))
            month_scores = [
                float(row["overall_pct"])
                for row in month_cases
                if row.get("overall_pct") is not None and not row.get("error")
            ]
            mtd = {
                "month": month,
                "rows": int(sum(len(pd.read_parquet(path)) for path in daily_paths)),
                "scored": len(month_scores),
                "avg_score": round(sum(month_scores) / len(month_scores), 1) if month_scores else None,
            }
            completeness = assess_completeness(
                raw_rows,
                cases,
                llm_queue_pending=self._llm_queue_pending(secure_dir, day),
            )
            report, public = build_daily_report(
                raw_rows,
                cases,
                day=day,
                run_id=run_id,
                revision=revision,
                quality=quality_payload,
                month_to_date=mtd,
                comparisons=self._comparisons(day),
                completeness=completeness,
            )
            write_daily_report(report, public, day=day, root=self.paths.data_root)
            warehouse_written = upsert_warehouse(self.paths.warehouse, raw_rows, cases, report)

            state.stage(day, "publishing")
            self._public_smoke(public)
            status = "partial" if completeness["partial"] else "success"
            state.stage(
                day,
                status,
                finished_at=utc_now(),
                error=None,
                rows=len(frame),
                scored=len(cases),
                revision=revision,
                partition=str(partition),
                meta=str(meta_path),
                completeness=completeness,
                warehouse=warehouse_written,
            )
            self._emit(
                day,
                "partial" if status == "partial" else "ok",
                rows=len(frame),
                scored=len(cases),
                revision=revision,
                detail=(
                    f"покрытие оценки {completeness['coverage_pct']}% "
                    f"({', '.join(completeness['reasons'])})"
                    if status == "partial"
                    else f"строк {len(frame)}, оценено {len(cases)}"
                ),
            )
            return {
                "date": day.isoformat(),
                "status": status,
                "rows": len(frame),
                "scored": len(cases),
                "revision": revision,
                "coverage_pct": completeness["coverage_pct"],
            }
        except BaseException as exc:
            stage = state.data["dates"][day.isoformat()].get("stage")
            state.fail(day, exc)
            blocked = isinstance(exc, RuntimeError) and str(exc).startswith("data-quality gate blocked")
            self._emit(
                day,
                "blocked" if blocked else "failed",
                detail=str(exc)[:200] if blocked else f"этап {stage}: {type(exc).__name__}",
            )
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def select_days(
        self,
        *,
        date_value: str = "yesterday",
        catch_up: bool = False,
        reconcile_days: int = 0,
        catch_up_limit: int = 31,
        first_date: date | None = None,
        previous_week: bool = False,
        this_week: bool = False,
        state: PipelineState | None = None,
        now: datetime | None = None,
    ) -> list[date]:
        yesterday = minsk_today(now) - timedelta(days=1)
        days: list[date] = []
        if previous_week:
            days.extend(previous_week_dates(now=now))
        if this_week:
            days.extend(this_week_dates(now=now))
        if catch_up:
            days.extend(
                catch_up_dates(
                    successful_dates=state.settled_dates() if state else (),
                    yesterday=yesterday,
                    first_date=first_date,
                    limit=catch_up_limit,
                )
            )
            # Недоделанные дни лежат до последнего успешного, поэтому окно catch-up их не видит.
            if state:
                days.extend(
                    day
                    for raw in state.rework_dates()
                    if (day := date.fromisoformat(raw)) <= yesterday
                )
            if reconcile_days:
                reconcile = [yesterday - timedelta(days=offset) for offset in reversed(range(max(1, reconcile_days)))]
                days.extend(reconcile)
        elif reconcile_days:
            count = max(1, reconcile_days)
            days.extend(yesterday - timedelta(days=offset) for offset in reversed(range(count)))
        elif not previous_week and not this_week:
            days.append(resolve_run_date(date_value, now=now))
        return sorted(set(days))

    def run(
        self,
        *,
        date_value: str = "yesterday",
        catch_up: bool = False,
        reconcile_days: int = 0,
        catch_up_limit: int = 31,
        first_date: date | None = None,
        previous_week: bool = False,
        this_week: bool = False,
        force: bool = False,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        state = PipelineState(self.paths.state) if not self.dry_run else None
        days = self.select_days(
            date_value=date_value,
            catch_up=catch_up,
            reconcile_days=reconcile_days,
            catch_up_limit=catch_up_limit,
            first_date=first_date,
            previous_week=previous_week,
            this_week=this_week,
            state=state,
            now=now,
        )
        if self.dry_run:
            self.last_results = [self.run_day(day, force=force) for day in days]
            return self.last_results

        self.events = []
        self.last_results = []
        self._collecting = True
        results: list[dict[str, Any]] = []
        failures: list[str] = []
        force_day = force or previous_week or this_week
        try:
            with exclusive_lock(self.paths.lock):
                for day in days:
                    try:
                        results.append(self.run_day(day, force=force_day))
                    except BaseException as exc:  # один плохой день не должен ронять остальные
                        failures.append(f"{day.isoformat()}: {type(exc).__name__}")
                        results.append({"date": day.isoformat(), "status": "failed", "error": type(exc).__name__})
                        if isinstance(exc, KeyboardInterrupt):
                            raise
        finally:
            self._collecting = False
            self.last_results = results
            if self.events:
                self.notify(build_digest(self.events))
        if failures:
            raise RuntimeError("дни завершились ошибкой: " + "; ".join(failures))
        return results

    @staticmethod
    def _llm_queue_pending(secure_dir: Path, day: date) -> int:
        path = secure_dir / f"kz_l1_{day.isoformat()}_llm_queue.json"
        if not path.is_file():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, Mapping):
            for key in ("pending", "queue", "items", "cases"):
                value = payload.get(key)
                if isinstance(value, list):
                    return len(value)
                if isinstance(value, int):
                    return value
        return 0

    def _same_weekday_counts(self, day: date) -> list[int]:
        counts = []
        for weeks in range(1, 9):
            previous = day - timedelta(days=7 * weeks)
            meta = self.paths.data_root / "raw" / f"{previous:%Y}" / f"{previous:%m}" / f"mo_{previous.isoformat()}.meta.json"
            if meta.is_file():
                counts.append(int(json.loads(meta.read_text(encoding="utf-8")).get("rows") or 0))
        return counts

    def _comparisons(self, day: date) -> dict[str, Any]:
        previous = self.paths.data_root / "public" / "daily" / f"{day - timedelta(days=1)}.json"
        if not previous.is_file():
            return {"previous_day": None}
        payload = json.loads(previous.read_text(encoding="utf-8"))
        return {"previous_day": payload.get("summary")}

    @staticmethod
    def _public_smoke(public: Mapping[str, Any]) -> None:
        serialized = json.dumps(public, ensure_ascii=False, allow_nan=False)
        for field in PII_FIELDS:
            if f'"{field}"' in serialized:
                raise RuntimeError(f"ПДн-поле попало в public snapshot: {field}")


@contextmanager
def _null_context() -> Any:
    yield
