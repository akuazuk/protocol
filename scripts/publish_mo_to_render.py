#!/usr/bin/env python3
"""Опубликовать данные МО на диск Render, чтобы прод показывал свежую аналитику.

Что уезжает в `/var/data/medical_exams`:

- `warehouse/mo_analytics.sqlite` - факты и справочники, **без** таблиц кабинета:
  статусы и события методистов живут в проде и не затираются публикацией;
- `reports/YYYY/MM/DD/report.json` - отчёты дней (страница «Вчера», список отчётов);
- `secure_cases/YYYY/MM/*` - оценки и дневные CSV за последние `--days` дней;
- `state/pipeline.json`, `public/*` - свежесть и обезличенные снапшоты.

Каналом служит ssh/rsync на сервис Render (как в `render_mis_protocol_data.sh`).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_publish import (  # noqa: E402
    build_publish_snapshot,
    merge_sql,
    snapshot_summary,
)

DEFAULT_SSH_HOST = "srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com"
DEFAULT_REMOTE_ROOT = "/var/data/medical_exams"
DEFAULT_LEGACY_REMOTE_DIR = "/var/data/mis_protocol"
SSH_OPTS = ("-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=30")


class Publisher:
    def __init__(
        self,
        *,
        data_root: Path,
        ssh_host: str,
        remote_root: str,
        dry_run: bool = False,
    ) -> None:
        self.data_root = data_root
        self.ssh_host = ssh_host
        self.remote_root = remote_root.rstrip("/")
        self.dry_run = dry_run
        self.commands: list[str] = []

    def _run(self, command: Sequence[str], *, capture: bool = False, attempts: int = 4) -> str:
        """SSH на Render рвёт соединение на больших файлах - повторяем с паузой."""
        self.commands.append(" ".join(shlex.quote(part) for part in command))
        if self.dry_run:
            return ""
        last_error: subprocess.CalledProcessError | None = None
        for attempt in range(1, attempts + 1):
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE if capture else None,
                )
                return (result.stdout or "").strip() if capture else ""
            except subprocess.CalledProcessError as error:
                last_error = error
                if attempt == attempts:
                    break
                delay = 10 * attempt
                print(
                    f"попытка {attempt}/{attempts} не удалась (код {error.returncode}), "
                    f"повтор через {delay} с",
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise last_error  # type: ignore[misc]

    def ssh(self, script: str, *, capture: bool = False) -> str:
        return self._run(["ssh", *SSH_OPTS, self.ssh_host, script], capture=capture)

    def rsync(self, source: Path, remote_subdir: str, *, mirror: bool = False) -> None:
        # rsync не создаёт вложенные каталоги (secure_cases/2026/07) - делаем это заранее.
        remote_dir = f"{self.remote_root}/{remote_subdir}".rstrip("/")
        self.ssh(f"mkdir -p {shlex.quote(remote_dir)}")
        target = f"{self.ssh_host}:{remote_dir}"
        command = ["rsync", "-az", "-e", f"ssh {' '.join(SSH_OPTS)}"]
        if mirror:
            command.append("--delete-after")
        self._run([*command, f"{source}/", f"{target}/"])

    def upload_file(self, source: Path, remote_path: str) -> None:
        """Поток gzip через ssh: scp и rsync на больших файлах Render обрывает.

        Пишем во временный файл и переименовываем, чтобы обрыв не оставил
        полуфайл вместо рабочих данных.
        """
        remote_tmp = f"{remote_path}.part"
        script = (
            f"set -e; gunzip -c > {shlex.quote(remote_tmp)}; "
            f"mv {shlex.quote(remote_tmp)} {shlex.quote(remote_path)}"
        )
        command = ["ssh", *SSH_OPTS, self.ssh_host, script]
        self.commands.append(
            f"gzip -c {shlex.quote(str(source))} | "
            + " ".join(shlex.quote(part) for part in command)
        )
        if self.dry_run:
            return
        self._stream_gzip(source, command)

    def _stream_gzip(self, source: Path, command: Sequence[str], *, attempts: int = 4) -> None:
        for attempt in range(1, attempts + 1):
            with source.open("rb") as handle:
                gzip_proc = subprocess.Popen(["gzip", "-c"], stdin=handle, stdout=subprocess.PIPE)
                assert gzip_proc.stdout is not None
                ssh_proc = subprocess.Popen(command, stdin=gzip_proc.stdout)
                gzip_proc.stdout.close()
                ssh_code = ssh_proc.wait()
                gzip_proc.wait()
            if ssh_code == 0:
                return
            if attempt == attempts:
                raise RuntimeError(f"не удалось передать {source.name}: ssh код {ssh_code}")
            delay = 10 * attempt
            print(
                f"передача {source.name}: попытка {attempt}/{attempts} не удалась "
                f"(код {ssh_code}), повтор через {delay} с",
                file=sys.stderr,
            )
            time.sleep(delay)


def _recent_days(data_root: Path, days: int) -> list[Path]:
    """Каталоги secure_cases с оценками за последние `days` дней."""
    today = date.today()
    months = {
        (today - timedelta(days=offset)).strftime("%Y/%m") for offset in range(max(days, 1))
    }
    return [
        path
        for month in sorted(months)
        if (path := data_root / "secure_cases" / month).is_dir()
    ]


def _months(first: str, last: str) -> list[str]:
    current = date.fromisoformat(first + "-01")
    stop = date.fromisoformat(last + "-01")
    result: list[str] = []
    while current <= stop:
        result.append(current.strftime("%Y-%m"))
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
    return result


def publish_legacy_months(
    publisher: Publisher,
    *,
    months: Sequence[str],
    source_dir: Path,
    remote_dir: str,
) -> dict[str, Any]:
    """Заменить месячные файлы оценок в проде на файлы с deep-оценками.

    Прод читает `kz_l1_YYYY-MM_cases.jsonl` из `/var/data/mis_protocol`. Там лежали
    файлы после пересчёта L1 без блока `deep`, поэтому прод показывал L1-балл (~74),
    а витрина - deep (~87). Заменяем, сохранив прежний файл рядом.
    """
    stamp = date.today().strftime("%Y%m%d")
    moved: dict[str, Any] = {"months": [], "missing": []}
    publisher.ssh(f"mkdir -p {shlex.quote(remote_dir)}")
    for month in months:
        source = source_dir / f"kz_l1_{month}_cases.jsonl"
        if not source.is_file():
            moved["missing"].append(month)
            continue
        remote_path = f"{remote_dir}/kz_l1_{month}_cases.jsonl"
        backup = f"{remote_dir}/kz_l1_{month}_cases.pre_deep_{stamp}.jsonl"
        publisher.ssh(
            f"set -e; if [ -f {shlex.quote(remote_path)} ] && [ ! -f {shlex.quote(backup)} ]; "
            f"then cp {shlex.quote(remote_path)} {shlex.quote(backup)}; fi"
        )
        publisher.upload_file(source, remote_path)
        moved["months"].append(month)
    return moved


def publish(args: argparse.Namespace) -> dict[str, Any]:
    data_root: Path = args.data_root.expanduser()
    warehouse = (args.warehouse or data_root / "warehouse" / "mo_analytics.sqlite").expanduser()
    publisher = Publisher(
        data_root=data_root,
        ssh_host=args.ssh_host,
        remote_root=args.remote_root,
        dry_run=args.dry_run,
    )
    report: dict[str, Any] = {"remote_root": publisher.remote_root, "dry_run": args.dry_run}

    with tempfile.TemporaryDirectory() as tmp:
        if args.skip_warehouse:
            report["warehouse"] = "skipped"
            _publish_files(publisher, data_root, args, report)
            return report

        snapshot = Path(tmp) / "mo_analytics.publish.sqlite"
        report["tables"] = build_publish_snapshot(warehouse, snapshot)
        report["snapshot"] = snapshot_summary(snapshot)
        if report["snapshot"]["crm_rows"]:
            raise RuntimeError("в снапшоте остались строки CRM: публикация затрёт работу методистов")

        remote_warehouse = f"{publisher.remote_root}/warehouse/mo_analytics.sqlite"
        remote_snapshot = f"{publisher.remote_root}/warehouse/mo_analytics.publish.sqlite"
        publisher.ssh(
            f"mkdir -p {shlex.quote(publisher.remote_root + '/warehouse')} "
            f"{shlex.quote(publisher.remote_root + '/reports')} "
            f"{shlex.quote(publisher.remote_root + '/secure_cases')} "
            f"{shlex.quote(publisher.remote_root + '/state')} "
            f"{shlex.quote(publisher.remote_root + '/public')}"
        )
        publisher.upload_file(snapshot, remote_snapshot)

        merge_path = Path(tmp) / "merge.sql"
        merge_path.write_text(
            merge_sql(sorted(report["tables"]), snapshot_path=remote_snapshot), encoding="utf-8"
        )
        remote_merge = f"{publisher.remote_root}/warehouse/merge.sql"
        publisher.upload_file(merge_path, remote_merge)
        # Если продовой витрины ещё нет, снапшот и есть витрина: CRM-таблицы создаст API.
        script = (
            f"set -e; if [ -f {shlex.quote(remote_warehouse)} ]; then "
            f"sqlite3 {shlex.quote(remote_warehouse)} < {shlex.quote(remote_merge)}; "
            f"else cp {shlex.quote(remote_snapshot)} {shlex.quote(remote_warehouse)}; fi; "
            f"rm -f {shlex.quote(remote_snapshot)} {shlex.quote(remote_merge)}; "
            f"sqlite3 {shlex.quote(remote_warehouse)} "
            f"\"SELECT 'days=' || COUNT(*) || ' cases=' || (SELECT COUNT(*) FROM fact_mo_case) "
            f"|| ' through=' || (SELECT MAX(visit_date) FROM fact_mo_daily) FROM fact_mo_daily;\""
        )
        report["remote_warehouse"] = publisher.ssh(script, capture=True)

    _publish_files(publisher, data_root, args, report)
    return report


def _publish_files(
    publisher: Publisher,
    data_root: Path,
    args: argparse.Namespace,
    report: dict[str, Any],
) -> None:
    for subdir in ("reports", "state", "public"):
        source = data_root / subdir
        if source.is_dir():
            publisher.rsync(source, subdir, mirror=subdir == "public")
    for month_dir in _recent_days(data_root, args.days):
        publisher.rsync(month_dir, f"secure_cases/{month_dir.parent.name}/{month_dir.name}")

    if args.legacy_first_month:
        report["legacy_months"] = publish_legacy_months(
            publisher,
            months=_months(args.legacy_first_month, args.legacy_last_month or args.legacy_first_month),
            source_dir=args.legacy_source.expanduser(),
            remote_dir=args.legacy_remote_dir,
        )

    report["commands"] = publisher.commands
    if args.verify and not args.dry_run:
        report["freshness"] = verify_freshness(args.prod_url, args.methodist_token)


def verify_freshness(prod_url: str, token: str | None) -> dict[str, Any]:
    url = prod_url.rstrip("/") + "/api/methodist/mo/freshness"
    command = ["curl", "-fsS", "--max-time", "45"]
    if token:
        command.extend(["-H", f"X-Methodist-Token: {token}"])
    command.append(url)
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
    except subprocess.CalledProcessError as error:
        return {"status": error.returncode, "error": (error.stderr or "").strip()}
    except json.JSONDecodeError as error:
        return {"status": "unreachable", "error": str(error)}
    return {
        "status": 200,
        "data_through": payload.get("data_through") or payload.get("latest_date"),
        "lag_days": payload.get("lag_days"),
        "roots": payload.get("roots"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "medical_exams")
    parser.add_argument("--warehouse", type=Path, default=None)
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument(
        "--days",
        type=int,
        default=45,
        help="за сколько последних дней везти secure_cases (месяцами)",
    )
    parser.add_argument(
        "--legacy-first-month",
        default=None,
        help="с какого месяца заменить kz_l1_YYYY-MM_cases.jsonl в проде (например 2026-01)",
    )
    parser.add_argument("--legacy-last-month", default=None)
    parser.add_argument(
        "--legacy-source",
        type=Path,
        default=ROOT / "data" / "ml" / "reports" / "deep_eval",
        help="каталог с месячными файлами оценок (deep)",
    )
    parser.add_argument("--legacy-remote-dir", default=DEFAULT_LEGACY_REMOTE_DIR)
    parser.add_argument("--prod-url", default="https://protocol-bimy.onrender.com")
    parser.add_argument("--methodist-token", default=None)
    parser.add_argument(
        "--skip-warehouse",
        action="store_true",
        help="не трогать витрину: только отчёты, состояние и месячные файлы",
    )
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = publish(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
