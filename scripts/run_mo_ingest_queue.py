#!/usr/bin/env python3
"""Обработать очередь POST /ingest-visit. Только GCE (venv-mis + docker score).

  source /opt/protocol/deploy/gcp-app/load_mis_env.sh
  PYTHONPATH=/opt/protocol /opt/protocol/venv-mis/bin/python \
    scripts/run_mo_ingest_queue.py --once
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _allow_ck_without_pydantic() -> None:
    """venv-mis has PyMySQL, not pydantic; skip clinical_knowledge/__init__.py."""
    try:
        import pydantic  # noqa: F401
        return
    except ImportError:
        pass
    pkg = types.ModuleType("clinical_knowledge")
    pkg.__path__ = [str(ROOT / "clinical_knowledge")]  # type: ignore[attr-defined]
    sys.modules["clinical_knowledge"] = pkg


_allow_ck_without_pydantic()
from clinical_knowledge.mo_mis_catalog import (  # noqa: E402
    mark_job_done_from_warehouse,
    next_queued_job,
    update_ingest_job,
)

_log = logging.getLogger("protocol.mo_ingest_queue")


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _process_one() -> dict | None:
    from scripts.ingest_mo_visit_from_mis import ingest_visit, merge_visit_csv, fetch_visit

    job = next_queued_job()
    if not job:
        return None
    job_id = job["job_id"]
    visit_id = job["visit_id"]
    data_root = Path(os.environ.get("MO_DATA_ROOT") or "/var/data/medical_exams")
    _log.info("mo_ingest_start visit_id=%s job_id=%s", visit_id, job_id)
    try:
        docker = (os.environ.get("MO_INGEST_DOCKER_SCORE") or "1").strip() not in {
            "0",
            "false",
            "no",
        }
        if docker and Path("/.dockerenv").exists() is False:
            header, rows = fetch_visit(visit_id)
            day = header["visit_date"]
            csv_path, _n = merge_visit_csv(rows, data_root, day)
            container = os.environ.get("MO_WEB_CONTAINER") or "protocol-web"
            cmd = [
                "sudo",
                "docker",
                "exec",
                "-e",
                f"MO_DATA_ROOT={data_root}",
                container,
                "python",
                "/app/scripts/ingest_mo_visit_from_mis.py",
                "--score-csv",
                str(csv_path),
                "--day",
                day,
                "--data-root",
                str(data_root),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout or "docker_score_failed")[:280])
        else:
            ingest_visit(visit_id, data_root=data_root)
        case_id = mark_job_done_from_warehouse(visit_id)
        update_ingest_job(job_id, status="done", case_id=case_id)
        _log.info("mo_ingest_done visit_id=%s job_id=%s", visit_id, job_id)
        return {"job_id": job_id, "visit_id": visit_id, "status": "done", "case_id": case_id}
    except LookupError as exc:
        update_ingest_job(job_id, status="error", error=str(exc)[:280])
        _log.info("mo_ingest_error visit_id=%s job_id=%s", visit_id, job_id)
        return {"job_id": job_id, "visit_id": visit_id, "status": "error"}
    except Exception as exc:
        update_ingest_job(job_id, status="error", error=str(exc)[:280])
        _log.info("mo_ingest_error visit_id=%s job_id=%s", visit_id, job_id)
        return {"job_id": job_id, "visit_id": visit_id, "status": "error"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--sleep", type=float, default=4.0)
    args = ap.parse_args()
    _load_env(Path("/opt/protocol/.env.mis"))
    if args.loop:
        while True:
            out = _process_one()
            if out:
                print(json.dumps(out, ensure_ascii=False), flush=True)
            else:
                time.sleep(max(args.sleep, 1.0))
        return 0
    out = _process_one()
    print(json.dumps(out or {"status": "idle"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
