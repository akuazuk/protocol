#!/usr/bin/env python3
"""Install, inspect or uninstall launchd jobs for daily MO pipeline."""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "deploy" / "launchd"
LABELS = (
    "by.protocol.mo-daily",
    "by.protocol.mo-daily-retry",
    "by.protocol.mo-daily-hourly",
    "by.protocol.mo-daily-weekly",
)


def launch_agents() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def render_template(label: str, log_dir: Path) -> bytes:
    template = (TEMPLATE_DIR / f"{label}.plist.in").read_text(encoding="utf-8")
    rendered = (
        template.replace("__ROOT__", str(ROOT))
        .replace("__WRAPPER__", str(ROOT / "scripts" / "run_mo_daily_launchd.sh"))
        .replace("__LOG_DIR__", str(log_dir))
        .replace("__PYTHON__", sys.executable)
    )
    payload = plistlib.loads(rendered.encode("utf-8"))
    return plistlib.dumps(payload, sort_keys=False)


def _launchctl(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(("launchctl", *args), check=check, text=True, capture_output=True)


def install() -> int:
    destination = launch_agents()
    log_dir = ROOT / "data" / "medical_exams" / "logs"
    destination.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    domain = f"gui/{os.getuid()}"
    for label in LABELS:
        path = destination / f"{label}.plist"
        path.write_bytes(render_template(label, log_dir))
        _launchctl("bootout", domain, str(path))
        _launchctl("bootstrap", domain, str(path), check=True)
        _launchctl("enable", f"{domain}/{label}", check=True)
        print(f"installed {path}")
    print(
        "Важно: launchd calendar использует системный timezone; "
        "Mac должен быть Europe/Minsk для точного 06:00 daily и понедельничного weekly."
    )
    return 0


def status() -> int:
    domain = f"gui/{os.getuid()}"
    failures = 0
    for label in LABELS:
        result = _launchctl("print", f"{domain}/{label}")
        loaded = result.returncode == 0
        failures += int(not loaded)
        print(f"{label}: {'loaded' if loaded else 'not loaded'}")
    state = ROOT / "data" / "medical_exams" / "state" / "pipeline.json"
    print(f"pipeline state: {state if state.exists() else 'not created'}")
    return 0 if failures == 0 else 1


def uninstall() -> int:
    destination = launch_agents()
    domain = f"gui/{os.getuid()}"
    for label in LABELS:
        path = destination / f"{label}.plist"
        _launchctl("bootout", domain, str(path))
        path.unlink(missing_ok=True)
        print(f"removed {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "status", "uninstall"))
    args = parser.parse_args()
    return {"install": install, "status": status, "uninstall": uninstall}[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
