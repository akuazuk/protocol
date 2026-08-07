#!/usr/bin/env bash
# Найти на Render disk дни с cases+queue без полного night LLM и сразу запустить прогон.
# Вызывается после publish (данные уже на диске) и из launchd drain-llm / hourly.
# Пример:
#   bash scripts/trigger_mo_render_llm_pending.sh
#   bash scripts/trigger_mo_render_llm_pending.sh --days 7
#   bash scripts/trigger_mo_render_llm_pending.sh --dates 2026-08-06
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_HOST="${RENDER_SSH_HOST:-srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com}"
SSH_ID="${RENDER_SSH_IDENTITY:-$HOME/.ssh/id_ed25519}"
# По умолчанию только свежие дни (не утаскивать июльский catch-up в auto).
DAYS="${MO_RENDER_LLM_PENDING_DAYS:-5}"
DATES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --days) DAYS="${2:?}"; shift 2 ;;
    --dates) DATES="${2:?}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [[ -x "$HOME/CURSOR/bin/vanya_vpn.sh" ]]; then
  "$HOME/CURSOR/bin/vanya_vpn.sh" ensure-off >/dev/null 2>&1 || true
fi

# Актуальный runner с Mac (без nested heredoc).
if [[ -f "$ROOT/scripts/mo_llm_range_runner.sh" ]]; then
  scp -o BatchMode=yes -o ConnectTimeout=25 -i "$SSH_ID" \
    "$ROOT/scripts/mo_llm_range_runner.sh" \
    "$ROOT/scripts/grade_kz_llm.py" \
    "$ROOT/scripts/run_mo_action_queue_llm_judge.py" \
    "$ROOT/scripts/recompute_mo_days.py" \
    "$SSH_HOST:/opt/render/project/src/scripts/" >/dev/null
fi

REMOTE_OUT="$(
  ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -i "$SSH_ID" "$SSH_HOST" \
    "DAYS='$DAYS' DATES='$DATES' python3 -" <<'PY'
import json, os, subprocess
from datetime import date, timedelta
from pathlib import Path

root = Path("/var/data/medical_exams/secure_cases")
days_back = int(os.environ.get("DAYS") or "14")
forced = [d.strip() for d in (os.environ.get("DATES") or "").split(",") if d.strip()]
today = date.today()
candidates: list[str] = []
if forced:
    candidates = forced
else:
    for i in range(days_back + 1):
        candidates.append((today - timedelta(days=i)).isoformat())

pending: list[str] = []
for day in candidates:
    y, m, _ = day.split("-")
    cases = root / y / m / f"kz_l1_{day}_cases.jsonl"
    queue = root / y / m / f"kz_l1_{day}_llm_queue.json"
    grades = root / y / m / f"kz_l1_{day}_llm_grades.jsonl"
    if not cases.is_file() or cases.stat().st_size < 32:
        continue
    if not queue.is_file():
        continue
    try:
        q = json.loads(queue.read_text(encoding="utf-8"))
    except Exception:
        continue
    need = int(q.get("n") or 0)
    if need <= 0 and isinstance(q.get("visit_ids"), list):
        need = len(q["visit_ids"])
    if need <= 0:
        continue
    ok = err = 0
    if grades.is_file():
        for line in grades.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                err += 1
                continue
            if row.get("error"):
                err += 1
            else:
                ok += 1
    if ok < need:
        pending.append(day)

print("PENDING=" + ",".join(pending))
if not pending:
    raise SystemExit(0)

# Уже крутится python grade - не дублируем.
running = subprocess.run(
    ["pgrep", "-f", r"[.]venv/bin/python .*grade_kz_llm[.]py"],
    capture_output=True,
    text=True,
)
if running.returncode == 0 and running.stdout.strip():
    print("ALREADY_RUNNING")
    print(running.stdout.strip())
    raise SystemExit(0)

runner = Path("/opt/render/project/src/scripts/mo_llm_range_runner.sh")
if not runner.is_file():
    print("NO_RUNNER")
    raise SystemExit(2)

first, last = pending[0], pending[-1]
# Диапазон только если дни подряд; иначе по одному с конца (свежие первыми).
def contiguous(days: list[str]) -> bool:
    if len(days) <= 1:
        return True
    cur = date.fromisoformat(days[0])
    for item in days[1:]:
        nxt = date.fromisoformat(item)
        if nxt != cur + timedelta(days=1):
            return False
        cur = nxt
    return True

ordered = sorted(pending)
Path("/var/data/medical_exams/logs").mkdir(parents=True, exist_ok=True)
if contiguous(ordered):
    first, last = ordered[0], ordered[-1]
    log = f"/var/data/medical_exams/logs/mo_llm_backfill_{first}_{last}.nohup"
    subprocess.Popen(
        ["bash", str(runner)],
        env={**os.environ, "FIRST": first, "LAST": last},
        stdout=open(log, "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"STARTED first={first} last={last} log={log}")
else:
    for day in reversed(ordered):
        log = f"/var/data/medical_exams/logs/mo_llm_backfill_{day}_{day}.nohup"
        subprocess.Popen(
            ["bash", str(runner)],
            env={**os.environ, "FIRST": day, "LAST": day},
            stdout=open(log, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"STARTED first={day} last={day} log={log}")
        break  # один диапазон за вызов; остальные подхватит следующий hourly
PY
)"

echo "$REMOTE_OUT"
if echo "$REMOTE_OUT" | grep -q '^STARTED'; then
  exit 0
fi
if echo "$REMOTE_OUT" | grep -q 'PENDING=$'; then
  echo "МО: на Render нет дней, ждущих night LLM"
  exit 0
fi
exit 0
