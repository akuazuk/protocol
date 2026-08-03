#!/usr/bin/env python3
"""Telegram: вопросы с кнопками Да/Нет и исполнение ответа на машине.

  python3 scripts/telegram_control.py loop          # фон: слушать ответы
  python3 scripts/telegram_control.py ask --id git_push --text "..." --yes git_push --no skip
  python3 scripts/telegram_control.py status
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from env_load import load_project_env

load_project_env(ROOT)

import telegram_bot_api as tg_api
from telegram_notify import telegram_enabled

STORE = ROOT / "data/ml/reports/telegram_control_state.json"
LOG = ROOT / "data/ml/reports/telegram_control.log"
PY = ROOT / ".venv/bin/python"

YES_WORDS = frozenset(
    {"да", "yes", "y", "ok", "пуш", "push", "1", "+", "делай", "го", "continue", "продолжай"}
)
NO_WORDS = frozenset(
    {"нет", "no", "n", "skip", "стоп", "stop", "0", "-", "не", "отмена", "pause", "пауза"}
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{_now()}] {msg}\n")


def _load() -> dict[str, Any]:
    if not STORE.is_file():
        return {"offset": 0, "pending": {}, "history": []}
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        return {"offset": 0, "pending": {}, "history": []}


def _save(st: dict[str, Any]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _authorized_chat(chat: dict[str, Any]) -> bool:
    want = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    return str(chat.get("id") or "") == want


def _run_shell(cmd: str, *, background: bool = False) -> tuple[int, str]:
    _log(f"exec: {cmd}")
    if background:
        subprocess.Popen(
            cmd,
            shell=True,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return 0, "started in background"
    p = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out.strip()[:1500]


def execute_action(action: str, *, context: str = "") -> str:
    action = (action or "").strip().lower()
    if action in ("", "skip", "noop", "git_push_skip", "redeploy_skip", "smoke_skip", "embed_skip"):
        return "Принято: без действий."

    if action == "git_push":
        rc, out = _run_shell("git push origin HEAD")
        return ("Push выполнен." if rc == 0 else f"Push ошибка ({rc}): {out[:400]}")

    if action == "embed_stop":
        _run_shell("pkill -f 'scripts/build_chunk_embeddings.py' || true")
        return "Embeddings остановлены."

    if action == "embed_start":
        log_path = ROOT / "data/ml/reports/embed_checklist_run.log"
        _run_shell(
            f"nohup {PY} scripts/build_chunk_embeddings.py >> {log_path} 2>&1 &",
            background=True,
        )
        return "Embeddings запущены в фоне."

    if action == "smoke_l2":
        out_dir = ROOT / "ml/experiments/batch_smoke_telegram"
        rc, out = _run_shell(
            f"{PY} scripts/run_clients_consult_render_batch.py "
            f"--tier L2 --cases gastro_1,kard_1,pediatr_1,report_lor_1,report_urolog_1 "
            f"--out {out_dir}/l2_sample"
        )
        m = re.search(r"avg overall=([0-9.]+%)", out)
        extra = f" Средний балл: {m.group(1)}." if m else ""
        return ("L2 smoke готов." + extra if rc == 0 else f"Smoke ошибка ({rc}): {out[:400]}")

    if action == "status":
        rc, out = _run_shell(f"{PY} scripts/telegram_control.py status --brief")
        return out if rc == 0 else "Не удалось получить статус."

    return f"Неизвестное действие: {action}"


def resolve_pending(st: dict[str, Any], decision_id: str, choice: str, *, via: str) -> str | None:
    pending = st.get("pending") or {}
    item = pending.get(decision_id)
    if not item or item.get("resolved"):
        return None
    yes_action = item.get("yes_action", "noop")
    no_action = item.get("no_action", "noop")
    action = yes_action if choice == "yes" else no_action
    item["resolved"] = _now()
    item["choice"] = choice
    item["via"] = via
    item["action_run"] = action
    pending[decision_id] = item
    st["pending"] = pending
    hist = list(st.get("history") or [])
    hist.append({"id": decision_id, "choice": choice, "action": action, "ts": _now()})
    st["history"] = hist[-50:]
    _save(st)
    return execute_action(action, context=decision_id)


def ask_decision(
    decision_id: str,
    text: str,
    *,
    yes_action: str,
    no_action: str = "noop",
    yes_label: str = "Да",
    no_label: str = "Нет",
    force: bool = False,
) -> bool:
    if not telegram_enabled():
        return False
    st = _load()
    pending = st.get("pending") or {}
    old = pending.get(decision_id)
    if old and not old.get("resolved") and not force:
        return True

    cb_yes = f"d:{decision_id}:yes"[:64]
    cb_no = f"d:{decision_id}:no"[:64]
    markup = tg_api.inline_keyboard([[(yes_label, cb_yes), (no_label, cb_no)]])
    try:
        resp = tg_api.send_message(text, reply_markup=markup)
    except Exception as e:
        _log(f"ask failed {decision_id}: {e}")
        return False
    msg = (resp.get("result") or {})
    pending[decision_id] = {
        "message_id": msg.get("message_id"),
        "text": text[:500],
        "yes_action": yes_action,
        "no_action": no_action,
        "created": _now(),
        "resolved": None,
    }
    st["pending"] = pending
    _save(st)
    _log(f"ask {decision_id}: {text[:120]}")
    return True


def _match_text_reply(text: str) -> str | None:
    t = (text or "").strip().lower()
    if t in YES_WORDS:
        return "yes"
    if t in NO_WORDS:
        return "no"
    return None


def _latest_open_decision(st: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    pending = st.get("pending") or {}
    open_items = [(k, v) for k, v in pending.items() if not v.get("resolved")]
    if not open_items:
        return None
    open_items.sort(key=lambda kv: kv[1].get("created") or "", reverse=True)
    return open_items[0]


def handle_update(st: dict[str, Any], upd: dict[str, Any]) -> None:
    cb = upd.get("callback_query")
    if cb:
        msg = cb.get("message") or {}
        chat = msg.get("chat") or {}
        if not _authorized_chat(chat):
            tg_api.answer_callback(str(cb.get("id") or ""), "Чужой chat")
            return
        data = str(cb.get("data") or "")
        m = re.fullmatch(r"d:([a-zA-Z0-9_\-]+):(yes|no)", data)
        if not m:
            tg_api.answer_callback(str(cb.get("id") or ""), "Неизвестная кнопка")
            return
        decision_id, choice = m.group(1), m.group(2)
        result = resolve_pending(st, decision_id, choice, via="button")
        tg_api.answer_callback(str(cb.get("id") or ""), "Готово")
        if result:
            tg_api.send_message(f"Protocol [{decision_id}]: {result}")
            _log(f"resolved {decision_id} {choice} -> {result[:200]}")
        return

    msg = upd.get("message")
    if not msg:
        return
    chat = msg.get("chat") or {}
    if not _authorized_chat(chat):
        return
    text = str(msg.get("text") or "")
    if text.startswith("/"):
        cmd = text.split()[0].lower()
        if cmd in ("/status", "/статус"):
            tg_api.send_message(status_text())
            return
        if cmd in ("/help", "/помощь"):
            tg_api.send_message(
                "Ответы: Да/Нет или push/skip.\n"
                "Кнопки под вопросом - предпочтительно.\n"
                "/status - краткий статус pipeline."
            )
            return
    choice = _match_text_reply(text)
    if not choice:
        return
    reply_to = msg.get("reply_to_message")
    decision_id = None
    if reply_to:
        rid = reply_to.get("message_id")
        for did, item in (st.get("pending") or {}).items():
            if item.get("message_id") == rid and not item.get("resolved"):
                decision_id = did
                break
    if not decision_id:
        latest = _latest_open_decision(st)
        if latest:
            decision_id = latest[0]
    if not decision_id:
        tg_api.send_message("Нет открытого вопроса. Ждите уведомление с кнопками.")
        return
    result = resolve_pending(st, decision_id, choice, via="text")
    if result:
        tg_api.send_message(f"Protocol [{decision_id}]: {result}")
        _log(f"text {decision_id} {choice} -> {result[:200]}")


def status_text(*, brief: bool = False) -> str:
    lines: list[str] = []
    try:
        import re

        bv = ""
        t = (ROOT / "rag_server.py").read_text(encoding="utf-8")
        m = re.search(r'BUILD_VERSION = "([^"]+)"', t)
        if m:
            bv = m.group(1)
        lines.append(f"BUILD: {bv}")
    except Exception:
        pass
    try:
        import urllib.request

        with urllib.request.urlopen(
            os.environ.get("PROTOCOL_PROD_URL", "https://protocol-bimy.onrender.com") + "/api/version",
            timeout=12,
        ) as r:
            prod = json.loads(r.read().decode()).get("version", "?")
        lines.append(f"prod: {prod}")
    except Exception:
        lines.append("prod: недоступен")
    try:
        p = subprocess.run(
            ["git", "rev-list", "--count", "origin/main..HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        lines.append(f"git ahead: {(p.stdout or '').strip() or '0'}")
    except Exception:
        pass
    embed = subprocess.run(["pgrep", "-f", "scripts/build_chunk_embeddings.py"], capture_output=True)
    lines.append("embed: " + ("идёт" if embed.returncode == 0 else "стоп"))
    st = _load()
    open_n = sum(1 for v in (st.get("pending") or {}).values() if not v.get("resolved"))
    lines.append(f"открытых вопросов: {open_n}")
    return "\n".join(lines)


def poll_loop() -> int:
    if not telegram_enabled():
        print("Telegram disabled", file=sys.stderr)
        return 1
    _log("control loop start")
    st = _load()
    if not st.get("loop_announced"):
        tg_api.send_message("Protocol control: слушаю ответы (кнопки и Да/Нет). /status - статус.")
        st["loop_announced"] = _now()
        _save(st)
    offset = int(st.get("offset") or 0)
    while True:
        try:
            updates = tg_api.get_updates(offset=offset, timeout=25)
            for upd in updates:
                offset = max(offset, int(upd.get("update_id") or 0) + 1)
                handle_update(st, upd)
                st = _load()
            st["offset"] = offset
            _save(st)
        except KeyboardInterrupt:
            _log("control loop stop")
            return 0
        except Exception as e:
            _log(f"poll error: {e}")
            time.sleep(5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram control loop")
    sub = parser.add_subparsers(dest="cmd")

    p_loop = sub.add_parser("loop", help="Listen for replies")
    p_loop.add_argument("--once", action="store_true", help="One poll iteration")

    p_ask = sub.add_parser("ask", help="Send decision question")
    p_ask.add_argument("--id", required=True)
    p_ask.add_argument("--text", required=True)
    p_ask.add_argument("--yes", required=True, dest="yes_action")
    p_ask.add_argument("--no", default="noop", dest="no_action")
    p_ask.add_argument("--yes-label", default="Да")
    p_ask.add_argument("--no-label", default="Нет")
    p_ask.add_argument("--force", action="store_true")

    p_status = sub.add_parser("status")
    p_status.add_argument("--brief", action="store_true")

    args = parser.parse_args()
    if args.cmd == "loop":
        if args.once:
            st = _load()
            offset = int(st.get("offset") or 0)
            for upd in tg_api.get_updates(offset=offset, timeout=0):
                offset = max(offset, int(upd.get("update_id") or 0) + 1)
                handle_update(st, upd)
                st = _load()
            st["offset"] = offset
            _save(st)
            return 0
        return poll_loop()
    if args.cmd == "ask":
        ok = ask_decision(
            args.id,
            args.text,
            yes_action=args.yes_action,
            no_action=args.no_action,
            yes_label=args.yes_label,
            no_label=args.no_label,
            force=args.force,
        )
        return 0 if ok else 2
    if args.cmd == "status":
        print(status_text(brief=args.brief))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
