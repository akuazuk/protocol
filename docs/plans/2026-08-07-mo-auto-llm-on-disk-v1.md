# МО: автозапуск night LLM сразу после данных на диске (v1)

Дата: 2026-08-07  
Статус: active  
Связанные: `2026-08-05-mo-august-llm-bi-backfill-v1.md`,
`docs/mo-daily-pipeline.md`, handoff `2026-08-07-night-next-agent.md`.

---

## 1. Контекст

«Вчера» пустое в проде, когда Mac launchd не дожал publish/LLM или завис на
warehouse merge. Данные (secure CSV/cases) могут уже лежать на Render, а grade
не стартовал.

Требование владельца: **прогон night LLM запускать сразу, как данные закачались
на диск Render**, не ждать следующего окна launchd и не блокироваться на merge.

---

## 2. Что сделано

| Шаг | Статус |
|--|--|
| `scripts/trigger_mo_render_llm_pending.sh` - скан pending days → start runner | done |
| `publish_mo_to_render.py` вызывает trigger **после** upload secure_cases (даже если warehouse merge потом падает) | done |
| launchd `hourly`/`retry`/`main`: всегда `drain` pending LLM; режим `drain-llm` | done |
| catch_up: учитывать зависший `scoring` (>2ч без heartbeat) | done |
| Action-judge: вся очередь (`limit=0`), fallback `action_queue`, source=local | done (2026-08-07) |

На Render за 2026-08-06 вручную догнано: **155/155** judges.jsonl.

---

## 3. Ops

```bash
# вручную: старт LLM для дней на диске без полного grades
bash scripts/trigger_mo_render_llm_pending.sh --days 14
bash scripts/run_mo_daily_launchd.sh drain-llm

# publish сам триггерит LLM
python3 scripts/publish_mo_to_render.py --days 14
# отключить: MO_RENDER_LLM_AFTER_PUBLISH=0 или --no-trigger-llm
```

Отключить: `MO_RENDER_LLM_AFTER_PUBLISH=0`.

Action-judge после night grade: `MO_ACTION_JUDGE_LIMIT` (default **0** = вся
action-очередь дня из `report.json` / `action_queue`). Не ставить 20 - иначе в UI
«LLM-оценка action-очереди ещё не готова» на остальных кейсах.

---

## 4. Риски

| Риск | Митигация |
|--|--|
| Двойной старт grade | pgrep python grade на Render |
| Publish hang держит lock часами | не ждать LLM в publish >180s; trigger async nohup |
| Stub-день 2 строки (volume gate off при <3 weekday samples) | отдельный harden gate; force re-export |
