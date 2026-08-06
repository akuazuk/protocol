# Handoff для агента на 2026-08-06 (МО review workspace + LLM backfill)

Писал агент вечером **2026-08-05**, утром **2026-08-06** добил recompute.

## 1. Что в проде

| PR | Тема | Версия |
|--|--|--|
| #17 | Named-column warehouse merge; August repair + LLM/BI path | r15-r16 |
| #18 | Filial filter: `|` + comma heuristic (`ул. Захарова, 50Д`) | r17 |
| #19 | Durable Render LLM backfill | r18 |
| #20 | Case review workspace + protocol suggest + gold export | r19 |
| #21 | Handoff / plan status | r20 |
| **recompute PR** | `recompute_mo_days` без pandas, CSV из secure_cases | **r1 2026-08-06** |

```text
curl -s https://protocol-bimy.onrender.com/api/version
```

План workspace: `docs/plans/2026-08-05-mo-case-review-workspace-v2.md`  
План backfill: `docs/plans/2026-08-05-mo-august-llm-bi-backfill-v1.md` (P2 закрыт).

## 2. LLM August 01-04 - DONE

| День | Grades | Action-judge | llm_queue_pending после recompute |
|--|--|--|--|
| 08-01 | 80 | ok | **0** (было 80) |
| 08-02 | 80 | ok | **0** |
| 08-03 | 80 | ok | **0** |
| 08-04 | 80 | ok | **0** |

Лог: `/var/data/medical_exams/logs/mo_llm_august_backfill.log`  
Активного grade сейчас нет. После deploy проверять процессы не обязательно, пока не стартуют новые дни.

Корневая проблема recompute: на Render **нет** `raw/*.parquet` и **нет pandas** в venv.
Источник - `secure_cases/YYYY/MM/mo_YYYY-MM-DD.csv`. Скрипт `scripts/recompute_mo_days.py`
теперь читает CSV без pandas.

Повтор:
```bash
ssh … 'cd /opt/render/project/src && .venv/bin/python scripts/recompute_mo_days.py \
  --data-root /var/data/medical_exams --first-date 2026-08-01 --last-date 2026-08-04 \
  --warehouse /var/data/medical_exams/warehouse/mo_analytics.sqlite'
```

## 3. Как коммитить / мержить / деплоить

1. Не в `main` напрямую - feature branch + PR.
2. Bump `BUILD_VERSION` (`YYYY-MM-DD-rN-kebab`).
3. Обновить активный план в `docs/plans/`.
4. `git push -u origin HEAD` → CI green → `gh pr merge --squash --delete-branch`.
5. Дождаться `/api/version`. Если mid-LLM - перезапустить supervisor.
6. Не коммитить `.env`, PDF из `minzdrav_protocols/`, секреты.

## 4. Smoke checklist

1. Dual-pane: CSS `mo-case-panes` / `mo-pane-clinical` / `mo-pane-decision`.
2. Нет полей «Полнота % / Диагноз % / Рек. %» в HTML.
3. Блок protocol-suggest + API `GET .../cases/{id}/protocol-suggest`.
4. Filial `ул. Захарова, 50Д` за 01-04.08 даёт большой total (~700).
5. Reports: `llm_queue_pending=0` за 01-04.

## 5. Gold export - ещё рано

На 2026-08-06: `crm_review_pack` = **0** строк, `training_use=0`.
Первый export только при ≥50 packs с `training_use=1`:

```bash
ssh … '.venv/bin/python scripts/export_mo_review_gold.py \
  --warehouse /var/data/medical_exams/warehouse/mo_analytics.sqlite \
  --out /var/data/medical_exams/gold_review/YYYY-MM-DD'
```

## 6. Быстрый старт

```bash
cd ~/Cursor_Folders/Protocol
git fetch origin && git checkout main && git pull origin main
curl -s https://protocol-bimy.onrender.com/api/version | python3 -m json.tool | head
ssh srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com \
  'wc -l /var/data/medical_exams/secure_cases/2026/08/kz_l1_2026-08-0*_llm_grades.jsonl; \
   python3 -c "import json;from pathlib import Path;\
[print(d, json.loads(Path(f\"/var/data/medical_exams/reports/2026/08/0{d}/report.json\").read_text())[\"completeness\"].get(\"llm_queue_pending\")) for d in range(1,5)]"'
```

SSH: `srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com`. Repo: `akuazuk/protocol`.
