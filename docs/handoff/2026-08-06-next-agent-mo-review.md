# Handoff для следующего агента (МО / Protocol)

Дата обновления: **2026-08-06 ~06:20 UTC+3**  
Писали агенты на Mac владельца 2026-08-05 (вечер) и 2026-08-06 (утро).  
Репо: `akuazuk/protocol`. Прод: `https://protocol-bimy.onrender.com`.  
SSH Render: `srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com` (ключ `~/.ssh/id_ed25519`).

Перед стартом:

```bash
cd ~/Cursor_Folders/Protocol   # или твой clone path
caffeinate -dims &
git fetch origin && git checkout main && git pull origin main
curl -s https://protocol-bimy.onrender.com/api/version | python3 -m json.tool | head
```

Ожидаемая версия на момент handoff: `2026-08-06-r2-handoff-gaps` (или новее).  
Если ещё видишь `r1-recompute-no-pandas` - подожди deploy / сделай pull.

Активные планы: `docs/plans/README.md` →  
`2026-08-05-mo-case-review-workspace-v2.md`,  
`2026-08-05-mo-august-llm-bi-backfill-v1.md`,  
`2026-08-05-mo-case-protocol-suggest-v1.md`.

---

## A. Что сделано вчера (2026-08-05)

| PR | Суть | Версия |
|--|--|--|
| #17 | Починка `merge_sql` (named columns); repair August warehouse; путь LLM/BI backfill | r15-r16 |
| #18 | Фильтр филиала: `|` + heuristic для адресов вида `ул. Захарова, 50Д` | r17 |
| #19 | Durable LLM backfill на Render (`scripts/run_mo_render_llm_backfill.sh`) | r18 |
| #20 | Workspace «Разбор случая»: dual-pane, без `%`, RU, таблица дня, protocol suggest + ratings, gold CLI | r19 |
| #21 | Короткий handoff + статусы плана | r20 |

### UI / продукт (#20)
- Drawer: `case-workspace-grid` - слева клиническое МО (свой scroll), справа решение.
- Убраны поля «Полнота % / Диагноз % / Рек. %»; textarea «Развёрнутый разбор» rows=14 / 12000.
- Prev/next, phrase chips, RU-лейблы.
- `GET /api/methodist/mo/cases/{id}/protocol-suggest` + оценка релевантности в `crm_review_pack`.
- `scripts/export_mo_review_gold.py`, `scripts/eval_mo_review_gold.py`.

### Данные / LLM (вечер 05-го)
- Night grades 01-04 августа догнаны до **80/день** (не полный корпус дня ~150-500).
- Action-judge за 01-04 выполнен.
- Первый финальный `recompute` в ночном supervisor **упал** (см. ошибки ниже) - починено утром 06-го.

### Корневой баг BI (починен для августа)
`merge_sql` делал `INSERT … SELECT *` → сдвиг колонок → `doctor_key`/`specialty`/`filial` битые.  
August 1-4 repaired. **Июль всё ещё битый** (см. §C).

---

## B. Что сделано сегодня (2026-08-06)

| PR | Суть | Версия |
|--|--|--|
| #22 | `recompute_mo_days.py` без pandas: читает `secure_cases/.../mo_YYYY-MM-DD.csv` | r1 |
| этот commit | Расширенный handoff: gaps / errors / next steps | r2 |

### Recompute
На Render **нет** `raw/*.parquet` и **нет** `pandas` в web venv.  
Источник среза: `/var/data/medical_exams/secure_cases/2026/08/mo_2026-08-0N.csv`.

Прогон выполнен вручную + скрипт в main:

| День | grades | llm_queue_pending | avg_score |
|--|--|--|--|
| 08-01 | 80 | **0** (было 80) | 88.4 |
| 08-02 | 80 | **0** | 88.1 |
| 08-03 | 80 | **0** | 84.7 |
| 08-04 | 80 | **0** | 86.1 |

Smoke API (с `METHODIST_TOKEN` из локального `.env`, не коммитить):
- Filial `ул. Захарова, 50Д` за 01-04: **total=1542**
- Protocol suggest: 200, top-3 items
- HTML: нет «Полнота %»; есть protocol-suggest / Развёрнутый разбор / dual-pane CSS
- LLM coverage: 4 дня `night_complete=true`, grades_ok=320, action_judges=28

Gold: packs=0 → export не запускали (только маркер  
`/var/data/medical_exams/gold_review/2026-08-06-status/STATUS.json`).

---

## C. Что НЕ сделано / чего не хватает (приоритет)

### P0 - данные
1. **Июль warehouse всё ещё сломан колонками**  
   Снимок 2026-08-06: `fact_mo_case` за июль ~13591, из них **~11902** с `doctor_key` вида status (`good`/`review`/`critical`/`ok`).  
   August: bad=0.  
   Нужен repair по образцу `scripts/repair_mo_warehouse_from_secure.py` для июля (secure CSV есть за 07-26..31).
2. **День 2026-08-05 на Render disk отсутствует**  
   Нет `mo_2026-08-05.csv`, нет `kz_l1_2026-08-05_*`, нет `reports/2026/08/05/`.  
   Continuous daily pipeline / publish за 05-е **не отработал** (или артефакты не залиты).  
   Нужно: выгрузка MIS → L1 → publish → night LLM → recompute для 05 (и дальше «вчера»).
3. **Gold / обучение**  
   `crm_review_pack` = **0**. Первые packs появятся только когда методист/эксперт сохранит разборы с `training_use=1`.  
   Порог для первого export: **≥50**. Пока нечего экспортировать.

### P1 - продукт / качество
4. **LLM grades не влиты в `cases.jsonl`**  
   `llm_used` у cases = 0; `overall_pct` в grades часто 0 при verdict critical - это отдельный grader-score, не замена L1.  
   BI avg_score остаётся от L1/deep. Night grades живут в `*_llm_grades.jsonl` + UI coverage.  
   Если нужна подмена/overlay score в витрине - это отдельная задача (сейчас не сделано).
5. **Night queue ≠ полный день**  
   ~80 кейсов/день + action-judge ≤20. Полный LLM на все КЗ дня (~1500) сознательно вне scope.
6. **Protocol suggest качество**  
   MVP поверх `match_protocol_cards`; бывают слабые матчи (например specialty ≠ протокол).  
   Улучшения audience/DDx - план `mo-case-protocol-suggest-v1`.
7. **Метрики gold «было/стало»** в плане workspace v2 - чекбокс открыт (нужны packs).
8. **GCE europe-north1 host** - deferred (review-pack / runtime планы).

### P2 - ops
9. После **каждого deploy** web-сервиса процессы grade на том же инстансе умирают - перезапускать supervisor если mid-run.
10. В старом логе `mo_llm_august_backfill.log` хвост всё ещё показывает traceback pandas от 05-го вечера - **не регрессия**: позже скрипт починен и recompute прошёл. Не путать с текущим состоянием.
11. Локально на Mac часто куча untracked `minzdrav_protocols/*.pdf` и experiments - **не коммитить**.

---

## D. Где были / есть ошибки

| Ошибка | Где | Статус |
|--|--|--|
| `INSERT … SELECT *` сдвиг колонок | `clinical_knowledge/mo_publish.py` merge | **fixed** named columns (#17); August repaired |
| Тот же сдвиг на **июле** | warehouse `fact_mo_case` | **OPEN** ~88% bad doctor_key |
| Filial filter резал адреса с запятой | facets / parse | **fixed** (#18) |
| `ModuleNotFoundError: pandas` в `recompute_mo_days` | Render venv + путь `raw/*.parquet` | **fixed** (#22), CSV path |
| Deploy убивает nohup grade | Render single instance | **mitigated**: durable script + ручной restart |
| Gemini geo-block с Mac | LLM | работать на Render |
| `crm_review_pack` пуст | нет human saves | **ожидание** методиста |
| Нет артефактов 2026-08-05 | daily pipeline/publish | **OPEN** |
| Ложный `ps \| grep grade` | SSH one-liner матчит сам себя | искать `.venv/bin/python scripts/grade_kz` |

---

## E. Как коммитить / мержить / деплоить (обязательно)

1. **Не пушить в `main` напрямую** - branch protection. Всегда `cursor/<тема>` + PR.
2. Перед осмысленным коммитом:
   - bump `BUILD_VERSION` в `rag_server.py` → `YYYY-MM-DD-rN-kebab` (N++ в рамках даты);
   - обновить активный план в `docs/plans/`;
   - UI-тексты: `python3 scripts/normalize_ui_dashes.py` **только на нужные файлы**.
3. Commit через HEREDOC, без `--no-verify`, без amend чужих коммитов.
4. `git push -u origin HEAD` → дождаться CI → `gh pr merge --squash --delete-branch`.
5. Дождаться `curl …/api/version` = новая версия (workflow `Production Render release`).
6. Если шёл LLM backfill - проверить процесс и при необходимости:
   `bash scripts/run_mo_render_llm_backfill.sh YYYY-MM-DD YYYY-MM-DD`
7. VPN «Дядя Ваня»: SQL MIS → `ensure-off`; сильные модели → `ensure-on`.
8. Секреты / `.env` / сырые ПДн / случайные PDF - не в git.

Шаблон:

```bash
git checkout main && git pull origin main
git checkout -b cursor/<короткая-тема>
# правки + BUILD_VERSION
git add <нужные файлы>
git commit -m "$(cat <<'EOF'
MO: краткое why.

EOF
)"
git push -u origin HEAD
gh pr create --title "…" --body "…"
gh pr checks
gh pr merge --squash --delete-branch
git checkout main && git pull origin main
```

---

## F. Команды проверки (утро)

```bash
# версия
curl -s https://protocol-bimy.onrender.com/api/version

# grades + pending
ssh srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com \
  'wc -l /var/data/medical_exams/secure_cases/2026/08/kz_l1_*_llm_grades.jsonl;
   for d in 01 02 03 04; do
     python3 -c "import json; r=json.load(open(\"/var/data/medical_exams/reports/2026/08/$d/report.json\")); print(\"$d\", r[\"completeness\"].get(\"llm_queue_pending\"), r[\"summary\"].get(\"avg_score\"))"
   done'

# июль broken doctor_key
ssh … 'cd /opt/render/project/src && .venv/bin/python - <<"PY"
import sqlite3
c=sqlite3.connect("/var/data/medical_exams/warehouse/mo_analytics.sqlite")
print(c.execute("""select count(*) from fact_mo_case
  where visit_date like \"2026-07%\"
  and doctor_key in (\"good\",\"review\",\"critical\",\"ok\")""").fetchone())
PY'

# packs for gold
ssh … 'sqlite3 /var/data/medical_exams/warehouse/mo_analytics.sqlite \
  "select count(*), sum(training_use) from crm_review_pack"'

# recompute (если снова stale pending)
ssh … 'cd /opt/render/project/src && .venv/bin/python scripts/recompute_mo_days.py \
  --data-root /var/data/medical_exams --first-date 2026-08-01 --last-date 2026-08-04 \
  --warehouse /var/data/medical_exams/warehouse/mo_analytics.sqlite'
```

---

## G. Рекомендуемый порядок работ следующему агенту

1. `git pull` main, сверить `/api/version`.
2. Поднять **2026-08-05** (и вчерашний день): MIS export → L1 → publish на Render → night LLM → recompute.
3. Repair **июльского** warehouse (`repair_mo_warehouse_from_secure.py --apply` или аналог).
4. UI smoke разбора случая глазами (dual-pane, suggest, save pack) - получить первые `crm_review_pack`.
5. Когда `training_use >= 50` - `export_mo_review_gold.py` + `eval_mo_review_gold.py`.
6. Улучшения suggest (audience/DDx) по отдельному плану - не смешивать с L1 scorer.

---

## H. Definition of Done для «августовского контура» (уже почти)

- [x] Named-column merge + August doctor/filial читаемы  
- [x] Night LLM 01-04 + action-judge  
- [x] Recompute без pandas, pending=0  
- [x] Case review workspace + protocol suggest в проде  
- [ ] День 05+ автоматически на диске  
- [ ] Июль warehouse repaired  
- [ ] ≥50 training packs + первый gold export  

Не трогай без нужды: CRM таблицы при repair; не force-push main; не ставь pandas в web image «на всякий» если CSV-путь работает.
