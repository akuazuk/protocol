# Handoff для следующего агента (другой компьютер)

> **Устарело как точка входа.** Актуальный handoff:
> [`docs/handoff/2026-08-06-afternoon-next-agent.md`](2026-08-06-afternoon-next-agent.md)
> (ICD full-document #32, doctor outliers #31, Gemini Render #30, merge runbook).

**Обновлено:** 2026-08-06 ~06:45 UTC+3 (Mac владельца уходит в сон после этого пуша).  
**Статус сессии на этом Mac:** работа по контуру МО Aug 5-6 **закрыта**; можно `git pull` и продолжать с gaps ниже.  
Репо: `akuazuk/protocol`. Прод: `https://protocol-bimy.onrender.com`.  
SSH Render: `srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com`.

## 0. Старт на другом компьютере (сделай первым)

```bash
cd ~/Cursor_Folders/Protocol   # или твой путь к clone
git fetch origin
git checkout main
git pull origin main
git log -3 --oneline
curl -s https://protocol-bimy.onrender.com/api/version | python3 -m json.tool | head
caffeinate -dims &   # если долгая сессия
```

Ожидай в git минимум:
- `01aafdcd` (#24) BUILD_VERSION UTC timestamp  
- `09468abe` (#23) расширенный handoff  
- `1218dbb2` (#22) recompute без pandas  

В `/api/version` после deploy #24 должно быть вида  
`2026-08-06-033457Z-version-utc-stamp` (или новее с суффиксом `Z`).  
Если ещё `…-r2-handoff-gaps` - подожди 1-2 мин deploy и обнови.

Активные планы: `docs/plans/README.md`.

---

## 1. Что сделано (05.08 вечер + 06.08 утро)

| PR | Что | Версия/коммит |
|--|--|--|
| #17 | Named-column merge; August warehouse repair; LLM/BI path | r15-r16 |
| #18 | Filial filter pipe + comma heuristic | r17 |
| #19 | Durable Render LLM backfill | r18 |
| #20 | Case review workspace + protocol suggest + gold CLI | r19 |
| #21 | Короткий handoff | r20 |
| #22 | `recompute_mo_days` CSV без pandas; pending 80→0 | r1 |
| #23 | Handoff с gaps/errors | r2 |
| #24 | **BUILD_VERSION = UTC до секунды** (`YYYY-MM-DD-HHMMSSZ[-slug]`) | `033457Z-…` |

### Прод / данные (снимок)
- Night LLM 01-04.08: **80/80** каждый день, action-judge ok, `night_complete=true`.
- Reports: `llm_queue_pending=0` для 01-04; avg ~84-88.
- Filial smoke `ул. Захарова, 50Д` 01-04: **1542**.
- Protocol suggest API жив (top-3).
- Dual-pane UI в проде (`case-workspace-grid`).
- `crm_review_pack` = **0** (gold рано).
- Активного grade-процесса на Render нет.

### BUILD_VERSION (важно для двух машин)
Старый `rN` давал коллизии. Теперь:

```bash
scripts/ops/bump_build_version.sh           # -> 2026-08-06-HHMMSSZ
scripts/ops/bump_build_version.sh my-slug   # -> …Z-my-slug
```

Правило: `.cursor/rules/build-version.mdc`.

---

## 2. Что НЕ сделано - бери в работу (приоритет)

### P0
1. **День 2026-08-05 (+ вчера)** отсутствует на Render disk  
   Нет `mo_2026-08-05.csv`, `kz_l1_2026-08-05_*`, `reports/2026/08/05/`.  
   Нужен daily: MIS export → L1 → publish → night LLM → recompute.
2. **Июль warehouse сломан**  
   ~11902/13591 `doctor_key` = status (`good`/`review`/…).  
   August bad=0. Repair: `scripts/repair_mo_warehouse_from_secure.py` за июльские дни с CSV.
3. **Gold** - ждать ≥50 packs с `training_use=1`, потом  
   `scripts/export_mo_review_gold.py` + `eval_mo_review_gold.py`.

### P1
4. LLM grades не overlay в `cases.jsonl` / BI avg (остаётся L1) - отдельная задача, если нужна.
5. Night queue ~80/день, не полный корпус.
6. Качество protocol suggest (audience/DDx) - план `mo-case-protocol-suggest-v1`.
7. GCE host - deferred.

---

## 3. Известные ошибки (не путать с текущим state)

| Ошибка | Статус |
|--|--|
| `INSERT SELECT *` сдвиг колонок | fixed (#17); **июль ещё broken** |
| Filial comma | fixed (#18) |
| `pandas` missing в recompute | fixed (#22); источник = secure CSV |
| Deploy убивает nohup grade | перезапуск `run_mo_render_llm_backfill.sh` |
| Старый traceback pandas в `mo_llm_august_backfill.log` | хвост от 05-го; recompute уже ок |
| Gemini geo-block с Mac | LLM на Render |

---

## 4. Git / PR / deploy

1. Не в `main` напрямую - branch + PR + squash merge.
2. `scripts/ops/bump_build_version.sh [slug]` в том же коммите.
3. Обновить активный план в `docs/plans/`.
4. CI green → `gh pr merge --squash --delete-branch`.
5. Дождаться `/api/version` = новый stamp.
6. Не коммитить `.env`, PDF из `minzdrav_protocols/`, секреты.
7. VPN: SQL MIS → `ensure-off`; сильные модели → `ensure-on`.

---

## 5. Быстрые проверки

```bash
curl -s https://protocol-bimy.onrender.com/api/version

ssh srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com \
  'wc -l /var/data/medical_exams/secure_cases/2026/08/kz_l1_*_llm_grades.jsonl;
   ls /var/data/medical_exams/secure_cases/2026/08/mo_2026-08-05.csv 2>/dev/null || echo MISSING_05;
   sqlite3 /var/data/medical_exams/warehouse/mo_analytics.sqlite \
     "select count(*), sum(training_use) from crm_review_pack"'
```

Recompute при stale pending:

```bash
ssh … 'cd /opt/render/project/src && .venv/bin/python scripts/recompute_mo_days.py \
  --data-root /var/data/medical_exams --first-date 2026-08-01 --last-date 2026-08-04 \
  --warehouse /var/data/medical_exams/warehouse/mo_analytics.sqlite'
```

---

## 6. Рекомендуемый порядок на другом компьютере

1. `git pull` + сверить version (UTC stamp).
2. Поднять **2026-08-05** и «вчера» на disk (export/publish/LLM/recompute).
3. Repair июля.
4. Smoke UI разбора + сохранить 1-2 review pack (чтобы gold пошёл).
5. Suggest improvements - позже, отдельным PR.

Этот Mac после пуша handoff уходит в sleep; дальше работай со своего clone.
