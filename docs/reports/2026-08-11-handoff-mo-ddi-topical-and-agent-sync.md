# Handoff: топический DDI + синхрон двух агентов (2026-08-11)

Дата: 2026-08-11 (вечер, Europe/Minsk)  
Владелец сессии: Cursor agent (pc1)  
Следующий агент: начать с этого файла + `AGENTS.md` + `git fetch origin`.

---

## Repo / prod (канон на момент handoff)

| | |
|--|--|
| `origin/main` | `298212ac` - `fix(mo): ignore P2/P3 safety findings in zone attention band (#139)` |
| Prod UI | **GCE** `https://protocol.kravira.by` (не Render как primary) |
| `BUILD_VERSION` | `2026-08-11-191552Z-safety-zone-p2` (`rag_ready=true`) |
| План DDI | `docs/plans/2026-08-11-mo-ddi-topical-demote-v1.md` (код в main; хвост - backfill данных) |
| Грязный Mac checkout | `~/Cursor_Folders/Protocol` часто behind + грязный - **не чинить pull/rebase**; новый worktree |

Preflight каждой сессии: блок из `AGENTS.md` (`status` / `fetch` / `left-right` / `gh pr list`).

---

## Что сделано сегодня (не переделывать)

### 1. Case-detail latency (раньше в той же дуге)

- PR [#134](https://github.com/akuazuk/protocol/pull/134): defer live concordance/prior, prewarm protocol suggest.
- Уже в `main`, задеплоено на GCE.

### 2. Топический Major DDI не в «Критично»

- PR [#138](https://github.com/akuazuk/protocol/pull/138) → `66cc93ed`
- `BUILD_VERSION` тогда: `2026-08-11-183912Z-ddi-topical-demote`
- Логика:
  - `clinical_knowledge/medication_safety.py` - `drug_mention_is_topical`, `ddi_pair_has_topical_partner`, `finding_suggests_topical_ddi`
  - `kz_deep_eval.py` - Major + topical partner → **P2**, title `(Major, топический путь - понижено)`, флаги `topical_ddi` / `ddi_level_effective`
  - `mo_action_queue_select.py` - topical DDI **вне** action-очереди; `ddi_is_major` = false для topical
- Системный НПВП + антикоагулянт **остаётся** Major / очередь Критично.
- Тесты: `tests/test_mo_action_queue_select.py`, `tests/test_nsaid_alternatives_topical.py`

### 3. Зона safety: P2 не даёт «важный сигнал риска»

- PR [#139](https://github.com/akuazuk/protocol/pull/139) → `298212ac`
- `mo_zone_scores._safety_from_findings`: только **P0/P1** поднимают `safety.band=important`
- Тест: `test_safety_ignores_p2_ddi` в `tests/test_mo_zone_scores.py`

### 4. Данные / кейсы

| Visit | Дата | Итог |
|-------|------|------|
| **3665385** | 2026-08-10 | Ксарелто + диклофенак **гель**. После single-visit deep + снятие stale `evaluation_v4` + `recompute_day`: overall **87.7**, status **good**, DDI **P2** topical, attention **none**, safety **none**. |
| **3651370** | 2026-08-10 | Невролог: **сертралин + амитриптилин** в плане. **Легитимный** Major DDI → P1 → очередь **Критично**. Demote не применять. Status записи `review`/60 - это нормально; «Критично» = полоса очереди (`major_lifts_to`). |

Важно при патче одного визита: в warehouse **`evaluation_v4` перекрывает `deep`**. Если правите только `deep` в `cases.jsonl`, старый `evaluation_v4` оставит старые findings - его нужно убрать/обновить, затем `recompute_day`.

### 5. Deploy GCE

```bash
GCE_OPS_USER=pavelkuzauka SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
```

Default `GCE_OPS_USER=pavel` даёт **Permission denied** на `/opt/protocol/.env.gcp-public` (mode 600).  
Gemini / night LLM / `grade_kz_llm` - **только** `deploy/gcp-llm/run_on_gce.sh`, не с Mac.

---

## Как считается «Критично» от Major DDI (контекст для завтра)

Две разные шкалы:

1. **Finding severity** P0/P1/P2 - таксономия замечания (P1 Major DDI = «Важно» у finding).
2. **Action queue band** `critical` / `important` - ярлык в очереди методиста (**«Критично»** / **«Важно»**).
3. **Status записи** `good` / `review` / `critical` - пороги overall после штрафов.

Цепочка DDI:

1. Препараты из плана → пары в **DDInter**.
2. Major → `C_ddi` P1; Moderate → P2 (обычно не в очереди).
3. Topical Major → P2 + вне очереди (фикс #138).
4. В `mo_action_queue_select`: `C_ddi` + Major → `major_lifts_to = critical` → UI **«Критично»**, даже если status записи ещё `review`.

Не путать с wire-status `critical` у всей записи.

---

## Что сделать завтра (приоритет)

### P0 - синхрон двух агентов (обязательно прочитать)

См. раздел ниже «Синхрон и push». Без этого будут гонки по одним файлам.

### P1 - backfill уже сохранённых DDI (по запросу владельца)

Код в проде действует на **новые** deep-eval. Старые дни с топическим Major в warehouse останутся P1, пока не:

- deep-rescore нужных дат (осторожно: полный день тяжёлый; для точечных визитов - single-visit как 3665385), затем
- `recompute_mo_days` / `recompute_day`,
- проверка, что нет stale `evaluation_v4`, перекрывающего новый `deep`.

Не гонять полный август без явного OK владельца.

### P2 - sanity очереди «Критично»

Выборочно открыть 5-10 кейсов с `C_ddi` Major в очереди за 2026-08-10 (или вчера):

- топический гель/мазь/местно - уже не должен быть Критично (если findings свежие);
- системные пары (SSRI+TCA, антикоагулянт+системный НПВП) - **оставить**.

### P3 - опционально (только если попросят)

- Починить шумный `detail_ru` у DDI (у 3651370 detail цитировал чужую Moderate-пару ibuprofen/ketoprofen при правильном title/evidence).
- UX-пояснение: «Критично в очереди» ≠ status записи.
- Не трогать без запроса: official SSOT overall formula, `methodist_labels.jsonl`, Render Action как primary (prod = GCE).

### Не трогать параллельно

- `clinical_knowledge/medication_safety.py`
- `clinical_knowledge/kz_deep_eval.py` (блок C_ddi)
- `clinical_knowledge/mo_action_queue_select.py`
- `clinical_knowledge/mo_zone_scores.py` (`_safety_from_findings`)
- Пока этот handoff не merged / второй агент не взял другую зону файлов.

---

## Синхрон двух агентов и push (как правильно)

Источник истины по процессу: корневой **`AGENTS.md`** + `docs/deploy/multi-agent-single-repo-render-runbook-v2.md`.

### Жёсткие правила

1. **Одна задача - одна ветка - один агент - один worktree.** Не делить ветку.
2. Base **только** свежий `origin/main`. Грязный `~/Cursor_Folders/Protocol` не чинить destructive-командами.
3. Старт:

```bash
cd ~/Cursor_Folders/Protocol
git fetch --prune origin
scripts/ops/git_task_start.sh <task-slug> --pc=<pc-id> \
  --branch=cursor/<task-slug>-<agent-id>-<pc-id>
cd /private/tmp/protocol-task-<…>   # путь из вывода скрипта
```

4. Перед правкой: `gh pr list --repo akuazuk/protocol --state open` - нет ли PR по тем же файлам. Если да - ждать merge или взять другую задачу.
5. Commit → **сразу** `git push -u origin HEAD` → `gh pr create`. Draft PR можно сразу после первого commit (владелец, файлы, зависимости, «не merge пока draft»).
6. **`main` только через merge PR на GitHub.** Прямой push в `main` запрещён. Не переиспользовать чужие/старые ветки и не `codex/main-sync` для новой работы.
7. После merge: **новая** task-ветка от нового `origin/main`. Старую ветку не продолжать.
8. Перед release-значимым commit: `scripts/ops/bump_build_version.sh <slug>`.
9. Deploy GCE (после merge, один координатор):

```bash
GCE_OPS_USER=pavelkuzauka SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
curl -sS https://protocol.kravira.by/api/version
```

10. Координация между агентами - **только GitHub** (PR body + handoff в `docs/reports/`), не stash и не «я локально поправил main».
11. Без AI co-author trailer / «Made with…» в commit/PR.
12. Force-push / `reset --hard` / `clean -fd` по чужой работе - запрещены.

### Если два агента работают в один день

| Агент A | Агент B |
|---------|---------|
| Своя ветка, свой набор каталогов | Своя ветка, **другие** файлы |
| Пишет handoff / обновляет свой план | Читает open PR A перед стартом |
| Merge A → B делает `git fetch` и rebase/merge **origin/main** в свою ветку | Не правит файлы из PR A, пока A не merged |

Конфликт по одному файлу: второй **ждёт** merge первого, потом переносит изменения на новый `main`.

### Одна безопасная следующая команда для завтрашнего агента

```bash
cd ~/Cursor_Folders/Protocol && git fetch --prune origin && \
  scripts/ops/git_task_start.sh mo-ddi-backfill-or-next --pc=pc1 \
  --branch=cursor/mo-ddi-backfill-or-next-pc1
```

Потом прочитать этот handoff и уточнить у владельца: backfill дат vs другая задача из `docs/plans/README.md`.

---

## Проверочные smoke (уже зелёные на момент handoff)

```bash
curl -sS https://protocol.kravira.by/api/version
# ожидание: 2026-08-11-191552Z-safety-zone-p2

# 3665385: good, P2 topical DDI, attention none
# 3651370: review, P1 Major sertraline+amitriptyline - оставить
```

---

## Handoff meta

| | |
|--|--|
| Branch этого документа | `cursor/mo-ddi-handoff-2026-08-11-pc1` |
| Base SHA | `298212ac` |
| Merge/deploy этого handoff | только docs; prod code уже на #138+#139 |
| Следующий deploy | не нужен, пока нет нового кода |
