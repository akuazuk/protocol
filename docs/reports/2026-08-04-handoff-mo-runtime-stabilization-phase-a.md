# Handoff PC2: MO runtime Phase A (2026-08-04)

**Владелец ветки (PC1 / agent1):** работа завершена на ветке  
`codex/mo-runtime-stabilization-agent1-pc1` @ `60f349a` (+ docs ниже после push).  
**План:** `docs/plans/2026-08-04-mo-runtime-stabilization-v1.md` (active).  
**Канон координации:** `AGENTS.md`, `docs/deploy/multi-agent-single-repo-render-runbook-v2.md`,  
`docs/deploy/two-computers-daily-checklist.md`.

Этот файл - инструкция для **другого агента на другом компьютере**. Не invent новых шагов:
сначала прочитай целиком, потом выполняй только чеклист в конце.

---

## 1. Что уже сделано сегодня (PC1) - НЕ повторять без нужды

### Код (в task-ветке, ещё не в `main` на момент handoff)

| ID | Суть | Файлы |
|---|---|---|
| A1 | `llm_queue_pending` = **advisory**, не `partial`, если coverage >= 99% и нет scoring_errors | `clinical_knowledge/mo_daily.py` |
| A1b | Скрипт переклассификации без пересчёта оценок | `scripts/mo_reclassify_advisory_partial.py` |
| A2 | Launchd wrapper читает `METHODIST_TOKEN` из `ROOT/.env` / `PROTOCOL_ENV_FILE` (не из plist) | `scripts/run_mo_daily_launchd.sh`, `scripts/manage_mo_daily_launchd.py` |
| A3 | Publish: exit 3 при freshness != 200 / missing token; Telegram на fail | `scripts/publish_mo_to_render.py` |
| A4 | Auto-clear stale `pipeline.lock` / `launchd-run.lock` если pid мёртв | `mo_daily.exclusive_lock`, launchd shell |
| A5 | Health `yesterday.{partial,reasons,advisory_reasons}` + UI «Вчера» | `mo_backend.py`, `frontend/web/shared/mo-app.js` |
| ver | `BUILD_VERSION` | `2026-08-04-r1-mo-runtime-phase-a` (или новее, если в ветке был docs-bump) |

### Данные (уже на Mac PC1 и на диске Render)

1. Reclassify `2026-08-01`, `2026-08-02`, `2026-08-03`:  
   `partial=true` + reasons=`llm_queue_pending` → `partial=false`, advisory=`llm_queue_pending`, warehouse `quality_status=passed`.
2. Publish на Render успешен: remote `through=2026-08-03`, freshness **HTTP 200**, `lag_days=1`.

**Не запускать снова** reclassify этих дат и полный publish «на всякий случай», пока smoke после deploy не покажет регресс.

### Чего ещё нет

- PR может быть ещё не открыт / не merged (на PC1 не было `gh auth` до конца сессии).
- Прод-код всё ещё может быть `c244c98` / `2026-08-03-r24-ci-release-handoff` - это **нормально**, пока PR не в `main`.
- Фаза B (Docker / cloud worker) **не начинать** в этом handoff.

---

## 2. Preflight на PC2 (обязательно)

```bash
git status --short --branch
git fetch --prune origin
git rev-list --left-right --count origin/main...HEAD
gh pr list --repo akuazuk/protocol --state open
```

Правила:

1. **Не** чинить грязный `main` pull/reset. Новый clean worktree от `origin/main`.
2. **Не** пушить в чужую ветку `codex/mo-runtime-stabilization-agent1-pc1` новыми фичами.
3. Если нужно только merge/deploy - работай через GitHub PR UI / `gh pr merge`, без локальных правок кода.
4. Секреты (`METHODIST_TOKEN`, DB password) не печатать в чат/PR/handoff.

---

## 3. Что сделать агенту на PC2 (по порядку)

### Шаг A - убедиться, что PR существует

Ветка: `codex/mo-runtime-stabilization-agent1-pc1`  
Создать PR если нет:

```bash
gh pr create --repo akuazuk/protocol \
  --base main \
  --head codex/mo-runtime-stabilization-agent1-pc1 \
  --title "fix(mo): Phase A runtime stabilization (LLM partial, publish, health)" \
  --body-file - <<'EOF'
## Summary
- llm_queue_pending is advisory when scoring coverage meets target
- launchd loads METHODIST_TOKEN from .env; publish fails closed on freshness/token
- stale locks cleared; health/UI expose yesterday partial/advisory reasons
- reclassify script + Phase A handoff for PC2

## Already done on data plane (do not redo)
- reclassify 2026-08-01..03 + publish (through=2026-08-03, freshness 200)

## Test plan
- [ ] CI green or known baseline noted
- [ ] After merge: Action Production Render release
- [ ] /api/version == BUILD_VERSION from merge commit
- [ ] health.yesterday + UI «Вчера» smoke
EOF
```

Или вручную:  
https://github.com/akuazuk/protocol/pull/new/codex/mo-runtime-stabilization-agent1-pc1

Перед merge: `git fetch origin` и убедиться, что head не отстаёт от `origin/main` настолько, что GitHub требует update. При конфликтах - **не force-push**; синхронизировать через merge/rebase только владелец ветки (PC1) или явный новый интеграционный PR.

### Шаг B - merge (только release-координатор / владелец)

```bash
gh pr merge <PR_NUMBER> --squash --delete-branch=false
git fetch --prune origin
git rev-parse origin/main   # зафиксировать merge SHA
```

Deploy: только Action `Production Render release` (`concurrency: production-render`).  
**Запрещено:** `render_promote_main.sh`, `deploy_promote_main_after_push.sh`, deploy task HEAD.

```bash
gh run list --repo akuazuk/protocol --workflow=render-production-deploy.yml --limit=3
gh run watch --repo akuazuk/protocol <run-id>
```

### Шаг C - smoke после deploy (обязательно)

Ожидаемо:

- `version` содержит `2026-08-04-r`…`mo-runtime-phase-a` (или docs-bump после него)
- `git_commit` == `origin/main` HEAD после merge (не `c244c98`)

```bash
curl -fsS https://protocol-bimy.onrender.com/api/version | jq '{version,git_commit}'

curl -fsS -H "X-Methodist-Token: $METHODIST_TOKEN" \
  https://protocol-bimy.onrender.com/api/methodist/mo/health \
  | jq '{status, yesterday, reason_codes}'

curl -fsS -H "X-Methodist-Token: $METHODIST_TOKEN" \
  "https://protocol-bimy.onrender.com/api/methodist/mo/daily-report?date=2026-08-03" \
  | jq '{partial, quality_status, data_completeness: .data_completeness|{partial,partial_reasons,advisory_reasons,llm_queue_pending,coverage_pct}}'
```

UI: страница «Вчера» - день не «eternal partial»; при очереди LLM - advisory, не блок.

### Шаг D - Mac launchd на PC2 (только если этот Mac тоже гоняет pipeline)

Секреты **не** в plist:

1. В корне checkout должен быть `.env` с `METHODIST_TOKEN` (локально, не в git).
2. После `git pull`/`worktree` с новым wrapper:

```bash
python3 scripts/manage_mo_daily_launchd.py status
# при необходимости переустановить jobs (подтянет новый wrapper path):
python3 scripts/manage_mo_daily_launchd.py install
```

3. Ручной publish (если нужен disaster recovery):

```bash
bash scripts/run_mo_daily_launchd.sh publish
```

VPN «Дядя Ваня»: `ensure-off` перед MariaDB; после SQL при необходимости `ensure-on` для сильных моделей.

### Шаг E - что дальше (не смешивать с merge)

Фаза B (Docker / вынос worker) - только после закрытого smoke фазы A и по плану  
`docs/plans/2026-08-04-mo-runtime-stabilization-v1.md`.  
Новая задача = **новая** ветка/worktree от свежего `origin/main`, не дописывать Phase A ветку.

---

## 4. Запреты (чтобы ничего не сбить)

| Не делать | Почему |
|---|---|
| Push в `main` напрямую | branch protection + runbook |
| Deploy не-`origin/main` SHA | fail-closed release scripts |
| Параллельный второй Render deploy | concurrency group |
| Дописывать код в ветку PC1 с PC2 | один владелец ветки |
| Повторный full re-score 01-03 «чтобы починить partial» | уже reclassify + publish |
| Класть `METHODIST_TOKEN` в LaunchAgents plist / git | секреты только `.env` / Secret Manager |
| Считать «готово» по merge без `/api/version` | deploy мог не пройти |

---

## 5. Одна следующая безопасная команда для PC2

```bash
git fetch --prune origin && gh pr list --repo akuazuk/protocol --head akuazuk:codex/mo-runtime-stabilization-agent1-pc1 --state all
```

Если PR open → review/merge → watch Action → smoke из шага C.  
Если PR отсутствует → `gh pr create` из шага A.

---

## 6. Файлы, которые нельзя параллельно ломать другими PR

Пока этот PR не merged:

- `scripts/publish_mo_to_render.py`
- `scripts/run_mo_daily_launchd.sh`
- `clinical_knowledge/mo_daily.py` (`assess_completeness` / locks)
- `clinical_knowledge/mo_backend.py` (`build_mo_health` / yesterday completeness)
- `frontend/web/shared/mo-app.js` (yesterday banner) - только согласованные правки

После merge координация через новый `origin/main`.
