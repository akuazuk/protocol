# Handoff следующему агенту (после ICD + eval fixes, 06.08 день)

**Актуальный handoff** (заменяет утренний
`docs/handoff/2026-08-06-next-agent-mo-review.md` как точку входа).  
**Когда:** 2026-08-06 ~14:30 UTC+3.  
**Репо:** `akuazuk/protocol`. **Прод:** `https://protocol-bimy.onrender.com`.  
**SSH Render:** `srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com`  
(ключ: `~/.ssh/id_ed25519`).

---

## 0. Старт (сделай первым, до любой правки)

```bash
cd /path/to/protocol          # твой clone
git fetch --prune origin
git checkout main
git pull --ff-only origin main
git status --short --branch
git log -5 --oneline
curl -sS https://protocol-bimy.onrender.com/api/version
gh pr list --repo akuazuk/protocol --state open
```

**Ожидай в `origin/main` минимум:**

| SHA (short) | PR | Суть |
|--|--|--|
| `96c19eb` | #32 | МКБ по всему тексту МО |
| `a2f8db9` | #31 | «Врачи ниже ожидаемого» без жёсткого R² |
| `9da300c` | #30 | Gemini только через Render + durable runner |

**Прод `/api/version`:** `2026-08-06-095636Z-icd-full-document`  
(или новее после твоего merge). Если версия старше - подожди Render webhook /
`scripts/ops/render_release_main.sh`.

Дальше: `docs/plans/README.md` (только `active`). Не начинай с грязного
checkout и не коммить в чужую task-ветку.

---

## 1. Как правильно работать и мержить (обязательно)

Канон: `docs/deploy/multi-agent-single-repo-render-runbook-v2.md` + `AGENTS.md`.

### Workflow одной задачи

1. **Только task-ветка** от свежего `origin/main`:
   `git checkout -b cursor/<topic>-pcN`
2. Правки + тесты. Перед коммитом:
   `scripts/ops/bump_build_version.sh <slug>`  
   → формат `YYYY-MM-DD-HHMMSSZ[-slug]` (не руками `rN`).
3. Commit **без** AI Co-authored-by / Cursor / Claude trailer  
   (правило `no-ai-vendor-attribution`).
4. `git push -u origin HEAD`
5. `gh pr create` → дождись **CI green** (`gh pr checks N --watch`)
6. Merge: **`gh pr merge N --squash --delete-branch`**  
   (не merge commit в main без необходимости; squash = стиль последних PR).
7. Deploy: только после того как SHA в **`origin/main`**:
   ```bash
   git fetch origin
   git checkout main && git pull --ff-only origin main
   scripts/ops/render_release_main.sh \
     --commit="$(git rev-parse origin/main)" \
     --prod-url=https://protocol-bimy.onrender.com
   ```
8. Не считать готовым, пока `/api/version` ≠ локальный `BUILD_VERSION`.

### Запреты

- Не пушить напрямую в `main` (кроме редкого docs-only если владелец явно сказал;
  по умолчанию - PR).
- Не `render_promote_main.sh` с task HEAD - скрипт специально отключён.
- Не force-push в `main` / shared ветки.
- Один release-координатор на merge/deploy; не параллельные деплои.
- Не коммитить `data/ml/reports/v4/` dumps, `.env`, пароли МИС.
- UI-тексты: короткий дефис `-`, не em/en dash (`hyphen-dash.mdc`).

### VPN

- Перед SQL МИС: `~/CURSOR/bin/vanya_vpn.sh ensure-off`
- Перед Gemini / сильными моделями Cursor: `ensure-on`
- **LLM grading МО никогда с Mac** - только Render (`gemini-via-render.mdc`)

---

## 2. Что уже в проде (сессия 06.08)

| PR | Что важно агенту |
|--|--|
| #26 | Publish merge named columns + keep CSV on failure |
| #27 | Ban AI vendor attribution |
| #28 | Case findings clarity (RU, P0 №55 demote false, proto-viewer) |
| #29 | Полные названия КП + поиск из разбора |
| #30 | Gemini только Render; `mo_llm_range_runner.sh` + `run_mo_render_llm_backfill.sh` |
| #31 | Yesterday `doctor_outliers`: убран hard `case_mix_reliable` (R²); soft note |
| #32 | **`mo_icd_resolve.resolve_icd_codes_from_mo`** - МКБ по всему МО |

### Ключевые файлы оценки

- `clinical_knowledge/mo_icd_resolve.py` - helper МКБ full-document
- `.cursor/rules/mo-icd-full-document.mdc` - правило для агентов
- `docs/plans/2026-08-06-mo-icd-full-document-search-v1.md` - P0-P2 done; **P3 warehouse soft-fill, P4 LLM prompts** open
- deep / v3 / reg55 / protocol suggest уже на helper

### Render disk (снимок)

- Night grades **2026-08-05: 80/80** ok  
  `secure_cases/2026/08/kz_l1_2026-08-05_llm_grades.jsonl`
- Deploy **убивает** nohup grade - после деплоя перезапуск:
  `bash scripts/run_mo_render_llm_backfill.sh YYYY-MM-DD YYYY-MM-DD`
- Лаунчер: **scp static** `scripts/mo_llm_range_runner.sh`  
  (nested heredoc раньше схлопывал `$DATA`/`$d` в пустоту)
- False `ALREADY_RUNNING`: pgrep только по  
  `[.]venv/bin/python .*grade_kz_llm[.]py`

### Gold / методист

- `crm_review_pack` ≥1: кейс **`3650612`** (уролог, post-circumcision), actor `expert:expert`
- Эксперт: FP `D_reg55_p0`, FP `C_nsaid_dup` (альтернативы в скобках);
  confirmed `B_icd_invalid`, `C_ddi`, `E_template_copy`;
  protocol suggest top - **челюстно-лицевые** → все `irrelevant`

---

## 3. Что брать дальше (приоритет)

### P0 (качество оценки по gold 3650612)

1. **`C_nsaid_dup`:** не считать НПВП в скобках / «и т.п.» / «или» одновременным приёмом  
   (`medication_safety.nsaid_labels_in_text` / deep).
2. **Дикловит** в словарь НПВП; сигнал = Дикловит + пероральный НПВП без запрета комбинации.
3. **Protocol Suggest:** specialty Уролог + постоперационные урологические КП,  
   не челюстно-лицевые на gap `C_nsaid_dup`.
4. После правок - **recompute** дней (код в проде ≠ пересчёт старых findings в warehouse).

### P1 (данные / BI)

5. ICD plan **P3**: soft-fill `mkb_code_main` в warehouse/export без ломки MIS agreement.
6. ICD plan **P4**: строка в LLM judge / methodist prompts про полный документ.
7. Июльский warehouse: ~11k bad `doctor_key` (status вместо hash) - repair script.
8. Gold export после ≥50 `training_use` packs.

### P2

9. Concordance Смирнова / shadow findings - план `mo-eval-smirnova-concordance-v1` (E0-E3 done; калибровка дальше по метрикам).
10. Continuous LLM на новый день + action-judge post-step (частично есть).

---

## 4. Быстрые команды

```bash
# LLM на Render (VPN off для SSH; Gemini на Render)
bash scripts/run_mo_render_llm_backfill.sh 2026-08-05 2026-08-05

# Пересчёт отчётов дня
# (на Render через runner или локально с путями к /var/data…)

# Версия
grep BUILD_VERSION rag_server.py
curl -sS https://protocol-bimy.onrender.com/api/version

# Тесты ICD
python3 -m pytest tests/test_mo_icd_full_document.py --noconftest -q
```

MIS SQL: VPN off → `~/CURSOR/sql_epam` + `KRAVIRA_DB_PASSWORD` из `.env`  
(см. `.cursor/rules/mis-mariadb.mdc`). Пароль не печатать.

---

## 5. Definition of Done для твоей следующей задачи

1. Task-ветка + bump `BUILD_VERSION` + PR squash-merge в `origin/main`.
2. `render_release_main.sh` → `/api/version` совпал.
3. План в `docs/plans/` обновлён (шаги/метрики); при новой теме - новый `…-vN.md`.
4. Этот handoff при существенном сдвиге - новый файл  
   `docs/handoff/YYYY-MM-DD-….md` и ссылка сверху у предыдущего.

Удачной работы.
