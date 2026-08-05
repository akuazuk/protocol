# Handoff для агента на 2026-08-06 (МО review workspace + LLM backfill)

Писал агент вечером **2026-08-05**. Владелец спал; работа шла без остановок.
Обновлено после merge/deploy ~19:55 UTC.

## 1. Что сделано сегодня (в git / прод)

### Merged и в проде
| PR | Тема | Версия |
|--|--|--|
| #17 | Named-column warehouse merge; August repair + LLM/BI path | r15-r16 |
| #18 | Filial filter: `|` + comma heuristic (`ул. Захарова, 50Д`) | r17 |
| #19 | Durable Render LLM backfill (`run_mo_render_llm_backfill.sh`) | r18 |
| **#20** | Case review workspace + protocol suggest + gold export | **r19** |

Прод сейчас:
```text
https://protocol-bimy.onrender.com/api/version
→ 2026-08-05-r19-mo-case-review-workspace  (commit c6bc7046)
```

План: `docs/plans/2026-08-05-mo-case-review-workspace-v2.md` (W0-W3 shipped).

### Содержимое #20
1. **UI разбора:** два scroll-pane (МО слева, разбор справа); убраны поля `%`;
   крупный «Развёрнутый разбор» (rows=14 / 12000); RU-лейблы; prev/next; чипы фраз.
2. **Таблица дня:** клик по отчёту → `documents` за день; sort по `th[data-sort-key]`;
   `page_size=100` для single-day (API до 500).
3. **Протоколы МЗ:** `clinical_knowledge/case_protocol_suggest.py` +
   `GET /api/methodist/mo/cases/{id}/protocol-suggest`; UI с оценкой релевантности;
   сохранение в `crm_review_pack` (`protocol_ratings` + snapshot suggest).
4. **Gold:** `scripts/export_mo_review_gold.py`, `scripts/eval_mo_review_gold.py`.
5. Flag `CASE_PROTOCOL_SUGGEST` (default on).

## 2. LLM backfill на Render (НЕ УБИВАТЬ без нужды)

Supervisor на Render disk:

```text
/var/data/medical_exams/logs/run_august_llm.sh
лог: /var/data/medical_exams/logs/mo_llm_august_backfill.log
скрипт репо: scripts/run_mo_render_llm_backfill.sh
```

Диапазон: `2026-08-01` … `2026-08-04` (night grade ~80/день + action-judge ≤20 + recompute).

**Снимок после deploy r19 (~19:55 UTC):**
- 08-01: **80**/80 DONE (+ judge DONE)
- 08-02: **34**/80 in progress (resume)
- 08-03: **82** (уже было)
- 08-04: **80** (уже было)

После **любого deploy** web-сервиса grade на том же инстансе **умирает**.
Сразу проверить и при необходимости перезапустить:

```bash
ssh srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com \
  'ps aux | grep "[.]venv/bin/python scripts/grade_kz"; tail -20 /var/data/medical_exams/logs/mo_llm_august_backfill.log'

# если python grade нет:
ssh … 'nohup bash /var/data/medical_exams/logs/run_august_llm.sh >/dev/null 2>&1 &'
# или с Mac (безопасно: ALREADY_RUNNING если жив):
bash scripts/run_mo_render_llm_backfill.sh 2026-08-01 2026-08-04
```

Важно: `ps aux | grep grade_kz` без якоря может ложно матчить саму команду SSH.
Ищи именно `.venv/bin/python scripts/grade_kz`.

## 3. Как правильно коммитить / мержить / деплоить в этом репо

1. **Не коммить в `main` напрямую** - branch protection. Всегда feature branch + PR.
2. Перед коммитом осмысленных изменений:
   - поднять `BUILD_VERSION` в `rag_server.py` (`YYYY-MM-DD-rN-kebab`, N++);
   - обновить активный план в `docs/plans/` (индекс `docs/plans/README.md`);
   - UI-тексты: `python3 scripts/normalize_ui_dashes.py` **только на нужные файлы**
     (глобальный прогон портит archive/md).
3. Коммит через HEREDOC, без `--no-verify`, без amend чужих коммитов.
4. `git push -u origin HEAD` → `gh pr create` → дождаться CI green →
   `gh pr merge --squash --delete-branch`.
5. После merge дождаться Render auto-deploy:
   `curl -s https://protocol-bimy.onrender.com/api/version`
   Workflow: `Production Render release` на push в `main`.
6. **Сразу после deploy** - проверить LLM supervisor (п.2) и перезапустить при необходимости.
7. **VPN «Дядя Ваня»:** для SQL MIS - `ensure-off`; для сильных моделей - `ensure-on`.
   Gemini с Mac часто geo-blocked → LLM на Render.
8. Не коммитить `.env`, пароли, сырые ПДн, случайные PDF из `minzdrav_protocols/`.

### Шаблон утреннего цикла

```bash
cd ~/Cursor_Folders/Protocol
caffeinate -dims &
git fetch origin && git checkout main && git pull origin main
git checkout -b cursor/<короткая-тема>
# … правки …
# bump BUILD_VERSION
git add -A && git commit -m "$(cat <<'EOF'
MO: краткое why-сообщение.

EOF
)"
git push -u origin HEAD
gh pr create --title "…" --body "…"
# ждать CI
gh pr checks
gh pr merge --squash --delete-branch
# ждать version
# перезапуск LLM если grade умер
```

## 4. Что проверить утром (smoke UI)

1. `/api/version` содержит `r19` (или новее).
2. Разбор случая: МО слева свой scroll; справа разбор скроллится отдельно.
3. Нет полей «Полнота % / Диагноз % / Рек. %».
4. Блок «Протоколы МЗ РБ»; radio релевантности сохраняется в pack.
5. Отчёты → клик по дню → таблица «Все случаи» + sort по заголовку.
6. Фильтр филиала `ул. Захарова, 50Д` даёт ~700 за 01-04 августа.
7. LLM: 01=80; 02→80; 03/04 уже есть; в логе `recompute_mo_days` после DONE.

## 5. Что ещё не закрыто / можно продолжить

- Накопить ≥50 `training_use` packs → первый `export_mo_review_gold.py` + eval.
- Улучшить suggest (audience pediatric, DDx seeds) по `mo-case-protocol-suggest-v1`.
- Repair **июльских** строк warehouse (колоночный сдвиг был и до августа) - опционально.
- GCE europe-north1 host - deferred в review-pack плане.
- Документировать метрики «было/стало» после первого gold export (чекбокс в плане v2).

## 6. Корневой баг BI (уже починен для августа)

`merge_sql` раньше делал `INSERT … SELECT *` → сдвиг колонок →
`doctor_key`/`specialty`/`filial` битые. August 1-4 repaired на Render warehouse.
Июль - ещё может быть битый; не путать с «фильтр сломан».

## 7. Команды быстрого старта

```bash
cd ~/Cursor_Folders/Protocol
caffeinate -dims &
git fetch origin && git checkout main && git pull origin main
git status -sb
gh pr list --state open
curl -s https://protocol-bimy.onrender.com/api/version | python3 -m json.tool | head
bash scripts/run_mo_render_llm_backfill.sh 2026-08-01 2026-08-04
```

SSH Render: `srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com` (ключ `~/.ssh/id_ed25519`).
Repo: `akuazuk/protocol`. Mac не должен засыпать на долгих сессиях: `caffeinate -dims`.
