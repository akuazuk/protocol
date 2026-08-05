# Handoff для агента на 2026-08-06 (МО review workspace + LLM backfill)

Писал агент вечером **2026-08-05**. Владелец спал; работа шла без остановок.

## 1. Что сделано сегодня (в git / прод)

### Merged ранее в тот же день
- **#17** warehouse named-column merge + August LLM/BI backfill path (`r15`→часть)
- **#18** filial filter pipe / comma heuristic (`r17`)
- **#19** durable Render LLM backfill script (`r18`)

### Эта ветка / PR (case review workspace v2)
План: `docs/plans/2026-08-05-mo-case-review-workspace-v2.md`

Ожидаемый merge: PR с ветки `cursor/mo-case-review-workspace-v2`, версия
`2026-08-05-r19-mo-case-review-workspace`.

Содержимое:
1. **UI разбора:** два scroll-pane (МО слева, разбор справа); убраны поля `%`;
   крупный «Развёрнутый разбор»; RU-лейблы; prev/next; чипы фраз.
2. **Таблица дня:** клик по отчёту → `documents` за день; sort по `th[data-sort-key]`;
   `page_size=100` для single-day.
3. **Протоколы МЗ:** `clinical_knowledge/case_protocol_suggest.py` +
   `GET /api/methodist/mo/cases/{id}/protocol-suggest`; UI с оценкой релевантности;
   сохранение в `crm_review_pack` (`protocol_ratings` + snapshot suggest).
4. **Gold export/eval:** `scripts/export_mo_review_gold.py`, `scripts/eval_mo_review_gold.py`.

## 2. LLM backfill на Render (НЕ УБИВАТЬ без нужды)

Supervisor на Render disk:

```text
/var/data/medical_exams/logs/run_august_llm.sh
лог: /var/data/medical_exams/logs/mo_llm_august_backfill.log
```

Диапазон: `2026-08-01` … `2026-08-04` (night grade + action-judge + recompute).

Проверка:

```bash
ssh srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com \
  'ps aux | grep "[g]rade_kz_llm\|[r]un_august"; tail -20 /var/data/medical_exams/logs/mo_llm_august_backfill.log'
```

После **любого deploy** web-сервиса процесс на том же инстансе может умереть.
Сразу перезапустить:

```bash
ssh … 'nohup /var/data/medical_exams/logs/run_august_llm.sh >/dev/null 2>&1 &'
# или с Mac:
bash scripts/run_mo_render_llm_backfill.sh 2026-08-01 2026-08-04
```

Скрипт сам не стартует второй grade, если уже есть `scripts/grade_kz_llm.py`.

## 3. Как правильно коммитить / мержить / деплоить в этом репо

1. **Не коммить в `main` напрямую** - branch protection. Всегда feature branch + PR.
2. Перед коммитом осмысленных изменений:
   - поднять `BUILD_VERSION` в `rag_server.py` (`YYYY-MM-DD-rN-kebab`);
   - обновить активный план в `docs/plans/`;
   - UI-тексты: `python3 scripts/normalize_ui_dashes.py` **только на нужные файлы**
     (глобальный прогон портит archive/md).
3. Коммит через HEREDOC, без `--no-verify`, без amend чужих коммитов.
4. `git push -u origin HEAD` → `gh pr create` → дождаться CI green →
   `gh pr merge --squash --delete-branch`.
5. Workspace rule: после merge/`push` задача не закончена без деплоя Render
   (обычно auto from `main`). Проверить:
   `curl -s https://protocol-bimy.onrender.com/api/version`
6. **VPN «Дядя Ваня»:** для SQL MIS - `ensure-off`; для сильных моделей - `ensure-on`.
   Gemini с Mac часто geo-blocked → LLM на Render.
7. Не коммитить `.env`, пароли, сырые ПДн, гигантские PDF из `minzdrav_protocols/`
   если они случайно untracked.

## 4. Что проверить утром (smoke)

1. `/api/version` содержит `r19` (или новее, если были ещё PR).
2. Разбор случая: МО слева не уезжает при скролле правой колонки.
3. Нет полей «Полнота % / Диагноз % / Рек. %».
4. Блок «Протоколы МЗ РБ» появляется; radio релевантности сохраняется в pack.
5. Отчёты → клик по дню → таблица «Все случаи» за этот день + сортировка по заголовку.
6. Фильтр филиала `ул. Захарова, 50Д` даёт ~700 за 01-04 августа.
7. LLM grades: 01=80, 02-04 догоняются; после DONE - `recompute_mo_days` в логе.

## 5. Что ещё не закрыто / можно продолжить

- Накопить ≥50 `training_use` packs → первый `export_mo_review_gold.py` + eval report.
- Улучшить suggest (audience pediatric, DDx seeds) по плану `mo-case-protocol-suggest-v1`.
- Repair **июльских** строк warehouse (колоночный сдвиг был и до августа) - опционально.
- GCE europe-north1 host - deferred в review-pack плане.

## 6. Команды быстрого старта

```bash
cd ~/Cursor_Folders/Protocol
caffeinate -dims &   # если долгая сессия
git fetch origin && git checkout main && git pull origin main
git status -sb
gh pr list --state open
# LLM status
bash scripts/run_mo_render_llm_backfill.sh 2026-08-01 2026-08-04   # безопасно: ALREADY_RUNNING если жив
```

SSH Render: `srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com` (ключ `~/.ssh/id_ed25519`).
