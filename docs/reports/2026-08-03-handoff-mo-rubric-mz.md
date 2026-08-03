# Handoff: рубрика МЗ и русские замечания МО

Дата: 2026-08-03
Repository: `akuazuk/protocol`
Owner: Cursor agent1 / PC1

## Git и PR

- Branch: `cursor/mo-rubric-mz-scoring-viz-agent1-pc1`
- Worktree: `/private/tmp/protocol-task-mo-rubric-mz-scoring-viz-pc1`
- Base SHA: `728c21cd0eaa2395e5c6d2e1ed65ed06aa61dc46`
- Implementation HEAD до handoff-коммита: `fa8dea8ebe8ef10655ed6092d8c473b7598b1fa2`
- PR: https://github.com/akuazuk/protocol/pull/5
- PR state: draft, mergeable
- Расхождение перед handoff: `origin/main...HEAD = 0 / 5`

## Сделано

1. Добавлена shadow-рубрика МЗ из таблицы «Как оценивать»:
   13 критериев, шкала `0 / 0.5 / 1 / n/a`.
2. Рубрика добавлена в case detail, overview и разрез по специальностям.
3. Добавлен поиск предыдущего визита для оценки коррекции по динамике.
4. Добавлены русские названия кодов замечаний.
5. В отчёте за вчера замечания открывают список МО за тот же день, а ссылки-примеры
   открывают содержание конкретного МО в drawer.

## Версия

- Ветка до handoff: `2026-08-03-r18-finding-labels-ru`.
- Для compliance-коммита версия поднята обязательным скриптом
  `scripts/ops/bump_build_version.sh rubric-handoff-workflow`.
- Новая версия: `2026-08-03-r19-rubric-handoff-workflow`.
- Production остаётся на `2026-08-03-r13-multi-agent-runbook`, потому что PR не merged
  и deploy не выполнялся.

## Проверки

- `git diff --check origin/main...HEAD` - успешно.
- `node --check frontend/web/shared/mo-app.js` - успешно.
- `pytest tests/test_mo_rubric_mz.py tests/test_mo_rubric_prior.py
  tests/test_mo_frontend_structure.py` - 18 passed.
- `pytest tests/test_mo_finding_labels_ru.py tests/test_mo_yesterday.py
  tests/test_mo_frontend_structure.py` - успешно.
- Manifest mode CI - success.
- Глобальный `lint-and-test` - failure: `ruff check .` на 102 baseline-ошибках,
  включая файлы вне PR (`scripts/run_kz_vector_index_eval.py`,
  `scripts/telegram_control.py`, исторические tests).
- Локальный Ruff отсутствует; changed-file Ruff отдельно не выполнен.

## Merge, deploy и smoke

- Merge: не выполнялся.
- Production deploy: не выполнялся.
- Production SHA: `origin/main` / `728c21cd`.
- Production version smoke: `2026-08-03-r13-multi-agent-runbook`.
- Feature smoke на production невозможен до merge/deploy.
- Красный required CI нельзя игнорировать без явного решения владельца.

## Не сделано

1. PR не переведён из draft в ready.
2. PR не reviewed и не merged.
3. Не выполнены серверный warehouse-фильтр рубрики и ECharts heatmap.
4. Не выполнен production feature smoke.

## Файлы, которые нельзя менять параллельно до завершения PR #5

- `rag_server.py`
- `clinical_knowledge/mo_backend.py`
- `clinical_knowledge/mo_case_document.py`
- `clinical_knowledge/mo_rubric_mz.py`
- `clinical_knowledge/mo_finding_labels_ru.py`
- `frontend/web/shared/mo-app.js`
- `frontend/web/shared/mo-ui.css`
- `frontend/web/methodist/mis-kz-quality.html`
- `config/mo_rubric_mz.yaml`
- связанные тесты и план рубрики МЗ

## Следующая безопасная команда

```bash
git fetch origin && git rev-list --left-right --count origin/main...HEAD
```

После проверки - review PR #5 и явное решение владельца по baseline-красному CI.
