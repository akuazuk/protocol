# Handoff: разделы Лекарства / Анализы (D0-D4)

Дата: 2026-09-05

## Repo

- repo: `akuazuk/protocol`
- branch: `cursor/mo-drugs-labs-scoring-agent1-pc1`
- worktree: `/private/tmp/protocol-task-mo-drugs-labs-scoring-pc1`
- план: `docs/plans/2026-09-05-mo-meds-labs-dashboards-v1.md`
- PR: https://github.com/akuazuk/protocol/pull/187 (ещё не содержит этот diff, пока локально)

## Сделано

- D0: `data/mo_finding_families/families_v1.json` + `clinical_knowledge/mo_finding_families.py`. Тест «нет сирот». `finding_family=drug|lab` в `/api/methodist/mo/cases`. `build_mo_drugs_labs_kpis` отдаёт `families`, `denominators`, `strips`.
- D1: меню «Лекарства», страница KPI / коды / специальность / врач / превью, drill в «Все случаи», полоска на Сегодня и Период, бейдж «черновик, не в общей оценке».
- D2: зеркало «Анализы»; знаменатель `cases_with_lab` из `mo_lab.sqlite` если склад есть, иначе доля от всех МО с явной подписью.
- D3: `drug_score` / `lab_score` в `evaluate_kz_deep` и case detail; блоки в разборе случая.
- D4: `MO_FAMILY_SCORES_IN_OVERALL=0` по умолчанию. Формула в JSON: `min(ось, подось)` только при флаге **и** `MO_*_PRIMARY`. №55 не трогаем.

## Не сделано

- A/B на живом месяце до включения overall (только владелец / GCE).
- Gold ≥50 по unused lab / Rceth / class-dup.
- Включение `*_PRIMARY`.
- Commit / push / merge / deploy этого diff.

## Тесты

```bash
/opt/homebrew/bin/pytest tests/test_mo_finding_families.py \
  tests/test_mo_family_scores.py tests/test_mo_meds_labs_dashboards.py \
  tests/test_mo_dashboard_nav_cleanup.py tests/test_mo_dashboard_hero_cleanup.py \
  tests/test_mo_frontend_structure.py tests/test_mo_ui_phase2.py -q
```

Прошли. Live warehouse в этой среде не открывался: UI drill на проде нужно кликнуть после деплоя.

## Deploy

Не было. Overall без флага не меняется.

## Одна безопасная следующая команда

```bash
# из worktree, после явного «закоммить»:
scripts/ops/bump_build_version.sh mo-meds-labs-dash
git add -p && git commit && git push origin HEAD
```

## Не трогать параллельно

- `frontend/web/shared/mo-app.js`
- `frontend/web/methodist/mis-kz-quality.html`
- `clinical_knowledge/mo_backend.py`
- `clinical_knowledge/kz_deep_eval.py`
- `clinical_knowledge/mo_finding_families.py`
