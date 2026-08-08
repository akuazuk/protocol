# Handoff: зоны МО Аналитика (A–E) — 2026-08-08

**Ветка:** `cursor/mo-zone-scores-engine-pc1`  
**PR:** https://github.com/akuazuk/protocol/pull/78  
**Worktree:** `/private/tmp/protocol-task-mo-zone-scores-engine-pc1`  
**BUILD_VERSION:** см. `rag_server.py` (ожидать после merge/deploy в `/api/version`)  
**Планы:** ui-target-v2, mz-sheet-layers-v2, implementation-blueprint-v1

---

## Сделано (релизы A–E в одном PR)

| Релиз | Содержание |
|--|--|
| A | Движок `mo_zone_scores`, DDL/upsert склада, overview/cases API, тесты |
| B | Лаконичный разбор (3 зоны → «Что не так»), колонки зон в таблицах |
| C | Меню 6 пунктов (Сегодня/Период/Очередь/Все случаи/Врачи/Отчёты), attention strip |
| D | Экран «Врачи» по % плохо, URL `zone`/`zone_band`/`attention_only`/`kp_status`/`history_tier`, 4 пресета |
| E | Обновлён `docs/methodist/mo-evaluation-catalog.md` (§Z = канон дашборда); этот handoff |

## Не сделано / нужно после merge

1. Merge PR #78 в `origin/main`.
2. GCE deploy: `deploy/gcp-app/deploy_to_gce.sh` (не Render promote task HEAD).
3. Recompute зон за **14 дней** на складе GCE (`MO_ZONE_SCORES=1`, resume/retry по runbook).
4. Smoke: `/api/version` = BUILD_VERSION; разбор случая с `zones.ok`; Сегодня/Период с полосой внимания (не «ещё не посчитано»); Врачи с % плохо.
5. Калибровка методиста 30 кейсов → `docs/reports/YYYY-MM-DD-mo-zones-calibration.md` (опционально, blueprint §13.2).

## Тесты (локально в worktree)

```text
pytest tests/test_mo_zone_scores.py tests/test_mo_zone_api.py tests/test_mo_frontend_structure.py
node --check frontend/web/shared/mo-app.js
```

Известные baseline CI вне этого diff не разбирались в этой сессии.

## Файлы (не трогать параллельно до merge)

- `clinical_knowledge/mo_zone_scores.py`, `mo_daily.py`, `mo_backend.py`
- `config/mo_rubric_mz.yaml`, `config/mo_zone_bands.yaml`
- `frontend/web/shared/mo-app.js`, `mo-ui.css`
- `frontend/web/methodist/mis-kz-quality.html`, `expert.html`
- `docs/methodist/mo-evaluation-catalog.md`
- `docs/plans/2026-08-08-mo-analytics-*.md`

## Одна безопасная следующая команда

```bash
gh pr merge 78 --squash --delete-branch
# затем release-координатор:
# deploy/gcp-app/deploy_to_gce.sh
# и recompute 14 дней зон на GCE
```

## Production

- Merge/deploy на момент handoff: **ещё нет** (код только в PR).
- Analytics UI production host: GCE `https://protocol.kravira.by` (не Render для этого контура).
