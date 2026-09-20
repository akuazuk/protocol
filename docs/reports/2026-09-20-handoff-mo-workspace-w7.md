# Handoff: стол МО (W1-W7) и план разбора случая

Дата: 2026-09-20
Repo: `akuazuk/protocol`
Прод: GCE `protocol.kravira.by` (не Render)

## Сделано

| Волна | PR | Merge SHA | BUILD_VERSION | Прод |
|--|--|--|--|--|
| W1 календарь | #262 | `b3a37e3b` | - | да |
| W2 инспектор на всю страницу | #263 | `804f87da` | - | да |
| W3 честный Обзор | #264 | `deb7e72f` | - | да |
| W4 Поиск МИС + ingest | #265 + hotfix #266 | `84b13b83` | - | да |
| W5 dx_text склада | #267 | `b4a900e1` | `2026-09-20-142257Z-mo-w5-dx-text` | да |
| W6 кнопки / Период | #268 | `e374e996` | `2026-09-20-145012Z-mo-w6-buttons` | да |
| W7 SQL overall_grade + индексы | #269 | `a22c4744` | `2026-09-20-153919Z-mo-w7-sql-grade` | да |

W7 smoke на проде (месяц, клинические, `score_eligible_only=1`):

- `/api/version`: `2026-09-20-153919Z-mo-w7-sql-grade`, `git_commit=a22c4744c984`
- `/health/live` ок
- фильтр `overall_grade`: good 329, fair 5752, poor 1867, important 114, сумма 8062 = без фильтра
- строки выборки совпадают с запрошенным grade
- UI «Найти МО · 329 записей» при `overall_grade=good`
- индексы на host sqlite: `idx_case_zone_bands`, `idx_case_attention_date`, `idx_finding_mis`, `idx_fact_mo_case_visit`

CI W7: первый прогон красный (`CREATE INDEX` по `zone1_band` до `_ensure_columns`). Фикс `387779c8`, затем CLEAN.

## Не сделано / не врать

- Чипы «Критично» и «Нет оценки» по SQL пустые: на складе нет `safety_band`, CASE не эмитит `critical`; `na` отсекается `score_eligible_only`. Отдельный PR, не смешивать с разбором.
- Warm `/cases?overall_grade=` около 2.2 с - как список без фильтра; это не python-post-filter 8k, но p95 ещё не «мгновенно».
- План разбора случая написан, runtime R0 ещё нет.
- `docs/plans/README.md` не трогали (занят).
- PR стола `#261` может быть ещё открыт - это индекс/план W0-W7, не runtime.

## Делается

План `docs/plans/2026-09-20-mo-case-review-accuracy-v1.md` (волны R0-R7).

## Нужно

Следующий runtime: R0 (split 1440, протокол на первом экране, без смены скоринга).

```bash
scripts/ops/git_task_start.sh mo-case-review-r0 --pc=pc1 \
  --branch=cursor/mo-case-review-r0-pc1
```

Запреты: не Render; не порт 8000; не dirty Cursor `main`; не `SYNC_PROTOCOL_CORPUS=1`; Gemini только GCE; PHI в чат не писать.

## Файлы, которые нельзя трогать параллельно с R0

`frontend/web/shared/mo-app.js`, `frontend/web/shared/mo-ui.css`, `frontend/web/mis-kz-quality.html`.
`rag_server.py` - только `BUILD_VERSION` / узкий Query.
`docs/plans/README.md` - не этот PR.
