# Handoff: merge #149, GCE deploy, July KP eval

- repo: `akuazuk/protocol`
- production SHA: `8d69404228faf3817098682eccabffcffbabe8b6`
- `BUILD_VERSION` на `protocol.kravira.by`: `2026-08-14-090123Z-kp-adult-pop-fix`
- `/health/live`: ok
- план точности: `docs/plans/2026-08-14-mo-kp-suggest-accuracy-v1.md`
- ветка плана: `cursor/kp-suggest-accuracy-plan-pc1`

## Сделано

- Merge PR #149 (CI зелёный после фикса `дет_нас` vs взрослый K21.9).
- Deploy GCE из worktree `origin/main` @ `8d69404`.
- Smoke: геморрой / I84.9 → КП №22.
- Прогон июля: склад 9991 clinical; выборка 300 по главам МКБ;
  available 57.3%, empty 42.7%; почти всё путём МКБ (текст Dx в витрине редкий).
  Отчёт на GCE: `/var/data/medical_exams/reports/kp_suggest_eval_2026-07.json` (без PHI).

## Не сделано

- Полный 9991 (медленно).
- CSV-прогон с жалобами/полным Dx.
- Шаги 2-7 плана (омнибус, пустые кластеры, возраст, скорость, golden).

## Не параллелить

`clinical_knowledge/case_protocol_suggest.py`, `protocol_match.py`,
`applicability.py`, `data/catalog/protocol_content_index.json`.
Не трогать PR #146 (Rceth).
