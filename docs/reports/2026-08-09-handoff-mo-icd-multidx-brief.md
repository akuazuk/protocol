# Handoff: multi-dx ICD name match + criteria brief

Дата: 2026-08-09  
Repo: `akuazuk/protocol`  
Branch: `cursor/mo-icd-name-multidx-pc1`  
Worktree: `/private/tmp/protocol-task-mo-icd-name-multidx-pc1`  
PR: https://github.com/akuazuk/protocol/pull/82  
Base: `origin/main` @ `619ff1df`  
HEAD: see PR  
Deploy: not yet (await merge → GCE)

## Сделано

1. Root cause ложного «Формулировка слабо совпадает… J30.4/R51 score≈0.38»:
   `mo_icd_name_match` сравнивал весь мультидиагноз с одним коротким `title_ru`.
2. Fix: phrase-split + max score; code-seed titles; Cyrillic ICD strip.
3. UI: блок «Разбор по критериям» (brief + полная таблица по зонам).
4. Title boilerplate «заменить словами» → filename.
5. Risk markers: семейная АГ / гиперхолестеринемия.

## GCE protocol-suggest (до деплоя PR)

Engine `case_protocol_suggest_v5`, mode `icd_first`.

Должен встать в разбор:

- **КП Диагностика и лечение пациентов д-нас с бронхиальной астмой, пост. МЗ 2025 №38**
  (`…_бронхиальной_38_l4`, PDF в allergologiya / pulmonologiya).

На текущем GCE без PR title ещё «заменить словами…»; после merge - имя из filename.

Вторично: аллергический ринит д-нас (пост. 38) - отдельный КП при фокусе на рините.

## Следующая команда

```bash
# после merge
bash deploy/gcp-app/deploy_to_gce.sh
# smoke: case review J45 pediatric - нет B_icd_name_weak_match; виден «Разбор по критериям»
```

## Не трогать параллельно

`clinical_knowledge/mo_icd_name_match.py`, `clinical_text_similarity.py`, `frontend/web/shared/mo-app.js`
