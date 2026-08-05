# Handoff: MO concordance E3 (UI / очередь)

Дата: 2026-08-05  
Ветка: `codex/mo-concordance-smirnova-agent1-pc1`  
План: `docs/plans/2026-08-05-mo-eval-smirnova-concordance-v1.md`

## Что сделано

1. Русские title для 6 concordance codes в `mo_finding_labels_ru.py` (+ upsert в `dim_finding`).
2. Warehouse пишет `deep.shadow_findings` в `fact_mo_finding` с `is_shadow=1`, `linked_fields_json`, `link_hint_ru` - **без** смены primary overall / axes.
3. Очередь «Вчера» (`action_cases`) включает P0/P1 shadow; UI badge `shadow`.
4. Case detail: highlight связанных клинических полей + кнопки перехода из finding; live merge concordance при наличии исходного текста (даже до re-pipeline дня).
5. L1 batch сохраняет `shadow_findings` в `case.deep`.

## Что нужно для прод-очереди

Перескорить/upsert дни после деплоя (или дождаться daily), иначе в warehouse нет shadow rows. Case detail уже может показать shadow live.

`MO_CONCORDANCE_PRIMARY` по-прежнему **off**.

## Дальше

E4 - рубрика МЗ / suggest soft finding.
