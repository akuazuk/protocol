# Handoff: синхронизация протокола и зоны плана

Дата: 2026-09-06.
Branch: `cursor/mo-protocol-zone-sync-agent1-pc1`.
Worktree: `/private/tmp/protocol-task-mo-protocol-zone-sync-pc1`.
Base: `f756dc25e190ba27f6b6ac89fc823a1505cec57b`.

## Ошибка

Верхняя карточка CASE Review могла показывать `План по протоколу:
протокол не подобран`, хотя ниже асинхронный подбор уже показывал подходящий
протокол. Причина: initial detail использовал persisted warehouse zones, а
отдельный `/protocol-suggest` обновлял только нижний список и объект protocol в
памяти, но не пересчитывал и не перерисовывал zones/brief.

## Реализация

- `/protocol-suggest` после подбора пересчитывает зоны тем же
  `compute_mo_zone_scores()` с фактическим `protocol_suggest`.
- Ответ возвращает согласованные `zones` и `review_brief`.
- Drawer применяет результат только если открыт тот же case id.
- Верхняя оценка, критерии и итог разбора обновляются атомарно из одного ответа.
- После замены DOM восстанавливаются действия zone cards и brief prefill.
- Если пересчёт зон недоступен, сам список предложенных протоколов остаётся
  доступным и получает явный `zones_refresh` error.
- Формулы, пороги, веса и primary-флаги не менялись.

## Проверки

- `node --check frontend/web/shared/mo-app.js` - passed.
- `python3.11 -m py_compile rag_server.py` - passed.
- `pytest tests/test_mo_zone_scores.py tests/test_mo_zone_api.py
  tests/test_mo_case_review_brief.py tests/test_mo_frontend_structure.py -q`
  - 29 passed.
- IDE lint и `git diff --check` - passed.

Постоянный stale-case browser scenario публикуется отдельным level-4 PR.

## Production

Не deployed. После merge и test-only PR нужен exact-main GCE release. Smoke:
открыть synthetic case с matched protocol и проверить одинаковый `kp_status`
в верхней zone card, criteria, review brief и нижнем protocol list.

## Следующий шаг

Добавить browser acceptance для delayed suggest при быстром Next/Previous и
matched/unmatched zone refresh.

## Не менять параллельно

- `rag_server.py`;
- `frontend/web/shared/mo-app.js`;
- этот handoff до merge.
