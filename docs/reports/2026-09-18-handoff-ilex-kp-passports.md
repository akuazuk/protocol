# Handoff: Ilex-паспорта КП (шаг 1)

Дата: 2026-09-18

## Repo

- worktree: `/private/tmp/protocol-task-ilex-kp-passports-pc1`
- branch: `cursor/ilex-kp-passports-agent1-pc1`
- base: `origin/main` `39c34ed9`

## Сделано

- Парсер локальной выгрузки Ilex → `output/registry/ilex_protocol_passports.jsonl`
  (415 строк, без HTML). МКБ режется по каждому внутреннему КП (ФП = I48, не коды соседей).
- Overlay на `protocol_cards`: только год+номер **и** пересечение нозологии с путём/названием.
  1273 карт получили нормальное имя вместо `КЛИНИЧЕСКИЙ ПРОТОКОЛ`.
- Запрос suggest дополняется названием паспорта Ilex (`enrich_diagnosis_with_ilex`).
- Алиас гипертон/гипертенз → в т.ч. «кровяным давлением» (корпус 2017).
- Выключить: `ILEX_PASSPORTS=0`. HTML Ilex в git нет.

## Проверка (было / стало)

| | off | on |
|--|--|--|
| Golden 40 | 0 fail | 0 fail |
| Нозологии АГ/бронхит/синусит/ФП в suggest | 3/4 | 3/4 |
| Прямой паспорт Ilex | - | **4/4** |
| Чужие ГСК/ПЦД/экстренка на этих 4 | 0 | 0 |

ФП 2026 (пост. N 34) и АГ 2026 (пост. N 38) есть в Ilex и нет как PDF в картах.
Suggest поэтому остаётся на КП 2017.

## Не сделано

- Merge / deploy / GCE.
- Nightly sync Ilex.
- Plan-score по главам.

## Не трогать параллельно

`clinical_knowledge/ilex_protocol_passports.py`, `loader.py`, `protocol_match.py`,
`dx_query_expand.py`, `output/registry/ilex_protocol_passports.jsonl`.
`rag_server.py` - только `BUILD_VERSION` (файл занят #186/#113).

## Следующая команда

Добор PDF 2026 N 34 и N 38 в корпус МЗ, затем повтор
`PYTHONPATH=. python3 scripts/eval_ilex_kp_suggest.py`.
Не деплоить с этой task-ветки.
