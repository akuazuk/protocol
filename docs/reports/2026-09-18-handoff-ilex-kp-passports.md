# Handoff: Ilex-паспорта КП (шаг 2)

Дата: 2026-09-18

## Repo

- worktree: `/private/tmp/protocol-task-ilex-kp-passports-pc1`
- branch: `cursor/ilex-kp-passports-agent1-pc1`
- base: `origin/main` `39c34ed9`
- PR: https://github.com/akuazuk/protocol/pull/250
- `BUILD_VERSION`: `2026-09-18-181209Z-ilex-kp-2026-pdfs`

## Сделано

- PDF с сайта МЗ (рубрика кровообращения): АГ 2026 N 38; N 34 как КП1-КП6
  (ФП = КП4). Карты: 5121 → 5193. HTML Ilex в git нет.
- Overlay: unique-PDF shortcut, если в имени файла есть год и номер
  (иначе «АГ» не стыкуется с «гипертензией»).
- Suggest: recency на текстовом пути МО; в запрос Ilex только top-1 паспорт;
  при равном балле новее раньше. АГ 2026 и ФП 2026 стали top-1.

## Проверка (шаг 2)

| | overlay off | overlay on |
|--|--|--|
| Golden 40 | 0 fail | 0 fail |
| Нозологии АГ/бронхит/синусит/ФП | 3/4 | **4/4** |
| Чужие ГСК/ПЦД/экстренка на этих 4 | 0 | 0 |

Без overlay АГ 2026 всё равно в топе, но имя файла «АГ» не проходит
`expect_any` (гипертенз/гипертон/давлен) - паспорт нужен для названия.

## Не сделано

- Merge / deploy / GCE.
- Nightly sync Ilex.
- Остальные PDF 2026.
- Plan-score по главам.
- Не коммитить усечённый `output/chunks/chunks.jsonl` (локальный
  changed-only сбой, не полный корпус).

## Не трогать параллельно

`clinical_knowledge/ilex_protocol_passports.py`, `loader.py`,
`protocol_match.py`, `case_protocol_suggest.py`, `dx_query_expand.py`,
`output/registry/protocol_cards.jsonl`,
`output/registry/ilex_protocol_passports.jsonl`.
`rag_server.py` - только `BUILD_VERSION` (файл занят #186/#113).

## Следующая команда

Review/merge PR #250. Деплой только координатором после merge:
`bash deploy/gcp-app/deploy_to_gce.sh`.
