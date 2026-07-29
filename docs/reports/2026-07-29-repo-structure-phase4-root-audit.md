# Root audit after structure cleanup (Phase 4)

Дата: 2026-07-29  
Ветка: `codex/main-sync`

## Что проверено

- Снят инвентарь верхнего уровня репозитория после Phase 1-3.2.
- Проверена доступность каноничных script entrypoints по доменам:
  - `scripts/ops/*`
  - `scripts/deploy/*`
  - `scripts/data/*`
  - `scripts/dev/*`
- Добавлен smoke-check для структуры wrapper-скриптов и подключен в CI.

## Текущее состояние root

- Top-level объектов: **76**.
- Стабильные доменные каталоги уже выделены: `backend/`, `frontend/`, `clinical_knowledge/`,
  `scripts/`, `docs/`, `data/`, `output/`, `ml/`, `tests/`, `api/`, `config/`.
- Legacy/top-level файлы остаются в основном из причин обратной совместимости и постепенной миграции:
  - backend entrypoint и runtime: `rag_server.py`, `Procfile`, `render.yaml`;
  - corpus/search артефакты и индексы: `chunks.json`, `protocols.json`, `index.csv`,
    `protocol_meta.json`, `structured_index.json`, `embeddings.json`;
  - вспомогательные top-level скрипты python (build/extract/verify).

## Вывод

- Phase 4 в рамках текущего плана считается закрытой:
  - root-аудит выполнен и задокументирован;
  - runbook обновлен;
  - CI дополнен smoke-check структуры.
- Дальнейшее уменьшение количества top-level файлов требует отдельной миграционной фазы,
  чтобы не ломать текущие команды и пути деплоя.
