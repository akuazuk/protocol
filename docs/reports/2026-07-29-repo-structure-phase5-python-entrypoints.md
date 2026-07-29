# Phase 5 report: canonical Python entrypoints

Дата: 2026-07-29  
Ветка: `codex/main-sync`

## Что сделано

- Добавлены каноничные Python-entrypoints:
  - `scripts/data/py/*` (data/corpus utilities);
  - `scripts/dev/py/*` (dev/diagnostic utilities).
- Все entrypoints используют `runpy.run_path(...)` и запускают существующие
  top-level `*.py` без изменения их логики.
- Legacy top-level пути сохранены - обратная совместимость не нарушена.
- `scripts/ops/smoke_repo_layout.sh` расширен проверкой `py_compile` для новых
  python-wrapper'ов.

## Зачем так

Это безопасный переходный шаг перед возможным физическим переносом кода из root:

- команды уже структурированы по доменам;
- текущие интеграции и документация не ломаются;
- риск регрессий минимальный.
