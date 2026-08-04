# Archive

Исторические и разовые материалы, которые не нужны в ежедневной навигации
по разделам продукта. Код production и активные docs остаются вне `archive/`.

| Путь | Что внутри |
|------|------------|
| `archive/docs/konkurs/` | Пакет заявки/бизнес-плана (Belinfund и смежное) |
| `archive/ml-experiments/` | Исторические batch/eval прогоны (`REPORT.md`, summary JSON) |

Правила:

1. Новые ежедневные batch-прогоны пишите в `ml/experiments/` (рабочая зона).
2. Локальные тяжёлые артефакты (`output/`, `corpus_vector_index/`, PDF КЗ) -
   в `.local-archive/` на диске машины, не в git (`gitignore`).
3. Не возвращайте archive в `docs/` / `ml/experiments/` без явной причины и плана.

План: `docs/plans/2026-08-04-repo-sections-archive-v2.md`.
