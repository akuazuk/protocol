# План: топ-50 худших КЗ + L2 + Gemini (v1)

- Дата: 2026-07-21
- Статус: active
- Преемник для: `2026-07-21-mis-kz-l1-batch-v1.md` (архив по индексу)

## 1. Контекст

Нужен методический разбор самых слабых визитов июля:
топ-50 по L1 overall (все врачи), с `patient_id` / `visit_id` / датой,
рядом L2-оценка и комментарий, плюс выборочный прогон через Gemini
(доступная max: `gemini-2.5-pro`; «3.6» в API нет).

## 2. Что меняем

| Артефакт | Назначение |
|----------|------------|
| `scripts/run_mis_protocol_l1_batch.py` | worst_visits=50 overall + patient_id + `--enrich-l2-worst` |
| `/var/data/.../kz_l1_*_summary.json` | L1+L2 поля в worst_visits |
| `/var/data/.../kz_l1_*_gemini_reviews.json` | Результаты выборочного Gemini |
| `GET /api/methodist/mis-kz-quality` | worst_visits + gemini_reviews |
| `POST .../mis-kz-quality/gemini-review` | Прогон выбранных visit_id |
| UI вкладка + `mis-kz-quality.html` | Чекбоксы, колонки L1/L2, таблица Gemini |

## 3. Метрики

| Показатель | Цель | Факт |
|------------|------|------|
| worst_visits | 50 overall | **50** |
| L2 coverage | 50/50 | **50/50** |
| Gemini model | ≥3.6 если есть, иначе max | **gemini-2.5-pro** (3.6 нет) |
| patient_id в таблице | да | **да** |

## 4. Шаги

- [x] Summary: топ-50 + patient_id
- [x] L2 enrichment на Render
- [x] API Gemini + UI
- [x] Пуш

## 5. Риски

- `patient_id` в summary - ПДн; полный файл на `/var/data`, в git - только при необходимости UI-fallback.
- L2 на Render = fast (без synthesize); Gemini - отдельный вызов methodist model.
- OOM при параллельном L2 - workers=1.
