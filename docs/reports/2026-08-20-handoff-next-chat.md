# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-20  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `bf1e26e` (#167)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Primary: `https://protocol.kravira.by`  
Прод runtime: **`2026-08-20-061113Z-mo-grade-ui`**.

---

## Сделано

- Волна 0: PR #166 merge `3bf14f3`, deploy. JSON зон содержит `overall_grade`.
- Волны 1-2: PR #167 merge `bf1e26e`, deploy. Чип/фильтр/колонка Оценка.
- Backfill склада: 9736 clinical 26.07-19.08. unmatched+bad план **3946 → 0**.
  Оценки: Важно 228, Слабо 2060, С замечанием 7075, Хорошо 373.
- Rceth shadow не двигает итог (contra → всё ещё good, пока primary выключен).
- `kp-eval-full` жив. Rceth `done`.

## Делается

`kp-eval-full` пишет `kp_suggest_eval_post165.json` (файла ещё нет).

## Нужно

1. Методист: 20 карточек (4/4/8/4) - слово vs зоны на той же карточке.
2. Волна 3: калибровка Rceth 30 кейсов. **Не** включать `MO_RCETH_LABEL_PRIMARY`.
3. Не деплоить повторно без новой причины.

## Запрет

- Второй full Rceth parse / `RCETH_PARSE_FORCE=1`
- Gemini с Mac, push в `main`, грязный checkout, PHI
