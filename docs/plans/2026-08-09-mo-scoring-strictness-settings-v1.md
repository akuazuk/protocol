# МО Аналитика: настройка жёсткости оценок и пересчёт (v1)

Дата: 2026-08-09
Статус: **active**

## Контекст

В «Настройках» МО есть справка по зонам, но нет ручек жёсткости полос
(`bad` / `weak` / `ok`) и порогов overall. Пересчёт витрины - только CLI
(`recompute_mo_days.py`). Нужен кабинетный контур: сохранить профиль →
пересчитать диапазон сразу или на следующей подгрузке данных.

## Что меняем в проде

1. JSON-профиль `/var/data/medical_exams/config/mo_scoring_profile.json`
   (пресеты soft / standard / strict + custom).
2. GET/PUT `/api/methodist/mo/scoring-config`, POST `…/recompute`.
3. Карточка в UI «Настройки» с пресетом, порогами зон/статусов и действиями
   пересчёта.
4. Хук `apply_on_next_load` в `mo_llm_range_runner.sh` / `score_inbound_day.sh`.

Официальный LLM-grade и Gemini night не трогаем; пересчёт - warehouse/zones
из уже сохранённых артефактов (без МИС).

## Метрики

| | Было | Цель |
|--|--|--|
| Editable scoring knobs in Settings | 0 | ≥4 (preset, bad_below, ok_at, status good/acc) |
| Recompute from UI | нет | период / всё / next load |
| Tests | - | unit profile + API contract |

## Шаги

- [x] План
- [x] `mo_scoring_profile.py` + wire bands/v4/deep/shadow
- [x] API + background recompute job
- [x] Settings UI
- [x] Pipeline hook next-load
- [x] Tests, bump, PR

## Риски

- Долгий recompute на большом диапазоне блокирует worker - ограничиваем
  sync-диапазон и пишем job status.
- Смена risk_caps без deep-rescore не меняет уже записанный `overall_pct`
  в cases.jsonl; UI явно разделяет «зоны/полосы» (recompute) и
  «полный deep rescore» (опционально, lead/admin).
