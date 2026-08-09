# Кабинет методиста: стиль МО + полная ширина (v1)

Дата: 2026-08-09
Статус: **active**

## Контекст

Классический кабинет (`/methodist/*`, `doctor/index.html` + `body.methodist-mode`) остался
на узком shell (1120px), своих зелёных кнопках и doctor-chrome («Найти протокол / Пациентам»).
МО Аналитика уже на `mo-tokens` / `mo-ui`. Нужно выровнять кабинет визуально и убрать мусор
(битые relative-ссылки, лишние B2C/онко вкладки, футер-карточки).

## Цель

1. Полная ширина рабочей области (до ~1520-1680px).
2. Кнопки, поля, баннер, login - язык МО (`mo-tokens` + `methodist-cabinet.css`).
3. Убрать doctor-chrome в mode методиста; навигация - короткие вкладки кабинета.
4. Починить absolute paths (`/patient.html`, `/onco-risk.html`, `/docs/...`).
5. Скрыть ненужное: B2C/онко из primary nav, футер-карточки, битые docs-ссылки.

## Шаги

- [x] План v1
- [x] CSS `methodist-cabinet.css` + wire routes
- [x] HTML/JS: вкладки, ссылки, скрытие clutter
- [x] Тесты + bump + PR
- [ ] GCE deploy после merge

## Риски

| Риск | Митигация |
|--|--|
| Конфликт inline CSS index.html | Overrides только под `body.methodist-mode` |
| Сломать doctor UI | Не трогать стили вне methodist-mode |
| Нужны B2C/онко иногда | Панели в DOM; ссылки absolute, не в primary tabs |
