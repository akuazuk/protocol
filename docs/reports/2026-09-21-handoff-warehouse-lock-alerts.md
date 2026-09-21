# Handoff: письма Alert recovered и lock витрины

Дата: 2026-09-21
Репозиторий: `origin/main`
Прод: `https://protocol.kravira.by`
SHA: `8000354f96f732e4ade2239689f27cbf2ec6acfe`
`BUILD_VERSION`: `2026-09-21-184253Z-warehouse-lock-alerts`
PR: [#292](https://github.com/akuazuk/protocol/pull/292) merged, GCE `PUBLIC_OK`

## Почему приходили письма

Политика `Protocol: всплеск ошибок приложения` открывается при >10 ошибок за 5 минут
и закрывается, когда счётчик снова ниже порога (в письме value 9 при пороге 10).
GCP по умолчанию шлёт и OPENED, и CLOSED. Письмо **Alert recovered** / «No severity» -
это закрытие инцидента, не новая авария.

Настоящий всплеск был из логина: пересчёт витрины держал SQLite, пока на каждый
случай крутился live-подбор КП. Логин на каждый запрос делал DDL и через ~5 с
отдавал `database is locked` / 500. Старая метрика считала traceback и access-log
` 500 `, поэтому один сбой выглядел как 6-8 ошибок.

## Что уже в GCP и в проде

- Метрика `protocol_app_errors`: `logName:"protocol_web" AND jsonPayload.log:"ERROR protocol.rag"`.
- Политика `3428894496387672037`: `notificationPrompts: [OPENED]` - recovered на почту не уходит.
- Код [#292](https://github.com/akuazuk/protocol/pull/292): схема витрины один раз за процесс,
  `busy_timeout` 30 с, suggest до записи, `commit` после каждого случая.
- Smoke: `/health/live` ok, `/api/version` совпал с SHA/версией, логин admin 200 за 1,8 с
  пока внутри `protocol-web` шёл `recompute_mo_days.py`.

## Склад КП

Пересчёт 01-17 сентября был убит деплоем контейнера. 2026-09-01 уже с matched=200.
После деплоя заново запущен только хвост:

`scripts/recompute_mo_days.py --first-date 2026-09-02 --last-date 2026-09-17 --skip-reports`

Второй writer не запускать. `MO_LAB_IN_PRIMARY` не включать.

## Нельзя трогать параллельно

`clinical_knowledge/mo_daily.py`, `clinical_knowledge/mo_backend.py`,
`clinical_knowledge/mo_app_accounts.py`, витрина
`/var/data/medical_exams/warehouse/mo_analytics.sqlite`.

## Следующий шаг

Дождаться flush дней 02-17 и сверить `zone2b_kp_status=matched` по `fact_mo_case`.
Не деплоить и не рестартовать `protocol-web`, пока этот recompute жив.
