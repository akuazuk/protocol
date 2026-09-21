# Handoff: волны D0-K3 и U1-U3 на GCE

Дата: 2026-09-21
Репозиторий: `origin/main`
Прод: `https://protocol.kravira.by`
SHA: `c0c84ac1dd5d955cf6d89d9ce43bd1ba71f8a187`
`BUILD_VERSION`: `2026-09-21-155929Z-mo-u3-trend-clear-band`
План: `docs/plans/2026-09-21-mo-visual-and-kp-v1.md` (черновик ещё в открытом docs PR #280)

## Сделано и на проде (merge + GCE, `/api/version` совпал)

| Волна | PR | SHA | Версия | Приёмка на проде |
|--|--|--|--|--|
| D0 спойлер решения свёрнут | #281 | `ae333752` | `2026-09-21-111645Z-mo-d0-dock-collapsed` | док закрыт на 3650914 |
| K1 persist КП в склад | #282 | `d1ef1755` | `2026-09-21-113944Z-mo-k1-warehouse-kp` | `protocol_id` пишется при hit |
| K2 Ilex overlay в образе | #283 | `02581f28` | `2026-09-21-120051Z-mo-k2-ilex-image` | `ilex_passports=415` в образе |
| K3 честный KPI плана | #284 | `1be4c4fa` | `2026-09-21-124004Z-mo-k3-plan-kpi` | подпись «не сравнивался с КП», не «0% = все хорошие» |
| U1 кольца Обзора | #285 | `12b2afef` | `2026-09-21-131023Z-mo-u1-overview-visual` | 3 кольца, легенда 6 пунктов, без серии №55 |
| U2 плотность разбора | #286 | `58ff9890` | `2026-09-21-142319Z-mo-u2-case-density` | findings свёрнуты, evidence закрыт, 720 без гор. скролла |
| U2b полоса протокола | #287 | `65ebf6ff` | `2026-09-21-145030Z-mo-u2-protocol-strip` | на первом экране только имя КП; concordance в аккордеоне |
| U3 клик графика = Найти МО | #288 | `244a8384` | `2026-09-21-151523Z-mo-u3-chart-click` | сегмент кольца ставит чипы зоны/полосы |
| U3b leftover band | #289 | `c0c84ac1` | `2026-09-21-155929Z-mo-u3-trend-clear-band` | после кольца «в норме» клик тренда План за 2026-09-20: `zone=zone2b`, без `zone_band` |

Smoke U3b (после деплоя `c0c84ac1`):

- кольцо Оформление / Хорошо → `page=documents&zone=zone1&zone_band=ok`, чип «Оформление · в норме»;
- затем точка тренда «План» за 2026-09-20 → `page=documents&zone=zone2b`, период custom 2026-09-20, чип «План по протоколу» без leftover «в норме».

## Склад КП (K3 данные)

После пересчёта 2026-09-18, 2026-09-19 и 2026-09-20 (`fact_mo_case`):

| Дата | clinical | с `protocol_id` | `matched` | zone2b bad/weak/na |
|--|--|--|--|--|
| 2026-09-18 | 458 | 157 | 157 | - |
| 2026-09-19 | 253 | 112 | 112 | - |
| 2026-09-20 | 158 | 52 | 52 | 40 / 12 / 106 |

Пересчёт 2026-09-20 внутри `protocol-web` завершился (writer больше не жив). Остальные дни сентября не пересчитаны: месяц на Обзоре по-прежнему в основном «план не сравнивался с КП».

## Не сделано

- Полный пересчёт сентября 2026-09-01..09-20 (~11 ч с live suggest).
- Merge docs PR #280 (план; ветка отстала от main, force-push нельзя).
- Строка в `docs/plans/README.md` (файл занят чужими PR).
- `MO_LAB_IN_PRIMARY` не включали.
- Trust A/B не ослабляли.

## Нельзя параллельно

`frontend/web/shared/mo-app.js`, `frontend/web/shared/mo-charts.js`, `frontend/web/shared/mo-ui.css`, night persist `protocol_id` / `zone2b_kp_status`, `docs/plans/README.md`, `rag_server.py` кроме `BUILD_VERSION`.

## Следующая безопасная команда

Опциональный пересчёт 2026-09-01..09-17, только когда кабинет не смоукают:

```bash
docker exec protocol-web python3 scripts/recompute_mo_days.py \
  --first-date 2026-09-01 --last-date 2026-09-17 \
  --data-root /var/data/medical_exams --skip-reports
```

Не включать `MO_LAB_IN_PRIMARY`. Не качать Rceth/Ilex с Mac. Docs PR #280 - rebase на main без force-push или закрыть и открыть новую ветку плана.
