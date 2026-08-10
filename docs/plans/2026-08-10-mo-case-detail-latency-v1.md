# План: ускорение «Разбор случая» (минимальный пакет latency)

Статус: **active**  
Дата: 2026-08-10  
Профиль: визит `3600047` на GCE - drawer ~5+ с из-за live concordance, CSV prior и cold protocol suggest.

## Цель

Первый ответ `GET /cases/{id}` с оценкой/findings **&lt;500 ms**.  
Протоколы МЗ - отдельным async-запросом, после прогрева каталога **&lt;100 ms**.

## Что изменено

1. **Live concordance / ICD / clinical gaps** - по умолчанию `auto`: не гонять, если в warehouse уже есть findings; `?live=1` принудительно.
2. **Prior clinical (до 90 дней CSV)** - по умолчанию выкл (`MO_CASE_DETAIL_PRIOR=0`); рубрика МЗ без prior; `?prior=1` включает.
3. **Zones live path** - больше не вызывает prior+suggest синхронно в drawer; КП остаётся на `/protocol-suggest`.
4. **Protocol suggest** - `attach_history=0` по умолчанию; прогрев registry+match при старте (`MO_PREWARM_PROTOCOL_SUGGEST=1`).

## Метрики

| Метрика | Было (3600047) | Цель |
|--|--|--|
| detail API (warehouse findings) | ~1.5-2+ с (live+prior) | &lt;500 ms |
| suggest cold | ~3.3 с | &lt;100 ms после prewarm |
| suggest warm | ~30 ms | держать |

## Риски

- Без live на пустых findings всё ещё считаем live (`auto`).
- Динамика рубрики МЗ без prior слабее до `?prior=1` / индекса склада.
- Зоны плана без sync-suggest могут чуть слабее на кейсах без warehouse zones.

## Следующее

Индекс prior в warehouse; опциональный background enrich в UI.
