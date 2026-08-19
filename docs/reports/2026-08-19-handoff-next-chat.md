# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-19  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `dd41297` (#162)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Primary: `https://protocol.kravira.by`  
Прод runtime всё ещё старый (`2026-08-14-123546Z-kp-golden-40`), пока координатор не выложит.

---

## Сделано

- Rceth parse GCE **done**; #159 identity и #161 shadow findings влиты в `main` (`34e2f72`, `43ab48f`).
- #162 омнибус ЛОР: не clinical по dump МКБ (`dd41297`).
- Метрика «6 дет_нас у взрослых»: скорее `взр_и_дет_население` (подстрока `дет_нас`). Правка в этой ветке.

## Делается

Нет живого Rceth job. Watchdog idle. Weekly cron выключен.

## Нужно

1. Координатор: `deploy_to_gce.sh`. `/api/version` сначала `2026-08-19-055149Z-rceth-label-shadow`, после этого PR - новая версия.
2. После деплоя: CSV KP eval на GCE (цель hit ≥75%, omnibus ≤5%).
3. Калибровка Rceth 30 кейсов. Не `MO_RCETH_LABEL_PRIMARY`, не weekly cron.
4. Render не удалять. Чужие PR не брать.

## Запрет

- Второй full parse / `RCETH_PARSE_FORCE=1`
- Gemini с Mac, push в `main`, грязный checkout, PHI
