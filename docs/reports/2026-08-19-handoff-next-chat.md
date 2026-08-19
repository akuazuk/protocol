# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-19  
Репозиторий: `akuazuk/protocol`  
Предшественник: `docs/reports/2026-08-18-handoff-next-chat.md`  
Worktree: `/private/tmp/protocol-task-rceth-identity-d-pc1`  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Primary: `https://protocol.kravira.by`

---

## Сделано

- Rceth full parse на GCE **done** 2026-08-18 12:24 UTC: 3017/3017, fail 0; 4.1+4.3 = 74%.
- Шаг D: identity бренд → МНН. PR **#159** (lint fix `cefd1f3`, unused import).
- Шаг F: shadow findings `C_rceth_off_label` / `C_rceth_contraindication` / `C_rceth_age_outside_label` (P2, не в overall, не в очередь). Ветка `cursor/rceth-label-shadow-f-pc1`.

## Делается

Нет живого Rceth job. Watchdog idle. Weekly cron выключен.

## Нужно

1. Merge **#159**, затем PR шага F (база - ветка D, пока #159 не влит).
2. После merge F координатор: `deploy_to_gce.sh` + `/api/version`. На GCE подхватится полный манифест и labels.
3. Калибровка 30 кейсов (цель FP off-label < 15%). Пока не primary и не weekly cron.
4. КП hit 71.2% → 75% - отдельный PR.
5. Render не удалять. Чужие PR не брать.

## Запрет

- Второй full parse / `RCETH_PARSE_FORCE=1`
- `MO_RCETH_LABEL_PRIMARY=1` до калибровки
- Gemini с Mac, push в `main`, грязный checkout, PHI
