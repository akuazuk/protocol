# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-26  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `9e4e532` (#176)  
Грязный `/Users/pavelkuzauka/Cursor_Folders/Protocol` на `main` не использовать
(отстаёт, локальные PDF/планы).

Worktree этой сессии: `/private/tmp/protocol-task-mo-kp-mentions-deploy-pc1`  
ветка `cursor/mo-kp-mentions-deploy-pc1`.

Primary: `https://protocol.kravira.by`  
Прод: **`2026-08-21-093740Z-kp-icd-mentions`** (smoke `/health/live` ok).

---

## Сделано

- `kp-eval-full` post174: exit 0, 2026-08-21 11:46 UTC, 18427 с.
  7605 clinical, hit **41.9%**, омнибус **0**, adult+child **0**,
  ПЦД / ГСК / экстренка / нейрохирургия / миелома не в топ-25.
- Влит #176 (`9e4e532`): `icd10_mentions`, Y/W/V/T не в suggest/индексе,
  каталог поверх реестра не кладём.
- Sample 300 post176 (one-off, код с `/tmp/protocol-eval-176`):
  hit **40.3%** (121/300), омнибус **0**, adult+child **0**, жалобы не ищут
  (query: diagnosis 296 / icd 1 / none 3). Чужие файлы из цели = 0.
- UI выложен. `protocol_cards` 6610. `MO_RCETH_LABEL_PRIMARY` не включали.

## Делается

Нет живого eval-контейнера.

## Нужно

1. Ночной score новым матчером - на складе КП всё ещё unmatched, «Хорошо» ~3%.
2. Калибровка Rceth 30 кейсов shadow. Не `MO_RCETH_LABEL_PRIMARY`, не weekly cron.
3. Docs PR этой ветки: план + этот handoff. Чужие PR (#158, #148, #168 stale) не брать.

## Запрет

- Второй full Rceth parse / `RCETH_PARSE_FORCE=1`
- Gemini с Mac, push в `main`, грязный checkout, PHI
- `deploy_to_gce.sh` с Mac SSH `pavelkuzauka`: chown `.env.gcp-public` на cron-user
  `pavel` ломает `assemble_web_env_from_sm.sh` (Permission denied). Обход:
  `sudo bash /opt/protocol/deploy/gcp-app/assemble_web_env_from_sm.sh`, затем
  docker build + run. Либо выставить `GCE_OPS_USER` на SSH-логин только для чтения
  public env, cron-owner потом вернуть на `pavel`.
