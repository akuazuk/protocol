# Handoff для нового чата / вкладки Cursor

Преемник: [2026-08-20-handoff-next-chat.md](2026-08-20-handoff-next-chat.md)

Дата: 2026-08-19  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `fae7266` (#163)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Worktree этой сессии: `/private/tmp/protocol-task-kp-eval-handoff-pc1`  
ветка `cursor/kp-eval-handoff-pc1`.

Primary: `https://protocol.kravira.by`  
Прод runtime всё ещё **`2026-08-14-123546Z-kp-golden-40`**. В `main` уже #159/#161/#162/#163, на GCE UI их нет.

---

## Сделано

- Rceth parse GCE **done** (2026-08-18 12:24 UTC); #159 identity, #161 shadow, #162 омнибус ЛОР, #163 child-only метрика - в `main`.
- #163 влит: https://github.com/akuazuk/protocol/pull/163 (`fae7266`).
- Eval sample 500 без рестарта UI: hit **70.2%**, omnibus **16** (4.6%), adult child-only **0**. Отчёт `/var/data/medical_exams/reports/kp_suggest_eval_sample_post163.json`.
- В этой ветке: `looks_omnibus` на ЛОР №78 (2026) и общехирургию 2007. Гин. 2018 №17 не демоутить.

## Делается

Нет живого eval-контейнера после sample (exit 0). Полный CSV - после merge этого PR.

## Нужно

1. Дождаться sample 500, затем полный CSV тем же one-off (не `docker exec protocol-web` - там старый код).
2. Следующий код КП: гинекология. КП 2018 №17 - dump 276 МКБ, 216 top-1; ревизия 18.06.2026 №73 есть в каталоге, **ICD пустой**, в content-index нет. Пустые N80 (101) и D25 (93) - кандидаты на №73, не на омнибус 2018.
3. Координатор: `bash deploy/gcp-app/deploy_to_gce.sh`. После выкладки `/api/version` = `2026-08-19-102831Z-kp-child-only-metric` (или новее). Потом калибровка Rceth 30 кейсов. Не `MO_RCETH_LABEL_PRIMARY`, не weekly Rceth cron.
4. Render не удалять. Чужие PR (#158, #148, stale) не брать.

## Запрет

- Второй full Rceth parse / `RCETH_PARSE_FORCE=1`
- Рестарт `protocol-web` ради eval (one-off достаточно)
- Gemini с Mac, push в `main`, грязный checkout, PHI
