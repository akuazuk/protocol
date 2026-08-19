# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-19  
Репозиторий: `akuazuk/protocol`  
Предшественник: `docs/reports/2026-08-18-handoff-next-chat.md`  
Канон git на старте задачи: `origin/main` `9b8f09e` (#157)  
Worktree: `/private/tmp/protocol-task-rceth-identity-d-pc1`  
Ветка: `cursor/rceth-identity-d-pc1`  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Primary: `https://protocol.kravira.by`

---

## Сделано

- Rceth full parse на GCE **done** 2026-08-18 12:24 UTC: 3017/3017, fail 0.
- 4.1+4.3 есть у 2234/3017 (74%, цель ≥70%). Строгий `parse_ok` 34% из-за пустого 4.2.
- Ночь 2026-08-18/19: KP sync OK, night extract/score OK, LLM backfill отработал.
- Шаг D: identity бренд → МНН / форма в `drug_normalizer` (seed 20 + runtime манифест, setdefault, регрессия мелоксикам).

## Делается

Нет живого Rceth job. Watchdog idle (`status=done no restart`). Weekly cron выключен.

## Нужно

1. Довести PR шага D до merge. После merge координатор: `deploy_to_gce.sh` + smoke `/api/version`, чтобы на GCE подхватился полный манифест.
2. Шаг F: shadow findings (`off_label_vs_dx`, противопоказания, возраст) - отдельная ветка, не смешивать с КП accuracy.
3. Weekly cron Rceth - только после калибровки F.
4. КП hit 71.2% → 75% (омнибус ЛОР 2017) - отдельный PR.
5. Render не удалять. Чужие PR (#158, #148, stale) не брать.

## Запрет

- Второй full parse / `RCETH_PARSE_FORCE=1`
- Gemini / `grade_kz_llm` с Mac
- push в `main`
- работа в грязном checkout
- печать PHI
