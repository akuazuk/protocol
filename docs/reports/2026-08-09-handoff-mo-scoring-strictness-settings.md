# Инструкция следующему агенту: жёсткость оценок МО + хвосты сессии

Дата: 2026-08-09 (UTC ~19:53Z)  
Компьютер уходит в сон сразу после этой записи.

## Repo / prod (канон)

| | |
|--|--|
| `origin/main` | `912df921` (после #117/#118/#119) |
| Prod GCE | `https://protocol.kravira.by` |
| `BUILD_VERSION` | `2026-08-09-194453Z-scoring-plan-status` |
| План | `docs/plans/2026-08-09-mo-scoring-strictness-settings-v1.md` (**completed**) |
| Checkout Mac | грязный `Protocol/` не трогать; новый worktree от `origin/main` |

Preflight: `AGENTS.md` + `git fetch` + clean worktree через `scripts/ops/git_task_start.sh`.

---

## Что сделано (не переделывать)

### A. Жёсткость оценок в Настройках МО

- PR: [#117](https://github.com/akuazuk/protocol/pull/117) feature, [#118](https://github.com/akuazuk/protocol/pull/118) polish, [#119](https://github.com/akuazuk/protocol/pull/119) plan status.
- Профиль на диске данных: `/var/data/medical_exams/config/mo_scoring_profile.json`
- Job status: `/var/data/medical_exams/config/mo_recompute_job.json`
- API: `GET/PUT /api/methodist/mo/scoring-config`, `POST /api/methodist/mo/recompute`
- UI: `/methodist/mo` → Настройки → «Жёсткость оценок» (пресеты soft/standard/strict, период/всё/next-load, poll job)
- Wire: zone bands, deep status thresholds, v4 risk caps, shadow cutoffs, queue `attention_score_below`
- Pipeline: `scripts/mo_apply_scoring_profile_on_load.py` в `mo_llm_range_runner.sh` и `score_inbound_day.sh`
- GCE smoke уже был: save `strict` → recompute `2026-08-06` (447 cases ok) → restore `standard`; `last_applied` выровнен

### B. Ранее в той же длинной сессии (shadow option B)

- PR [#114](https://github.com/akuazuk/protocol/pull/114) + handoffs #115/#116
- Shadow Dx/Plan conservative в проде; SSOT/official overall **не** меняли
- Живой LLM shadow smoke упёрся в **Gemini monthly spend cap** на оба ключа (`GOOGLE_API_KEY` / `_2`)

---

## Что надо ещё (приоритет)

1. **Ручной UI smoke жёсткости** (не сделан глазами методиста):  
   войти в `/methodist/mo` → Настройки → сменить пресет → «Пересчитать период» на 1 день → убедиться, что poll доходит до `done` и полосы зон в таблице/очереди меняются ожидаемо. Потом вернуть `standard`, если тест был на `strict`.

2. **Gemini spend cap** (блокер night LLM / shadow backfill):  
   поднять лимит в [ai.studio/spend](https://ai.studio/spend) для обоих проектов ключей на GCE. После этого:  
   `python scripts/run_mo_shadow_dx_plan.py --date 2026-08-06 --resume` на GCE (resume переигрывает строки с `error`).

3. **Опционально / не стартовать без запроса владельца:**  
   полный deep-rescore из UI (`rescore_mo_deep_days` + recompute), чтобы risk-caps/status thresholds переписали уже лежащий `overall` в `cases.jsonl`. Сейчас UI честно считает только **warehouse/zones**.

4. **Не трогать параллельно:**  
   official SSOT overall / primary queue reason; `methodist_labels.jsonl` pilot pack; грязный основной checkout `~/Cursor_Folders/Protocol`.

---

## Безопасные команды

```bash
# версия прода
curl -sS https://protocol.kravira.by/api/version

# профиль на GCE
gcloud compute ssh protocol-app --zone=europe-central2-a --command \
  'sudo cat /var/data/medical_exams/config/mo_scoring_profile.json; sudo cat /var/data/medical_exams/config/mo_recompute_job.json | head -c 800'

# новый task worktree
cd ~/Cursor_Folders/Protocol
scripts/ops/git_task_start.sh <slug> --pc=pc1 --branch=cursor/<slug>-pc1
```

Deploy после merge: `bash deploy/gcp-app/deploy_to_gce.sh` (prod = GCE, не Render Action).  
Gemini/LLM для МО - только GCE (`deploy/gcp-llm/run_on_gce.sh`), не с Mac.

---

## Состояние машины

Mac переводится в **сон** после записи этого handoff. Следующий агент начинает с `git fetch origin` и чтения этого файла + `AGENTS.md`.
