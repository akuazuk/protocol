# Handoff следующему агенту (ночь 07.08 после gold wave)

**Актуальный handoff** (заменяет
`docs/handoff/2026-08-06-evening-gold-wave.md`).  
**Когда:** 2026-08-07 ~03:25 UTC+3.  
**Репо:** `akuazuk/protocol`. **Прод:** `https://protocol-bimy.onrender.com`.  
**SSH Render:** `srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com`  
(ключ: `~/.ssh/id_ed25519`).

Mac после пуша этого handoff уходит в sleep - работай со своего clone.

---

## 0. Старт (сделай первым)

```bash
cd /path/to/protocol
git fetch --prune origin
git checkout main
git pull --ff-only origin main
git status --short --branch
git log -5 --oneline
curl -sS https://protocol-bimy.onrender.com/api/version
gh pr list --repo akuazuk/protocol --state open
```

**Ожидай в `origin/main` и в проде:**

| SHA (short) | PR | Суть |
|--|--|--|
| `cf2ea739` | **#35** | rescore joins CSV; NSAID свечи; suggest `doctor_specialization`; evening handoff |
| `3bfd8356` | #34 | gold sweep: NSAID скобки/гель, path-block ЧЛХ, `mis_diagnos`, UX confirm |

**Прод `/api/version`:** `2026-08-06-172103Z-evening-handoff`  
(или новее после твоего merge). Если старше - подожди webhook /
`scripts/ops/render_release_main.sh`.

Дальше: `docs/plans/README.md` (только `active`). Не коммить в чужую task-ветку.

---

## 1. Что сделано предыдущим агентом (волна «делай всю волну»)

План: `docs/plans/2026-08-06-mo-gold-pack-error-sweep-v1.md` (волна S0-S5 **закрыта**).

### Код в main

1. **#34** - NSAID: альтернативы в скобках / oral+гель не dup; Diclovit/Найз aliases;
   suggest path-block стоматология/ЧЛХ для уролога; `mis_diagnos` в ICD resolve;
   UX confirm при `training_use` + все findings unreviewed;
   скрипт `scripts/rescore_mo_deep_days.py` (первая версия - баг, см. ниже).
2. **#35** - **обязательный** CSV-join в rescore; не затирать `overall_pct`;
   NSAID «свечи» = topical для dup; suggest читает `doctor_specialization` из clinical/record.

### Ops на Render disk (уже сделано, не повторять без нужды)

| Шаг | Результат |
|--|--|
| LLM grades Aug 01-05 | **80/80** ok каждый день (`secure_cases/2026/08/kz_l1_*_llm_grades.jsonl`) |
| `recompute_mo_days` 01-05 | `llm_queue_pending=0` (баннер «День принят с замечанием» ушёл) |
| Deep rescore с CSV join | Aug 01-05 пересчитаны; avg deep ~79-81 |
| Gold export | `/var/data/medical_exams/gold_review/2026-08-06/` (3 packs) |
| Smoke 3650612 | нет `C_nsaid_dup` / `D_reg55_p0`; suggest без ЧЛХ |
| Smoke 3643304 | ICD `M54.8` из `mis_diagnos` / full-doc |

### Инцидент (не повторять)

Первый rescore (#34) гонял `evaluate_kz_deep` **без клиники**
(`kz_l1_*_cases.jsonl` = scores/meta; текст в `mo_YYYY-MM-DD.csv`).  
Avg упал ~85→61. Чинится только скриптом из #35 (CSV join).  
**Никогда** не запускать rescore без join CSV.

LLM grades - **только Render**, не Mac (`gemini-via-render.mdc`).  
Deploy убивает nohup grade - после деплоя перезапуск:
`bash scripts/run_mo_render_llm_backfill.sh YYYY-MM-DD YYYY-MM-DD`.

### Workflow

Task-ветка → bump `BUILD_VERSION` (`scripts/ops/bump_build_version.sh`, UTC stamp) →
PR → CI green → `gh pr merge --squash --delete-branch` →
`scripts/ops/render_release_main.sh --commit="$(git rev-parse origin/main)"`.  
Коммиты **без** AI Co-authored-by / Cursor trailer.

---

## 2. Что делать дальше (приоритет)

### P0 / P1 residuals из gold-sweep плана

1. **ICD P3:** soft-fill `mkb_code_main` в warehouse/export (план
   `2026-08-06-mo-icd-full-document-search-v1.md`).
2. **ICD P4:** строка в LLM judge / methodist prompts - «МКБ может быть в любом разделе».
3. **Suggest ranking:** после path-block уролог всё ещё может получать слабые КП
   (не ЧЛХ) - улучшить match по диагнозу/specialty.
4. **July warehouse:** `repair_mo_warehouse_from_secure` - bad `doctor_key` (~11k).
5. **Pack UX D2/D3:** summary vs finding_decisions; optional `note_ru` на FP.

### P1 ops / качество

6. Continuous LLM на новый день августа + action-judge post-step (частично есть).
7. Gold export после роста `training_use` packs (≥50 цель раньше).
8. Проверить UI баннеры дней 01-05 без `llm_queue_pending` после свежего UI-кэша.

### Не трогать без задачи

- Не пушить в `main` мимо PR.
- Не коммитить `minzdrav_protocols/*.pdf`, `.env`, dumps `data/ml/reports/`.
- Не запускать empty-clinical rescore.

---

## 3. Быстрые команды

```bash
# Версия
grep BUILD_VERSION rag_server.py
curl -sS https://protocol-bimy.onrender.com/api/version

# LLM только Render
bash scripts/run_mo_render_llm_backfill.sh 2026-08-06 2026-08-06

# Deep rescore (только с CSV join из #35)
ssh srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com \
  'cd /opt/render/project/src && .venv/bin/python scripts/rescore_mo_deep_days.py \
   --data-root /var/data/medical_exams --first-date 2026-08-01 --last-date 2026-08-05'

# Recompute отчётов
.venv/bin/python scripts/recompute_mo_days.py \
  --data-root /var/data/medical_exams \
  --first-date 2026-08-01 --last-date 2026-08-05 \
  --warehouse /var/data/medical_exams/warehouse/mo_analytics.sqlite

# Тесты волны
PYTHONPATH=. pytest -q \
  tests/test_nsaid_alternatives_topical.py \
  tests/test_case_protocol_suggest.py \
  tests/test_mo_icd_full_document.py
```

VPN: перед SQL МИС `~/CURSOR/bin/vanya_vpn.sh ensure-off`; перед сильными моделями `ensure-on`.

---

## 4. Definition of Done для следующей задачи

1. Task-ветка от свежего `origin/main` + bump `BUILD_VERSION` + squash-merge PR.
2. `render_release_main.sh` → `/api/version` совпал.
3. План в `docs/plans/` обновлён; при новой теме - новый `…-vN.md`.
4. Этот handoff при существенном сдвиге - новый файл и ссылка сверху у предыдущего.
