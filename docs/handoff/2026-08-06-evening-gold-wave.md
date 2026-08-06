# Handoff (вечер 06.08): gold error sweep + rescore recovery

**Актуальный handoff** (заменяет afternoon
`docs/handoff/2026-08-06-afternoon-next-agent.md`).  
**Когда:** 2026-08-06 ~20:20 UTC+3.  
**Репо:** `akuazuk/protocol`. **Прод:** `https://protocol-bimy.onrender.com`.  
**SSH Render:** `srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com`.

---

## 0. Старт

```bash
git fetch --prune origin && git checkout main && git pull --ff-only origin main
curl -sS https://protocol-bimy.onrender.com/api/version
gh pr list --repo akuazuk/protocol --state open
```

**Ожидай в `origin/main` минимум:**

| SHA / PR | Суть |
|--|--|
| #34 `3bfd835` | Gold sweep: NSAID brackets/topical, suggest path-block, `mis_diagnos`, rescore script, UX confirm |
| **Открыт #35** | Hotfix: rescore **joins CSV**, NSAID свечи, suggest читает `doctor_specialization` |

Прод на момент handoff: `2026-08-06-155618Z-gold-error-sweep` (#34).  
Код #35 уже **hotpatch** на Render disk (`medication_safety`, `case_protocol_suggest`, `rescore_mo_deep_days`), но в git main его ещё нет, пока CI #35 не зелёный.

---

## 1. Что сделано в волне

План: `docs/plans/2026-08-06-mo-gold-pack-error-sweep-v1.md`.

| Тема | Статус |
|--|--|
| A1-A3 NSAID скобки / gel / Diclovit | done (#34 + свечи в #35) |
| B1 suggest не ЧЛХ для уролога | done после #35 (нужен `doctor_specialization`) |
| C1 `mis_diagnos` | done; smoke 3643304 → `M54.8` |
| E1 LLM Aug 01-05 | grades **80/80**, `llm_queue_pending=0` после recompute |
| Gold export | `/var/data/medical_exams/gold_review/2026-08-06/` (3 packs) |
| Re-score deep Aug 01-05 | done **с CSV join**; avg ~79-81 (было ~85 до первого empty rescore) |

### Инцидент (обязательно знать)

Первый `rescore_mo_deep_days.py` (#34) считал deep **без клинического текста**
(`cases.jsonl` хранит scores/meta; клиника в `mo_YYYY-MM-DD.csv`).  
Avg упал ~85→61. Восстановлено hotpatch-скриптом с join CSV + recompute.
**Никогда** не запускать rescore без CSV join.

Smoke после recovery (Render, hotpatched code):

- **3650612:** нет `C_nsaid_dup`, нет `D_reg55_p0`; suggest без ЧЛХ (урология).
- **3643304:** ICD `M54.8` из `mis_diagnos` / full-doc.

---

## 2. Сделать первым

1. Дождаться CI green на **#35** → `gh pr merge 35 --squash --delete-branch`.
2. `scripts/ops/render_release_main.sh --commit="$(git rev-parse origin/main)" --prod-url=https://protocol-bimy.onrender.com`
3. Проверить `/api/version` = stamp `#35` (`…Z-rescore-csv-join` или новее).
4. Обновить чекбоксы S5 в плане gold-sweep; residuals C3/C4, D2/D3, july repair.

GitHub Actions сегодня часто **cancel** jobs на queue 15m без steps - rerun /
close-reopen PR.

---

## 3. Residuals (не блокер волны)

- ICD P3 warehouse soft-fill / P4 LLM prompts.
- Suggest ranking всё ещё может давать слабые уро-КП (не ЧЛХ) - улучшить match.
- `B_icd_invalid` на 3650612 без кода в тексте - вероятно TP (подтверждено экспертом).
- July `doctor_key` repair.
- Pack UX D2/D3 notes.

---

## 4. Команды Render

```bash
# После деплоя #35 (если снова нужен deep):
.venv/bin/python scripts/rescore_mo_deep_days.py \
  --data-root /var/data/medical_exams \
  --first-date 2026-08-01 --last-date 2026-08-05
.venv/bin/python scripts/recompute_mo_days.py \
  --data-root /var/data/medical_exams \
  --first-date 2026-08-01 --last-date 2026-08-05 \
  --warehouse /var/data/medical_exams/warehouse/mo_analytics.sqlite
```

LLM только на Render: `bash scripts/run_mo_render_llm_backfill.sh …`
