# Инструкция агенту: продолжение калибровки МО (3 дня работ)

Дата handoff: 2026-08-09  
Аудитория: другой агент / другой компьютер  
Язык ответов владельцу: русский, коротко.

Читай целиком до любых правок Git, GCE или production.

---

## 0. Preflight (обязательно)

```bash
cd <clean-worktree-or-new>
git fetch --prune origin
git status --short --branch
git rev-list --left-right --count origin/main...HEAD
gh pr list --repo akuazuk/protocol --state open
```

Обязательные файлы:

1. `AGENTS.md`
2. `docs/plans/README.md`
3. этот handoff
4. `docs/plans/2026-08-09-mo-score-ssot-llm-recompute-v3.md`
5. `docs/plans/2026-08-09-mo-calibration-confirmatory-proxy-v1.md`

Правила, которые нельзя нарушать:

- Не работать в грязном/отстающем checkout; новый clean worktree от `origin/main`.
- Не push в `main`; только task-ветка → PR → merge.
- Gemini / live LLM для МО только через GCE (`deploy/gcp-llm/run_on_gce.sh`), не с Mac.
- PHI (клинические тексты, visit/patient IDs, secret JSONL) не писать в git, PR, handoff, ответы.
- Production scoring / thresholds / action queue / SSOT **не менять**, пока formal C6-C9 с human gold не закрыты.
- AI-proxy **не является** human gold и не пишется в `methodist_labels.jsonl`.

---

## 1. Зачем всё это (цель владельца)

Владелец хочет понять, какая оценка лучше ловит плохое МО:

- текущий deterministic engine (зоны, №55, axes, overall);
- новые Endpoint C/D (Dx evidence, plan/protocol concordance через blind LLM);
- ensembles.

Кардинальный подход: **сначала калибровка и сравнение**, потом SSOT №55 и recompute. Полный production rollout запрещён до confirmatory gate.

---

## 2. Что уже сделано за ~3 дня (хронология)

### A. UI / кабинет (уже в main, не трогать без отдельной задачи)

- Full-width protocol reader (Study mode, абзацы).
- Кабинет методиста в стиле MO Analytics.
- Secure C6 labeling UI: `/methodist/calibration` на GCE primary `protocol.kravira.by`.

### B. Formal calibration plan v3 (active)

План: `docs/plans/2026-08-09-mo-score-ssot-llm-recompute-v3.md`

| Phase | Status | Суть |
|--|--|--|
| C0 | done | stratified sample 30, seed 42, secret/public split |
| C1 | done | engine snapshot + replay; drift finding: exact match 0/30 |
| C2 | done | Endpoint C/D contracts |
| C3 | done | blind prompts + leakage tests |
| C4 | done | GCE smoke 5×2 |
| C5 | done | GCE pilot 30×2 + LLM adjudication; 18 disagreement cases |
| C6 pack+UI | done | blinded pack + secure form; **human labels 0/22** |
| C7 formal | blocked | нужен human gold |
| C8 formal | blocked | нужен human gold |
| C9 formal | blocked | нужен human gold |

### C. Exploratory без методиста (сделано / в работе)

Владелец явно разрешил продолжать без human labels, с оговоркой: proxy ≠ gold.

| Phase | Status | Суть |
|--|--|--|
| C6A | done | independent proxy `gemini-3.1-pro-preview` на frozen 30 |
| C7A | done | PHI-safe compare score families vs proxy + bootstrap CI |
| C8A | done | provisional shadow: plan=`blind.pass_1`; Dx=нестабильно |
| C9A | **in progress** | July confirmatory ≥100 on GCE |

Merged:

- PR #112 `feat(mo): add exploratory calibration proxy` → `origin/main` SHA `a76f93f3`

Open / green CI:

- PR #113 `feat(mo): add confirmatory proxy calibration path`  
  https://github.com/akuazuk/protocol/pull/113  
  branch `cursor/mo-calibration-confirmatory-proxy-c9a-pc1`  
  HEAD at handoff write-time: `42c0f740`  
  CI: green / mergeState CLEAN

---

## 3. Текущее состояние прямо сейчас

### Git

- `origin/main`: содержит C6A/C7A (после #112), **не** содержит C9A sampler/GCE mode пока #113 не merged.
- Активная задача: PR #113.
- Не открывать параллельные PR на те же calibration scripts.

### GCE live job (C9A)

Directory:

```text
/var/data/medical_exams/calibration/mo-score-v3-confirmatory-2026-07-26-2026-07-31/
```

Уже готово:

- sample `100/100`, seed 43, `--no-sentinel`, exclude pilot 30 keys, overlap 0
- replay audit_complete=true (exact match по-прежнему плохой; это ожидаемый drift)
- public_manifest + secret artifacts

В процессе на момент handoff:

- `secret/blind_confirmatory.jsonl` — flash model, ~30+/100, errors 0
- `secret/agent_proxy_confirmatory.jsonl` — ещё не начат
- затем `agent_proxy_eval_summary.json` + `provisional_methodology.json`

Команда, которая это запускает:

```bash
bash deploy/gcp-llm/run_on_gce.sh 2026-07-26 2026-07-31 --calibration-confirmatory-proxy
```

Мониторинг без PHI:

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a --quiet --command="
sudo docker exec protocol-web python -c \"
import json
from pathlib import Path
root=Path('/var/data/medical_exams/calibration/mo-score-v3-confirmatory-2026-07-26-2026-07-31/secret')
for name in ('blind_confirmatory.jsonl','agent_proxy_confirmatory.jsonl'):
 p=root/name
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
 print(name, {'n':len(rows),'ok':sum(not r.get('error') for r in rows),'err':sum(bool(r.get('error')) for r in rows),'last':rows[-1].get('sample_id') if rows else None})
\""
```

ETA ориентир: blind остаток + proxy ≈ **1.5-2.5 часа** от момента ~34/100 blind.

Если SSH-обёртка локально умерла, а JSONL растёт - job на VM ещё жив. Resume уже в командах (`--resume`).

### Formal C6 human gate

- UI: `https://protocol.kravira.by/methodist/calibration`
- Roles: methodist/lead/admin
- Labels: **0/22**
- Comparison unseal только после 22/22 valid labels
- Secret pack только на GCE; не копировать на Render backup

### Production

- Scoring/thresholds **не менялись** этой калибровкой.
- Render backup deploy идёт только после merge в main через Action; GCE primary для calibration UI деплоится отдельно release-координатором.
- Не запускай production deploy, если ты не назначенный release-координатор.

---

## 4. Ключевые артефакты (куда смотреть)

### В git (PHI-safe)

| Path | Что |
|--|--|
| `docs/plans/2026-08-09-mo-score-ssot-llm-recompute-v3.md` | formal plan C0-C9 |
| `docs/plans/2026-08-09-mo-calibration-agent-proxy-v1.md` | C6A/C7A completed |
| `docs/plans/2026-08-09-mo-calibration-confirmatory-proxy-v1.md` | C8A/C9A active |
| `eval/mo_score_calibration/agent-proxy-summary.json` | C7A aggregate |
| `eval/mo_score_calibration/provisional-methodology-c8a.json` | C8A decision |
| `scripts/build_mo_score_calibration_sample.py` | sampler + exclude/clamp |
| `scripts/run_mo_calibration_blind_judge.py` | blind/proxy judge |
| `scripts/eval_mo_score_agent_proxy.py` | C7A metrics |
| `scripts/select_mo_calibration_provisional.py` | C8A selector |
| `deploy/gcp-llm/run_on_gce.sh` | GCE modes |
| `clinical_knowledge/mo_calibration_methodist_ui.py` | C6 UI backend |
| `frontend/web/methodist/mo-calibration.html` (+ css/js) | C6 form |

### Только на GCE (secret)

| Path | Что |
|--|--|
| `.../mo-score-v3-2026-08-01-2026-08-08/` | frozen pilot 30 (не перезаписывать) |
| `.../mo-score-v3-2026-08-01-2026-08-08/secret/methodist/` | C6 pack + labels |
| `.../mo-score-v3-confirmatory-2026-07-26-2026-07-31/` | C9A cohort |

---

## 5. Главные находки (уже известны)

1. **Replay drift**: stored warehouse/snapshot ≠ current engine replay (pilot 0/30 exact; confirmatory similarly poor exact match). Arm D fingerprint нужен как baseline, warehouse не переписывать «чтобы совпало».
2. **№55 / zones / case review diverge** из-за разных путей расчёта; SSOT отложен до calibration gate.
3. **C7A (n=30 proxy)**:
   - Dx: только 2 proxy-bad → ranking нестабилен.
   - Plan: blind pass 1 лучший provisional (PR-AUC ~0.59), CI перекрываются; ensembles не выиграли.
   - `zone2b` часто без покрытия.
4. **C8A**:
   - Dx → `no_stable_provisional`
   - Plan → `provisional_shadow:blind.pass_1`
   - `production_rollout.allowed=false`
5. **July warehouse**: `reg55_*` отсутствуют → sampler clamp'ит эти floors в 0 для confirmatory.

---

## 6. Что делать дальше (порядок)

### Шаг 1 - дождаться / добить C9A на GCE

1. Проверить рост `blind_confirmatory.jsonl` до ~100.
2. Дождаться `agent_proxy_confirmatory.jsonl` ~100.
3. Убедиться, что появились:
   - `.../agent_proxy_eval_summary.json`
   - `.../provisional_methodology.json`
4. Если оборвалось:

```bash
# из worktree с кодом #113 / после merge
bash deploy/gcp-llm/run_on_gce.sh 2026-07-26 2026-07-31 --calibration-confirmatory-proxy
```

Resume внутри уже есть. Не пересобирай sample без нужды: directory уже содержит 100 cases.

### Шаг 2 - забрать только PHI-safe aggregates в git

```bash
gcloud compute scp \
  protocol-app:/var/data/medical_exams/calibration/mo-score-v3-confirmatory-2026-07-26-2026-07-31/agent_proxy_eval_summary.json \
  eval/mo_score_calibration/confirmatory-agent-proxy-summary.json \
  --zone=europe-central2-a

gcloud compute scp \
  protocol-app:/var/data/medical_exams/calibration/mo-score-v3-confirmatory-2026-07-26-2026-07-31/provisional_methodology.json \
  eval/mo_score_calibration/confirmatory-provisional-methodology.json \
  --zone=europe-central2-a
```

Проверить, что в JSON нет sample IDs / clinical text / mis_id.

Обновить:

- `docs/plans/2026-08-09-mo-calibration-confirmatory-proxy-v1.md`
- handoff
- при необходимости комментарий в PR #113

### Шаг 3 - merge #113 после зелёного CI и актуальных aggregates

Обычный путь: squash merge #113 → не деплоить scoring-изменения (их нет) → GCE primary для UI не обязателен для C9A artifacts.

### Шаг 4 - formal C6 (настоящий блокер продукта)

Методист заполняет 22 labels в `/methodist/calibration`.

После `passed=true`:

1. Formal C7 против human gold (не proxy).
2. Formal C8 выбрать methodology.
3. Formal C9 confirm (≥100 или ≥30 bad) уже можно сопоставить с C9A.
4. Только потом P0/P1 из plan v3: SSOT №55, queue signal, recompute.

### Чего не делать

- Не записывать AI-proxy в methodist labels.
- Не менять production thresholds / queue / SSOT по C7A/C8A.
- Не перезаписывать pilot dir `mo-score-v3-2026-08-01-2026-08-08`.
- Не гонять Gemini с Mac.
- Не force-push / reset чужой worktree.

---

## 7. Как подключиться с другого компьютера

```bash
git fetch --prune origin
scripts/ops/git_task_start.sh mo-calibration-c9a-followup --pc=<pc-id> \
  --branch=cursor/mo-calibration-c9a-followup-<agent>-<pc-id>
# если продолжаешь именно PR #113 и он ещё open:
# создай worktree от origin/cursor/mo-calibration-confirmatory-proxy-c9a-pc1
# и не плоди второй PR на те же файлы
```

Если #113 ещё open и ты продолжаешь ту же задачу - работай в этой ветке/PR, не создавай дубль.

Проверка GCE доступа:

```bash
gcloud config set project protocol-home-e1 --quiet
gcloud compute ssh protocol-app --zone=europe-central2-a --quiet --command='hostname'
```

VPN: для Gemini/GCE LLM обычно нужен рабочий контур без белорусского geo на клиенте; live calls всё равно только с GCE. Для MIS SQL - VanyaVPN `ensure-off` (см. `.cursor/rules/mis-mariadb.mdc`); для этой калибровки MIS SQL не нужен.

---

## 8. Self-check перед ответом «готово»

- [ ] C9A blind≈100 и proxy≈100 или объяснена ошибка
- [ ] PHI-safe aggregates в git, secret остаётся на GCE
- [ ] plan/handoff обновлены
- [ ] PR актуален / merged
- [ ] Explicitly сказано: formal C6 still 0/22; production scoring unchanged
- [ ] Нет AI co-authored trailer в коммитах

---

## 9. Одна безопасная следующая команда

Сначала статус C9A:

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a --quiet --command="
sudo docker exec protocol-web python -c \"
import json
from pathlib import Path
root=Path('/var/data/medical_exams/calibration/mo-score-v3-confirmatory-2026-07-26-2026-07-31/secret')
for name in ('blind_confirmatory.jsonl','agent_proxy_confirmatory.jsonl'):
 p=root/name
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
 print(name, len(rows), sum(not r.get('error') for r in rows))
\""
```

Если blind/proxy не закончены и процесс мёртв - resume той же GCE командой из §6 шаг 1.

Если закончены - забрать aggregates (§6 шаг 2) и обновить PR #113.

---

## 10. Файлы, которые нельзя трогать параллельно

- `scripts/build_mo_score_calibration_sample.py`
- `scripts/run_mo_calibration_blind_judge.py`
- `scripts/eval_mo_score_agent_proxy.py`
- `scripts/select_mo_calibration_provisional.py`
- `deploy/gcp-llm/run_on_gce.sh`
- `clinical_knowledge/mo_calibration_methodist_ui.py`
- `frontend/web/methodist/mo-calibration.*`
- `eval/mo_score_calibration/*`
- GCE dirs `mo-score-v3-2026-08-01-2026-08-08` и `mo-score-v3-confirmatory-2026-07-26-2026-07-31`
