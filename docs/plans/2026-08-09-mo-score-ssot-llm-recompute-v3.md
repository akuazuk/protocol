# МО: клиническая цепочка + план по КП + калибровка всех оценок (v3)

Дата: 2026-08-09
Статус: **active** (C0-C5 выполнены; C6 secure UI готов, human labels 0/22)
Преемник: `2026-08-09-mo-score-ssot-llm-recompute-v2.md`

Связанные планы:
- `2026-08-09-mo-reg55-section-pack-v1.md` - один section-pack №55;
- `2026-08-08-mo-icd-absent-ok-with-dx-v1.md` - отсутствие МКБ при Dx не дефект;
- `2026-08-09-mo-icd-name-multidx-brief-v1.md` - осторожная проверка МКБ;
- `2026-08-09-mo-case-review-quality-parity-v1.md` - разбор случая;
- `2026-08-07-by-home-gcp-llm-split-v1.md` - живой Gemini только на GCE.

---

## 1. Что добавлено относительно v2

Весь дизайн v2 сохраняется: 30 стратифицированных МО, current-engine snapshot,
blind LLM ×2, methodist adjudication, bootstrap, confirmatory cohort ≥100,
PHI только на GCE и запрет полного пересчёта до calibration gate.

Добавлены две самостоятельные клинические оценки:

1. **Dx evidence concordance** - подтверждается ли поставленный диагноз (текстом или
   МКБ) жалобами, анамнезом, объективным статусом и результатами обследований.
2. **Dx-conditioned plan concordance** - если принять диагноз правильным, соответствуют
   ли назначенные обследования, лечение и наблюдение найденному КП; если надёжного КП
   нет - отдельная LLM-оценка по общим клиническим знаниям с пониженным trust.

Они не заменяют существующие баллы, а добавляются в пилот как отдельные кандидаты.

---

## 2. Диагноз текущей системы

### 2.1 Живой пример `3643940`

| Метрика | Значение |
|--|--|
| `reg55_section_pct` | 83.3 |
| axis regulatory | 77.8 |
| `overall_pct` | 60.0 |
| zones | 92.9 / 100 / n/a |
| МКБ | Z00.8 |
| Dx | артериальная гипертензия / гипотиреоз |

Несколько чисел описывают разные задачи и частично повторяют одни признаки.

### 2.2 Что уже умеет движок

| Контур | Что реально проверяет | Ограничение |
|--|--|--|
| `overall_pct` v4 | weighted axes + P0/P1 caps | среднее скрывает провал одной зоны |
| axis `clinical_concordance` | наличие supporting slots, МКБ, часть plan-vs-protocol | смешивает диагноз, кодирование и план |
| `B_dx_no_support` | есть ли хоть жалобы / анамнез / статус | не доказывает клиническую правдоподобность |
| zone2a | заполненность Dx/МКБ/клинических слотов | completeness, не semantic fit |
| `mo_clinical_gaps` / concordance | отдельные cross-field heuristics | shadow и ограниченные домены |
| ICD directory / name | код и название по справочнику | не сравнивает диагноз со всей клиникой |
| LLM Stage A | complaints/history/status/tests → Dx + ICD | нужная семантика, но shadow |
| B4-B6 deep/v4 | required exams / criteria / treatment по protocol ctx | substring и другой resolver, чем UI |
| zone2b | наличие плана + alignment proxy при найденном КП | без КП становится n/a / максимум 0.5 |
| `rule_checker` | богатые protocol rules | работает в consult-review, не primary MO |
| LLM Stage B | обследование / лечение / follow-up | КП в prompt не передаётся |

**Вывод:** нужные компоненты существуют, но двух явных и надёжных endpoints пока нет.

---

## 3. Целевые задачи (четыре endpoints)

### Endpoint A - плохое МО в целом

Подтверждён хотя бы один существенный дефект:

- опасное упущение / потенциальный вред;
- диагноз не подтверждается клиникой;
- код МКБ явно относится к другой сущности;
- необходимый план обследования / лечения / наблюдения отсутствует или неадекватен;
- запись настолько неполна, что решение нельзя проверить.

### Endpoint B - соответствие №55

Только `evaluate_reg55_section`: `0 / 0.5 / 1`, `n/a` вне знаменателя.
Это нормативный, а не общий клинический score.

### Endpoint C - Dx evidence concordance

Вопрос: **следует ли поставленный диагноз из имеющихся данных?**

Вход:

- диагноз свободным текстом;
- один или несколько МКБ;
- жалобы;
- анамнез;
- объективный статус;
- результаты лабораторных / инструментальных исследований;
- возрастная группа, пол, специальность.

Если есть только МКБ, его справочное название используется как формулировка диагноза.
Если нет ни Dx, ни МКБ - отдельный completeness defect, а не semantic mismatch.

Выход:

```json
{
  "dx_evidence_pct": 0,
  "verdict": "good|partial|poor|critical|blocked|na",
  "supported_by": [],
  "not_supported_by": [],
  "contradictions": [],
  "icd_fit": "fit|partial|mismatch|unknown|na",
  "potential_harm": false,
  "provenance": "deterministic|llm_blind|methodist"
}
```

Правила:

- `blocked`, если клинических данных недостаточно для semantic verdict;
- валидный МКБ не означает автоматически правильный диагноз;
- отсутствие МКБ при содержательном Dx не штрафуется;
- неизвестный код - отдельный дефект справочника;
- несоответствие кода клинике - отдельный осторожный finding;
- findings имеют evidence из входных слотов.

### Endpoint D - Dx-conditioned plan / protocol concordance

Вопрос: **если принять диагноз правильным, достаточны ли обследование, лечение и
наблюдение?**

Диагноз здесь является premise. Endpoint D не пересчитывает Endpoint C.

#### D1. КП найден с достаточным trust

Единый matcher:

1. ICD-first / text fallback;
2. тот же `case_protocol_suggest` / `clinical_kp_hit`, что показывается в UI;
3. threshold и выбранный КП сохраняются в snapshot;
4. structured summary / rules / source refs - источник требований.

Отдельные баллы:

```json
{
  "exam_protocol_pct": 0,
  "treatment_protocol_pct": 0,
  "followup_protocol_pct": 0,
  "plan_protocol_pct": 0,
  "verdict": "good|partial|poor|critical|na",
  "kp_path": "...",
  "kp_trust": "A|B|C|D",
  "missing_required": [],
  "off_protocol": [],
  "source_refs": [],
  "provenance": "kp_grounded"
}
```

Проверка:

- required exams;
- диагностические критерии / контроль;
- treatment groups / дозы, если доступны;
- противопоказания / safety;
- follow-up / повторный приём / маршрутизация;
- applicability по возрасту, полу, роли и типу визита.

#### D2. КП не найден или trust ниже порога

Нельзя писать «не соответствует КП».

Выполняется blind LLM fallback:

- prompt содержит принятый диагноз и клинические/плановые слоты;
- prompt не содержит scores, findings, queue reason и результаты matcher;
- модель оценивает общую клиническую достаточность;
- результат хранится отдельно:
  `plan_general_llm_pct`, `provenance=llm_no_kp`, `kp_status=unmatched`;
- trust ниже, чем у КП-grounded оценки;
- результат не входит в №55 и не притворяется protocol compliance.

Если данных мало - `blocked`, а не `poor`.

---

## 4. Все оценки, которые сохраняем в эксперименте

Ничего из текущих оценок не удаляется до анализа:

1. `overall_pct` v4.
2. `overall_pct_v3`.
3. axes:
   - documentation;
   - clinical_concordance;
   - safety;
   - regulatory.
4. zones:
   - zone1 documentation;
   - zone2a diagnosis;
   - zone2b plan.
5. `rubric_pct`.
6. `reg55_section_pct`, band, applicable_n, weak points.
7. P0 / P1 / P2 findings по семействам.
8. Action-queue membership / attention reason.
9. ICD directory / name / full-document pipeline.
10. Existing LLM grade / action judge / ICD review, если есть.
11. Новый `dx_evidence_pct`.
12. Новый `plan_protocol_pct` для KP-matched.
13. Новый `plan_general_llm_pct` для KP-unmatched.
14. Решения методиста по Dx / ICD / plan / safety / №55.

Коррелирующие оценки не считаются независимыми голосами.

---

## 5. Калибровочная выборка

### 5.1 Pilot 30

- период `2026-08-01..2026-08-08`;
- seed `42`;
- sentinel `3643940`;
- пять bands `overall_pct`: по 4 случая;
- 3 high-score + action/P0-P1;
- 3 расхождения reg55 vs regulatory ≥5 п.п.;
- 2 reg55 ≥80 + weak points;
- все доступные `training_use=1`.

Обязательное покрытие:

- минимум 4 специальности;
- минимум 4 ICD ↔ Dx disputes;
- минимум 8 KP-matched;
- минимум 6 KP-unmatched;
- минимум 4 случая с результатами обследований;
- минимум 4 случая с непустым лечением;
- не более 3 случаев одного врача.

### 5.2 Confirmatory cohort

До изменения primary composite и полного recompute:

- ≥100 уникальных МО **или** ≥30 adjudicated bad cases;
- другой период;
- тот же frozen prompt / model / config;
- specialty-specific breakdown при `n≥10`.

---

## 6. Три независимых слоя

### Arm D - engine snapshot

Все оценки §4 + version/config hashes. Replay выполняется на полном clinical payload.

### Arm L - blind LLM на GCE

Для каждого случая:

1. Stage A: Dx evidence + ICD.
2. Stage B-KP: план по evidence найденного КП, если trust A/B.
3. Stage B-general: общая adequacy, если КП не найден.
4. Второй независимый pass.
5. Adjudication LLM при расхождении повторов.

Для Stage B-KP LLM получает только:

- принятый диагноз;
- плановые слоты;
- минимальный набор проверяемых требований / цитат КП;
- provenance и applicability.

Он не получает engine verdict или процент совпадения.

### Arm H - методист

Обязательно проверяются:

- все D ↔ L disagreement по Endpoint A/C/D;
- все critical / harm;
- все ICD ↔ Dx mismatch;
- все reg55 ≥80 с «невыполненными»;
- все KP-unmatched, где LLM ставит poor/critical;
- минимум 5 случайных согласованных случаев.

Primary gold - решение методиста. LLM без H остаётся provisional.

---

## 7. Что сравниваем

### Одиночные candidates

- overall v4 / v3;
- каждая axis;
- minimum clinical/safety axes;
- каждая zone и minimum applicable zone;
- reg55 только для Endpoint B;
- action queue;
- confirmed P0/P1;
- `dx_evidence_pct`;
- `plan_protocol_pct`;
- `plan_general_llm_pct` для отдельной KP-unmatched страты.

### Заранее заданные ensembles

| Candidate | Правило |
|--|--|
| `clinical_gate` | safety <85 или dx_evidence <70 или confirmed P0/P1 |
| `dx_plan_gate` | dx_evidence <70 или plan_protocol <70 |
| `dx_plan_general_gate` | dx_evidence <70 или соответствующий plan score <70 |
| `ensemble_union_70` | overall <70 или applicable zone <70 или clinical_gate |
| `ensemble_with_reg55` | ensemble_union_70 или reg55 <55 |
| `review_priority` | minimum из clinical scores; reg55 показывается отдельно |
| `queue_current` | текущий action-queue selector |

Endpoint D использует **ровно один** plan score:

- KP matched → `plan_protocol_pct`;
- KP unmatched → `plan_general_llm_pct`;
- никогда оба одновременно.

№55 сравнивается отдельно и не должен ухудшать Endpoint A из-за чисто формального пункта.

---

## 8. Как выбираем методику

Primary:

1. 100% recall adjudicated critical / potential harm на pilot.
2. Максимальный recall bad МО.
3. Recall отдельно для bad Dx и bad plan.
4. При близком recall - выше precision и ниже review load.
5. Нет провала на high-score cases и одной специальности.

Метрики:

- sensitivity, specificity, precision, NPV;
- false-negative rate при исходном score ≥80;
- diagnosis / ICD / plan / safety recall;
- PR-AUC / ROC-AUC как pilot;
- Spearman, MAE, QWK;
- review load;
- KP match coverage и fallback rate.

Uncertainty:

- paired stratified bootstrap 1000×;
- 95% CI;
- числитель / знаменатель;
- n=30 - directional pilot, не окончательная победа.

---

## 9. Leakage и double-counting guards

- Blind LLM не видит scores/findings/queue reason.
- Stage A не видит plan verdict.
- Stage D принимает Dx как premise и не переоценивает его.
- KP-grounded и no-KP LLM scores не смешиваются в одном случае.
- ICD name match не считается доказательством клинической правоты Dx.
- Existing `clinical_concordance`, zone2a и `dx_evidence` сравниваются, но не
  суммируются как три независимых голоса.
- Existing B4-B6 и новый plan endpoint дедуплицируются по finding family.
- №55 - отдельный Endpoint B.

---

## 10. Реализация calibration harness

Новые файлы:

| Файл | Назначение |
|--|--|
| `scripts/build_mo_score_calibration_sample.py` | PHI-safe stratified sample |
| `clinical_knowledge/mo_dx_evidence_score.py` | explicit Endpoint C contract |
| `clinical_knowledge/mo_plan_protocol_score.py` | KP-grounded Endpoint D contract |
| `scripts/run_mo_calibration_blind_judge.py` | GCE Stage A / B-KP / B-general |
| `scripts/eval_mo_score_calibration.py` | candidates, ensembles, bootstrap |
| `eval/mo_score_calibration/README.md` | pre-registered protocol |
| `tests/test_mo_score_calibration*.py` | sampler, contracts, leakage, metrics |

Переиспользовать:

- `case_protocol_suggest` / `clinical_kp_hit`;
- structured protocol summaries;
- `rule_checker`;
- `mo_llm_action_judge` Stage A/B schemas;
- `build_kz_gold_sample.py`;
- `kz_gold_annotation.py`;
- `calibrate_deep_thresholds.py`;
- review packs;
- `deploy/gcp-llm/run_on_gce.sh`.

---

## 11. Gates и шаги

### C - до изменения production

- [x] **C0** Sampler + secret/public manifests
- [x] **C1** Snapshot всех существующих scores и replay reproducibility
- [x] **C2** Контракты Endpoint C/D + synthetic tests
- [x] **C3** Blind prompts + автоматический leakage test
- [x] **C4** GCE smoke 5 случаев ×2
- [x] **C5** GCE pilot 30 ×2 + LLM adjudication
- [ ] **C6** Methodist labels: ≥15 и все disagreements
- [ ] **C7** Сравнение одиночных scores и ensembles с CI
- [ ] **C8** Pilot report + выбор provisional methodology
- [ ] **C9** Confirmatory cohort ≥100 или ≥30 bad

#### Результат C0-C4 от 2026-08-09

- C0: 30 случаев, seed 42, sentinel включён, все 5 `training_use=1` включены,
  максимум 3 случая одного врача, deficits отсутствуют.
- Покрытие: 13 специальностей, ICD ↔ Dx disputes 4, KP matched 8, KP unmatched
  22, обследования 18, лечение 30; каждый score band содержит минимум 4 случая.
- C1: snapshot всех семейств оценок сохранён; replay выполнен для 30/30 без
  runtime errors, но точное совпадение current engine с сохранённым snapshot -
  0/30. Это отдельный drift blocker до C5, warehouse не перезаписывался.
- C2-C3: 23 synthetic/unit tests; смешение KP/no-KP маршрутов, неподтверждённый
  ICD mismatch и score leakage отклоняются.
- C4 на GCE: 5 случаев × 2, `gemini-3.6-flash`, 10/10 parse, 0 leakage,
  0 geo/API/runtime errors; routes: 4 KP-grounded и 6 no-KP.
- Repeat agreement: Dx verdict 4/5, plan verdict 5/5; средняя абсолютная
  разница score 4.0 п.п. для Dx и 1.2 п.п. для plan.
- Secret artifacts остались только в `/var/data/medical_exams/calibration/`;
  в git и отчёт попали только агрегаты.
- Production scoring, action queue, warehouse и UI не изменялись.

#### Результат C5 от 2026-08-09

- Arm D заморожен fingerprint
  `9ab7bfcb5a84f47354aed8f916c08d5285319bd94d43bc2614a848cb54da2e49`:
  code/config hashes, 1876 protocol summaries и relevant environment flags.
- Blind judge/model/contracts заморожены fingerprint
  `1e106ed8e0980127643aaf247f978a326a4d26f3bc167918274da843cdbefca0`.
- Frozen pilot artifacts:
  `secret_cases=d4005d89...ac88f`,
  `secret_manifest=70b89aed...535a`,
  `engine_snapshot=111e7cea...2657`; resume проверяет hashes и Arm D fingerprint.
- Drift root cause: метка `v4.0.0` не идентифицирует code/data state; source
  `_content_hash` и warehouse `content_hash` не совпали в 30/30, а сохранённые
  overall/axes описывают неоднородные snapshots. Поэтому старый warehouse
  snapshot не является Arm D baseline; Arm D для pilot - текущий replay с
  полным fingerprint.
- GCE pilot: 30 случаев × 2 = 60/60 валидных blind passes; 19 случаев
  KP-grounded и 11 no-KP; 0 parse/leakage/geo/runtime errors.
- Disagreement adjudication: 22/22 успешно (Dx 9, plan 13), 0 leakage/errors.
- Repeat agreement: Dx verdict 28/30, plan verdict 20/30, ICD fit 27/30,
  potential harm 30/30 для Dx и 29/30 для plan.
- Score repeatability: Dx median absolute difference 0, mean 9.04, max 99 п.п.;
  plan median 0.5, mean 9.71, max 45 п.п. Большие outliers требуют обязательной
  проверки методистом в C6; provisional methodology по LLM-only данным не выбирается.
- Production scoring, action queue, warehouse и UI не изменялись.

#### Подготовка C6 от 2026-08-09

- Secure blinded review pack создан на GCE для всех 18 случаев с
  расхождениями: 22 endpoint labels, Dx 9 и plan 13.
- Methodist видит только клинический case и разрешённый KP context; engine,
  два LLM passes и LLM adjudication не входят в review directory. Comparison
  создаётся только после успешной фиксации human labels.
- Public status не содержит case IDs или clinical text; secret directory mode
  `0700`, файлы `0600`.
- Текущий label gate: `complete_label_n=0/22`, `case_n=18`,
  `missing_n=0`, `extra_n=0`, `passed=false`.
- C6 не отмечен выполненным: требуется реальная разметка методистом и
  `passed=true`. LLM adjudication не считается human gold.
- Защищённая форма доступна по `/methodist/calibration`: только роли
  methodist/lead/admin, API `no-store`, server-side reviewer/timestamp,
  file lock + atomic replace и отдельный access audit без clinical text.
- UI не получает engine/LLM outputs и не хранит clinical case в browser
  storage. Comparison создаётся сервером только после 22/22 valid labels.
- Рабочий контур UI - GCE primary `protocol.kravira.by`, где уже находится
  secret pack. В Render backup clinical calibration artifacts не копируются.

### P0 - после pilot gate

- [ ] **S1** Один `reg55_section_pct` в таблице / case / axis regulatory
- [ ] **S2** Zones подписаны отдельно
- [ ] **S3** 0.5 = «Частично», 0 = «Не выполнено»
- [ ] **S4** Live attach только с полным payload
- [ ] **S5** Новые Endpoint C/D остаются shadow до C9

### P1 - после confirmatory gate

- [ ] **S6** Выбранный clinical selector становится primary queue signal
- [ ] **S7** Dx evidence и plan scores показываются отдельно в case review
- [ ] **S8** Осторожный ICD mismatch code
- [ ] **S9** LLM fallback маркируется как no-KP / lower trust
- [ ] **S10** Gold tests из adjudicated pilot без PHI

### P2 - полный пересчёт

- [ ] **S11** GCE recompute всего доступного горизонта
- [ ] **S12** Table №55 == case №55 на 100% smoke
- [ ] **S13** Измерить bad-MO recall, FN high-score и review load после rollout
- [ ] **S14** Handoff: SHA, versions, model, prompt/config hashes, metrics

---

## 12. Метрики успеха

| Метрика | Цель |
|--|--|
| Critical / harm recall | 100% на adjudicated sample |
| Bad-Dx recall | максимальный при согласованном review load |
| Bad-plan recall, KP matched | максимальный, с protocol source refs |
| KP-unmatched fallback coverage | 100% eligible, provenance `llm_no_kp` |
| Table №55 == case №55 | 100% после recompute |
| Partial 0.5 в «Невыполненные» | 0 |
| Blind prompt leakage | 0 |
| PHI в git / report | 0 |
| GCE LLM parse/geo errors | 0 после retry |

---

## 13. Следующая безопасная команда

```bash
scripts/ops/git_task_start.sh mo-score-calibration-harness --pc=pc1 \
  --branch=cursor/mo-score-calibration-harness-pc1
# Реализовать C0-C4. Production scoring не менять.
```
