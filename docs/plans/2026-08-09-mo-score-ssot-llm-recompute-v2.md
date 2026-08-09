# МО: калибровка оценки → один №55 → полный пересчёт (v2)

Дата: 2026-08-09
Статус: **active** (согласование калибровочного протокола)
Преемник: `2026-08-09-mo-score-ssot-llm-recompute-v1.md`

Связанные планы:
- `2026-08-09-mo-reg55-section-pack-v1.md` - один section-pack №55;
- `2026-08-08-mo-icd-absent-ok-with-dx-v1.md` - отсутствие кода при наличии Dx не дефект;
- `2026-08-09-mo-icd-name-multidx-brief-v1.md` - осторожная проверка МКБ;
- `2026-08-09-mo-case-review-quality-parity-v1.md` - разбор случая;
- `2026-08-07-by-home-gcp-llm-split-v1.md` - живой Gemini только на GCE.

---

## 1. Почему сначала калибровка

По визиту `3643940` в production warehouse одновременно записаны:

| Метрика | Значение |
|--|--|
| `reg55_section_pct` | **83.3** |
| `fact_mo_score_axis.regulatory` | **77.8** |
| `overall_pct` | **60.0** |
| zone1 / zone2a / zone2b | **92.9 / 100 / n/a** |
| weak points | `C4`, `D3` |

Это не четыре независимых подтверждения. Часть показателей считает разные цели, часть
повторно использует одни и те же признаки, часть получена старым контуром.

До изменения формул и полного пересчёта нужно:

1. Зафиксировать снимок текущих оценок.
2. Независимо оценить 30 МО по клиническому тексту без подсказок движка.
3. Разобрать расхождения движок ↔ LLM ↔ методист.
4. Определить, какая отдельная оценка или простая совокупность лучше обнаруживает
   плохое МО при приемлемом объёме ручной проверки.
5. Только затем менять primary-формулу и пересчитывать весь warehouse.

---

## 2. Две разные целевые задачи

Нельзя выбирать одну метрику сразу для двух разных вопросов.

### Endpoint A - плохое МО в целом

МО считается плохим, если подтверждён хотя бы один **клинически существенный** дефект:

- опасное упущение / потенциальный вред;
- диагноз не подтверждается данными или противоречит им;
- код МКБ явно относится к другой клинической сущности;
- отсутствует необходимый план обследования / лечения / наблюдения;
- критически неполная запись, из которой нельзя восстановить врачебное решение.

Отсутствие кода при содержательном тексте диагноза само по себе **не дефект**.

### Endpoint B - соответствие постановлению №55

Это отдельная нормативная оценка:

- только `evaluate_reg55_section`;
- роль / специальность / тип записи → section pack;
- score `0 / 0.5 / 1`, `n/a` вне знаменателя;
- один процент и один band;
- частичное выполнение (`0.5`) не называется «невыполнено».

**Решение:** для продукта допустимы два результата - «клиническое качество» и «№55».
Один общий индекс можно оставить только как навигационный приоритет, если эксперимент
покажет пользу. Он не заменяет две исходные оценки.

---

## 3. Дизайн пилота: 30 МО

### 3.1 Период и отбор

- Основной период: `2026-08-01..2026-08-08`.
- Seed: `42`; выбор воспроизводимый.
- Все клинические тексты и соответствие `visit_id ↔ visit_ref` остаются только на
  GCE `/var/data/medical_exams/calibration/<run_id>/`, права `0600`.
- В репозиторий и отчёт попадают только `visit_ref`, числовые оценки, verdict и
  агрегированные метрики.

### 3.2 Страты

Целевой размер - **30 уникальных случаев**:

| Страта | Цель |
|--|--:|
| `overall_pct` 0-49 | 4 |
| `overall_pct` 50-59 | 4 |
| `overall_pct` 60-69 | 4 |
| `overall_pct` 70-79 | 4 |
| `overall_pct` 80-100 | 4 |
| Высокий балл, но action-queue / P0-P1 finding | 3 |
| Расхождение `reg55_section_pct` и axis regulatory ≥5 п.п. | 3 |
| `reg55 ≥80`, но weak points / `D_reg55_gap` | 2 |
| Существующий methodist gold (`training_use=1`) | все доступные, целевой минимум 2 |

Дополнительные ограничения:

- минимум 4 специальности;
- минимум 4 случая с МКБ ↔ Dx спором;
- минимум 4 случая без matched КП / zone2b n/a;
- не более 3 записей одного врача;
- `3643940` включить как обязательный sentinel.

Страты могут пересекаться; sampler добирает до 30 по score band и специальности.

---

## 4. Три независимых слоя оценки

### Arm D - текущий детерминированный движок

Снимок **до** изменения методики:

- `overall_pct`, `overall_pct_v3`;
- axes: documentation / clinical_concordance / safety / regulatory;
- zone1 / zone2a / zone2b + bands;
- `reg55_section_pct`, `reg55_band`, applicable_n, weak points;
- action-queue membership и P0/P1 findings;
- scorer / schema / config versions.

Повторный детерминированный прогон выполняется на том же полном clinical payload.
Если snapshot и replay различаются, это отдельный дефект воспроизводимости.

### Arm L - слепой LLM-review на GCE

LLM получает только очищенные клинические слоты:

- жалобы;
- анамнез;
- объективный статус;
- обследования;
- диагноз и МКБ;
- лечение;
- рекомендации / follow-up;
- специальность, возрастная группа и тип документа.

LLM **не получает**:

- `overall_pct`, zones, №55, findings;
- queue reason / severity;
- названия сработавших правил;
- итог другого LLM.

Для каждого случая выполняются:

1. Blind pass A - полнота + диагноз + МКБ.
2. Blind pass B - обследования + лечение + наблюдение.
3. Независимый повтор тем же зафиксированным model/version.
4. При расхождении повторов - adjudication pass с исходным текстом, но без движка.

Выход:

- verdict по каждому блоку: `good / partial / poor / critical / n/a`;
- `bad_mo` и `potential_harm`;
- список дефектов с evidence;
- уверенность;
- отдельная оценка №55 по пунктам только как **shadow comparison**.

Живые вызовы - только GCE (`protocol-app`, `europe-central2-a`).
Локальный Mac не вызывает Gemini.

### Arm H - методист / adjudication

LLM - рецензент, не gold.

Методист проверяет:

- все случаи, где Arm D и Arm L расходятся по `bad_mo`;
- все `critical` / `potential_harm`;
- все 4+ случая МКБ ↔ Dx;
- все случаи `reg55 ≥80` с «невыполненными»;
- случайную контрольную выборку минимум 5 согласованных случаев.

Форма решения:

- bad МО: `yes / no / partial / insufficient_data`;
- диагноз, МКБ, план, safety - отдельные verdict;
- №55: подтверждение каждого спорного `0 / 0.5 / 1`;
- finding decision: `confirmed / rejected / partial`;
- комментарий без ФИО / patient_id.

**Reference label пилота:** решение методиста. Если его пока нет, LLM-label считается
`provisional`, не используется для утверждения production-порогов.

---

## 5. Какие оценки сравниваем

### 5.1 Одиночные кандидаты

1. `overall_pct` (текущий primary).
2. `overall_pct_v3`.
3. Каждая axis отдельно.
4. `min(axis documentation, clinical, safety)`.
5. `min(zone1, zone2a, zone2b)` только по applicable zones.
6. `reg55_section_pct` - только для Endpoint B.
7. Action-queue boolean.
8. Наличие подтверждённого P0/P1.

### 5.2 Простые совокупности

Не обучаем сложную модель на 30 наблюдениях. Сравниваем заранее заданные правила:

| Кандидат | Правило |
|--|--|
| `ensemble_union_70` | overall <70 **или** любая applicable zone <70 **или** P0/P1 |
| `ensemble_union_80` | overall <80 **или** reg55 <80 **или** любая zone <70 |
| `clinical_gate` | safety <85 **или** clinical <70 **или** confirmed P0/P1 |
| `review_priority` | minimum из overall, applicable zones и reg55 |
| `queue_current` | текущий action-queue selector |

`reg55` не должен ухудшать клинический Endpoint A из-за чисто формального пункта.
Поэтому отдельно считаем ensemble с №55 и без №55.

---

## 6. Как выбираем лучшую оценку

### 6.1 Primary-критерий

Для Endpoint A главный критерий - **recall подтверждённых плохих МО**, особенно
`critical / potential_harm`.

Порядок выбора:

1. 100% recall critical/potential-harm на пилоте (или документированная причина промаха).
2. Максимальный recall bad МО.
3. При сопоставимом recall - выше precision и меньше review load.
4. Не выбирать метрику, которая хорошо работает только на одной специальности.

### 6.2 Метрики

- sensitivity / specificity / precision / NPV;
- false-negative rate среди случаев с исходным score ≥80;
- PR-AUC и ROC-AUC (только как pilot, с широкими CI);
- Spearman ρ и MAE для непрерывной оценки;
- quadratic weighted kappa для ordinal verdict;
- review load: доля случаев, отправленных методисту;
- recall по отдельным типам: diagnosis / ICD / plan / safety / №55.

Для всех ключевых метрик:

- paired stratified bootstrap 1000 повторов;
- 95% percentile CI;
- exact binomial CI для малых долей;
- показывать числитель / знаменатель рядом с процентом.

При `n=30` результат называется **направлением**, не окончательной победой.

### 6.3 Gate решения

| Gate | Условие |
|--|--|
| C0 smoke | 5 случаев; blind prompt не содержит scores/findings; нет `_error` |
| C1 pilot | 30 случаев; ≥15 adjudicated методистом; все disagreement рассмотрены |
| C2 методика | выбран интерпретируемый candidate; нет critical FN; CI опубликованы |
| C3 confirm | ≥100 случаев или ≥30 bad cases; повтор на другом периоде |
| C4 rollout | только после C2: P0 SSOT/UI; полный recompute только после C3 либо явного решения владельца |

---

## 7. Файлы и инструменты калибровки

Новые:

| Файл | Назначение |
|--|--|
| `scripts/build_mo_score_calibration_sample.py` | стратифицированная PHI-safe выборка |
| `scripts/run_mo_calibration_blind_judge.py` | blind LLM A/B + repeat + adjudication на GCE |
| `scripts/eval_mo_score_calibration.py` | paired metrics, bootstrap, candidate comparison |
| `eval/mo_score_calibration/README.md` | pre-registered protocol |
| `tests/test_mo_score_calibration*.py` | sampler / leakage / metrics |
| `docs/reports/YYYY-MM-DD-mo-score-calibration-pilot.md` | агрегированные результаты |

Переиспользовать:

- `scripts/build_kz_gold_sample.py`;
- `clinical_knowledge/kz_gold_annotation.py`;
- `scripts/calibrate_deep_thresholds.py`;
- `scripts/export_mo_review_gold.py`;
- `clinical_knowledge/mo_llm_action_judge.py`;
- `deploy/gcp-llm/run_on_gce.sh`.

Secret artifacts:

```text
/var/data/medical_exams/calibration/<run_id>/
  sample_secret.jsonl
  engine_snapshot.jsonl
  blind_llm_pass1.jsonl
  blind_llm_pass2.jsonl
  blind_llm_adjudication.jsonl
  methodist_labels.jsonl
```

В git:

```text
eval/mo_score_calibration/<run_id>/
  strata_summary.json
  metrics.json
  candidate_comparison.json
```

Никаких ФИО, patient_id, raw visit_id и клинических цитат в git / PR / handoff.

---

## 8. Этапы работ

### C - калибровка перед изменением production-оценки

- [ ] **C0** Реализовать sampler + public/secret manifest; выбрать 30 МО seed=42
- [ ] **C1** Зафиксировать engine snapshot и проверить replay reproducibility
- [ ] **C2** Реализовать blind prompt; тест на отсутствие leakage
- [ ] **C3** GCE smoke 5 случаев × 2 passes
- [ ] **C4** GCE pilot 30 случаев × 2 passes + adjudication disagreements
- [ ] **C5** Получить methodist labels минимум для 15 случаев и всех disagreements
- [ ] **C6** Сравнить одиночные scores и ensembles; bootstrap CI
- [ ] **C7** Опубликовать PHI-safe pilot report и выбрать методику / пороги
- [ ] **C8** Confirmatory cohort ≥100 или ≥30 bad cases

### P0 - единая правда №55 и честный UI

- [ ] **S1** Таблица и разбор читают один `reg55_section_pct` + band
- [ ] **S2** Hero подписан «Зоны методики», не «№55»
- [ ] **S3** `score=0.5` → «Частично», `score=0` → «Не выполнено»
- [ ] **S4** Live attach только с полным clinical; иначе warehouse snapshot
- [ ] **S5** `axes.regulatory := reg55_section_pct` в единственном writer

### P1 - МКБ / C4 / D3 после pilot

- [ ] **S6** Отсутствие МКБ при Dx = ok; неизвестный код = defect
- [ ] **S7** Явное несоответствие кода клинике - отдельный осторожный finding
- [ ] **S8** LLM grey-zone только для спорных ICD / treatment / follow-up
- [ ] **S9** Gold tests из adjudicated pilot без PHI

### P2 - пересчёт

- [ ] **S10** После gate C3 пересчитать весь доступный горизонт на GCE
- [ ] **S11** Проверить table №55 == case №55 на 100% smoke sample
- [ ] **S12** Повторно измерить distribution shift, queue load, bad-MO recall
- [ ] **S13** Handoff: версия, SHA, модель, prompt hash, config hash, метрики

---

## 9. Метрики успеха

| Метрика | Цель пилота / rollout |
|--|--|
| Blind prompt leakage | 0 полей scores/findings/queue reason |
| Engine replay mismatch | 0/30 или объяснённый version drift |
| Critical / harm recall | 100% на adjudicated sample |
| Bad-MO recall | максимальный при review load, согласованном владельцем |
| Table №55 == case №55 | 100% после recompute |
| «Невыполнено» для partial 0.5 | 0 |
| LLM geo / parse errors | 0 после retry-errors |
| PHI в git/report | 0 |

---

## 10. Риски

| Риск | Митигация |
|--|--|
| LLM повторяет движок | blind prompt, без system score / findings |
| Один LLM не независим | два passes + methodist adjudication |
| n=30 недостаточно | pilot → confirm ≥100; CI и числители в отчёте |
| Selection bias action queue | обязательные high-score и non-queue cases |
| Переобучение composite | только заранее заданные простые правила |
| PHI в логах | GCE-only secret dir; public numeric export |
| Gemini geo с Mac | только `run_on_gce.sh` / container on `protocol-app` |
| Полный пересчёт до проверки | hard gate C3 перед S10 |

---

## 11. Следующая безопасная команда после согласия

```bash
scripts/ops/git_task_start.sh mo-score-calibration-harness --pc=pc1 \
  --branch=cursor/mo-score-calibration-harness-pc1
# Только C0-C3: sampler, blind prompt, eval harness, GCE smoke 5.
# Формулы production и warehouse пока не менять.
```
