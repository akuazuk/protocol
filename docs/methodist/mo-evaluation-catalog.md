# Каталог оценок МО (источник для дашбордов)

Дата: 2026-08-08  
Код: `origin/main` + план `docs/plans/2026-08-08-mo-history-scores-catalog-v1.md`

Этот документ перечисляет **все** виды оценок / метрик Medical Oversight в Protocol.
Его нужно использовать при переделке дашбордов, таблиц, фильтров и диаграмм:
каждая колонка / KPI / фильтр должна ссылаться на строку каталога.

Гейт по умолчанию: оценки только для `document_kind ∈ {clinical_visit, consultation}`
(`score_eligible` / `mo_score_eligible`). Остальные виды документов в таблицы скоринга
не попадают.

---

## 0. История пациента (контекст, не primary-балл)

| Поле | Описание |
|--|--|
| API | `patient_history`, finding `B_patient_history_context` |
| Warehouse | `fact_mo_case.history_prior_n`, `history_tier`, `patient_key`; кэш `fact_mo_patient_history_cache` |
| Код | `clinical_knowledge/mo_patient_history_bundle.py` |
| UI | блок «История пациента» в разборе; чип в таблицах дня/очереди |

### Когда строится

1. При upsert дня в warehouse (если есть `patient_id` → `patient_key`).
2. В `build_case_detail` (черновик).
3. **Обязательно заново** в `GET /api/methodist/mo/cases/{id}` после identity lookup
   (`patient_id` / `doctor_id`), иначе бандл часто пустой.

Условия пустоты: нет `patient_id`/`patient_key`, нет `visit_date`, нет более ранних
визитов на складе, `MO_PATIENT_HISTORY_BUNDLE=0`, склад недоступен.

### Полки и tier

| Tier | RU | Смысл |
|--|--|--|
| `known_to_doctor` | код уже был у этого врача | текущий МКБ встречался у того же врача |
| `known_in_specialty_only` | код был у коллег специальности | у врача код новый, у коллег специальности уже был |
| `new_for_profile` | новый код для профиля у врача | у врача были визиты, но этот код новый |
| `first_contact` | первый контакт с этим врачом | `n_same_doctor = 0` |
| `insufficient` | истории недостаточно | нет ключа пациента / данных |

Полки UI: к этому врачу / другие врачи специальности / прочие специальности.
Горизонт: вся лента склада as-of даты визита (не «окно 90д», если не задан
`MO_PATIENT_HISTORY_LOOKBACK_DAYS`).

### Как используется в оценках

| Потребитель | Влияние |
|--|--|
| Итоговая оценка / оси | **Нет** по умолчанию (`shadow`). Вкл. primary: `MO_PATIENT_HISTORY_IN_PRIMARY=1` |
| Finding `B_patient_history_context` | одно замечание P2/P3 по tier; в списке findings |
| Name-match МКБ | порог мягче (−0.05), если код уже был у врача; жёстче (+0.05) для first_contact / new_for_profile |
| Concordance | `history_dx_line_break`, если код новый для профиля |
| Подбор КП | эпизод Dx из prior визитов (`case_protocol_suggest` / `mo_dx_episode`) |
| Рубрика МЗ | критерии `exam_correction` / `treatment_correction` требуют prior clinical текст (отдельный prior-path) |
| LLM action queue | boost приоритета для first_contact / new_for_profile |
| Чипы таблиц | `history_prior_n` / `history_tier` |

---

## 1. Primary / итоговые баллы случая

### 1.1 `overall_pct` (витрина)

- **UI:** «Итоговая оценка», колонки дня/очереди/месяца.
- **Код:** `mo_daily.case_overall_pct` → `fact_mo_case.overall_pct`.
- **Метод:** приоритет: `evaluation_v4.score_pct` (если primary) → `deep.overall_pct` →
  `evaluation_v3.score_pct` → L1 `overall_pct` / `core_overall_pct`. Шкала 0–100.
  Risk-cap от findings (типично P0→40, P1→60).
- **История:** косвенно, только если primary-флаг истории включён.

### 1.2 Deep overall (`deep.overall_pct` / `deep_overall_pct`)

- **Код:** `kz_deep_eval.evaluate_kz_deep`.
- **Метод:** среднее доступных осей A/B/C/(D=№55), затем risk-gate по findings.
- **UI:** KPI в разборе («по доступным данным»).

### 1.3 Scorer v3 (`evaluation_v3.score_pct`)

- **Код:** `kz_evaluation_engine.evaluate_kz_v3`.
- **Метод:** веса осей doc 0.30 / concordance 0.35 / safety 0.25 / regulatory 0.10
  (перенормировка). Caps при пустом Dx/recs/objective. Trust A/B влияет на штрафы.
  Coverage/confidence отдельно (0–1). Env `KZ_EVALUATION_V3_*`.

### 1.4 Scorer v4 (`evaluation_v4.score_pct`)

- **Код:** `kz_evaluation_v4` + `config/mo_scorer_v4.yaml`.
- **Метод:** те же детекторы, что v3; веса 0.25 / 0.35 / 0.30 / 0.10;
  risk_caps P0=40 P1=60; `attention_required` при P0/P1 или низких coverage/confidence.
  Env `KZ_EVALUATION_V4_PRIMARY` (default off).

---

## 2. Оси (0–100)

| Поле | RU | Метод |
|--|--|--|
| `axes.documentation` | Оформление | Наличие полей (обязательные / условные / рекомендуемые) + hard caps |
| `axes.clinical_concordance` | Согласованность | Поддержка Dx, МКБ, покрытие осмотра/лечения/критериев по КП (trust-aware) |
| `axes.safety` | Безопасность | Старт 100, минус штрафы severity; DDI / NSAID / high-alert / red flags |
| `axes.regulatory` | Регуляторика (№55) | = `regulatory_compliance_pct` из §3 |

Warehouse: `fact_mo_score_axis(mis_id, axis, score)`.

---

## 3. Постановление №55

| Поле | `reg55.regulatory_compliance_pct` / `reg55_pct` |
|--|--|
| Код | `reg55_criteria.evaluate_reg55` ← `data/regulations/mz_2021_55.json` |
| Метод | pass/fail/na по пунктам; **100 × passed / applicable**; `na` и `score_eligible:false` вне знаменателя |
| Findings | `D_reg55_p0`, `D_reg55_gap` |
| UI | колонка «Балл №55», блок критериев в разборе, KPI месяца |
| История | нет |

---

## 4. Рубрика МЗ «Как оценивать» (shadow)

| Поле | `rubric_mz.rubric_pct`, критерии `score ∈ {0, 0.5, 1, n/a}` |
|--|--|
| Код | `mo_rubric_mz` + `config/mo_rubric_mz.yaml` |
| Метод | 13 критериев полноты/глубины (в основном №127); % по scored (без n/a) |
| Критерии | `mo_complete`, `datetime`, `complaints`, `anamnesis`, `risk_factors`, `objective`, `exam_data`, `diagnosis`, `exam_plan`, `treatment_plan`, `exam_correction`, `treatment_correction`, `follow_up` |
| История | **да** для `exam_correction` / `treatment_correction` (нужен prior clinical) |
| UI | таблица в разборе; сводка месяца; фильтр по критерию |
| Роль | shadow (`primary: false`); отдельно от №55 |

---

## 5. Meta: coverage / confidence / attention

| Поле | RU | Метод |
|--|--|--|
| `coverage.overall` / `coverage_pct` | Полнота проверки | доля оцениваемых частей осей (0–1 → %) |
| `confidence.overall` / `confidence_pct` | Надёжность | среднее doc_parse / protocol_match / evidence_match / protocol_knowledge |
| `attention_required` | требует внимания | P0/P1 или coverage/confidence &lt; 0.5 (v4) |

---

## 6. Legacy L1 / блоки протокола

| Поле | Метод |
|--|--|
| L1 `overall_pct` / `core_overall_pct` | `compliance_engine` + `scoring.compute_overall`, веса `config/compliance_weights.yaml` |
| `block_scores.*` | выравнивание текста случая к блокам КП (diagnosis/exams/treatment/follow_up…) |
| Потребители | №55 (`alignment_min`), рубрика (планы), fallback UI |

---

## 7. LLM-оценки

### 7.1 Night Gemini grader

- Файлы: `*_llm_grades.jsonl`; скрипт `scripts/grade_kz_llm.py` (**только GCE**).
- Поля: `overall_pct`, оси documentation/concordance/safety, `verdict`, checklist.
- Не является формулой drawer primary; идёт в BI / обучение.

### 7.2 Action-queue LLM judge

- Код: `mo_llm_action_judge.py`.
- Поля: `completeness.score_pct`, `diagnosis_assessment.score_pct`, `plan_assessment.score_pct`.
- Shadow JSONL; UI блок в разборе; history tier может влиять на очередь.

### 7.3 ICD LLM review

- Код: `mo_icd_llm_review.py` (GCE).
- Findings `B_icd_llm_review_{yes,partial,no}`; chip пайплайна МКБ.

---

## 8. МКБ / concordance / подбор КП

| Слой | Поле / код | Метод | История |
|--|--|--|--|
| Directory fit | `icd_match.score_pct`, `B_icd_dir_*` | токен/title vs справочник МКБ; код и текст Dx **только** из `clinical_diagnosis` / `mis_diagnos` (не весь МО) | порог ± от summary |
| Name match | `mo_icd_name_match`, `weak_name` | сходство текста Dx ↔ title_ru (те же слоты) | −0.05 / +0.05 |
| Visit status chip | `icd_visit_status` | enum: ok / missing_dx / not_in_directory / weak_name | нет |
| Concordance findings | `mo_concordance_findings` | правила status↔Dx↔план; shadow default | `history_dx_line_break` |
| Protocol suggest | `protocol_suggest.items[].score` | ICD-first + text bridge v5 | эпизод Dx из истории |
| KP golden (offline) | `mo_kp_suggest_golden_eval` | right/wrong vs gold | episode-aware |

---

## 9. Очередь / CRM (производные, не клинический scorer)

| Поле | Метод |
|--|--|
| Action band critical/important | whitelist точных сигналов (`mo_action_queue_select` v2): red flags, Major DDI, high-alert без дозы, NSAID dup, `B_dx_no_support` / `B_dx_absent`, uncertainty unrouted. **Не** тикетует №55/ICD alone |
| `priority_from_score` | пороги по overall/осям (&lt;40 / &lt;60 / &lt;75); display/demote |
| CRM status / assignee / due | человеческий workflow |

---

## 10. Агрегаты BI (не отдельные scorers)

Средние `avg_overall`, `avg_coverage`, `avg_confidence`, `avg_rubric_pct`, средние осей,
частоты findings, heatmap врачей/специальностей/диагнозов - всё это производные от
case-level полей выше, с фильтром `score_eligible`.

---

## 11. Рекомендуемые колонки/фильтры для новых дашбордов

**Primary (одна «главная» метрика на таблицу):** `overall_pct`  
**Regulatory:** `reg55_pct`  
**Shadow depth:** `rubric_mz.rubric_pct`  
**Axes (развернуть):** documentation / concordance / safety / regulatory  
**Meta:** coverage_pct, confidence_pct  
**Context chips:** `icd_visit_status`, `history_tier` (+ `history_prior_n`)  
**Queue:** action band / whitelist reason (не сырой P0–P3)  
**LLM (отдельная вкладка):** night grader + action judge (не смешивать с primary %)

Фильтры, которые стоит сохранить/добавить:

- `document_kinds` / `score_eligible_only` (жёстко clinical+consultation)
- `history_tier`
- `icd_visit_status`
- `rubric` criterion fail
- `reg55` critical fail
- action-queue reason codes
- оси ниже порога

---

## 12. Что не смешивать в одном KPI

1. Рубрика МЗ (0/0.5/1, №127) ≠ №55 (pass/fail).
2. Night LLM / action judge ≠ warehouse `overall_pct`.
3. История пациента ≠ «оценка врача»; это контекст + shadow finding.
4. L1 block_scores ≠ итоговая deep/v4 оценка.
5. Protocol suggest score ≠ compliance / оформление.
