# МО: КП по диагнозу эпизода из истории визитов + golden-тесты

Статус: **active**  
Дата: 2026-08-08  
Связано: `2026-08-08-mo-kp-suggest-dx-accuracy-v1.md` (suggest v4),  
`2026-08-08-mo-patient-history-bundle-v2.md` (лента пациента),  
`2026-08-08-mo-icd-dx-matching-pipeline-v3.md` (Dx↔МКБ).

## 1. Контекст и цель

Сверка МО с КП МЗ требует **правильно найти КП**. Suggest v4 уже ищет по тексту
диагноза **текущего** визита. Нужно:

1. Учитывать **историю визитов пациента** как эпизод по диагнозу (не все диагнозы жизни).
2. Иметь **регрессионный golden**: «КП найден верно» / «КП найден неверно» на реальном каталоге.
3. Потом подключить найденный КП к №55 (alignment), не смешивая с L1 оформления.

**Не цель:** отдельный поиск КП на каждый прошлый визит и union всех протоколов.

## 2. Продуктовые правила

| Правило | Смысл |
|--|--|
| Поиск по тексту Dx, не по `mis_diagnos` как gate | код случая не выбирает КП напрямую |
| Эпизод ≠ вся история | только визиты с близким Dx к текущему |
| Текущий визит первичен | история обогащает тот же эпизод; новый Dx не подменяет старым КП |
| Один suggest на случай | один query → top clinical КП |
| Без PHI в тестах/логах | синтетические visit_id, без ФИО/patient_id |

## 3. Целевой пайплайн в МО

```text
fact_mo_case (patient_key) + secure clinical текущего визита
        │
        ▼
build_patient_history_bundle(as_of=visit_date)   # уже есть v2
        │
        ▼
resolve_dx_episode_for_suggest(current, bundle)  # НОВОЕ
  • current_text = clinical_diagnosis / main / short (strip ICD)
  • prior_candidates = same_doctor ∪ same_specialty (date < as_of)
  • episode = prior, где:
      - ICD stem совпал (если оба есть), ИЛИ
      - combined_score(name) ≥ порога, ИЛИ
      - alias/token overlap (ПВУС↔плосковальгус)
  • query = expand( current_text + лучшие формулировки episode )
  • если current пустой/только код - взять лучший текст episode
  • если current явно другой Dx - episode не тащим (new_for_profile)
        │
        ▼
suggest_protocols_for_case(clinical{…}, record, episode_query=…)  # v4
        │
        ▼
case.protocol_suggest + (позже) reg55 KP criteria 39.2/39.5/39.8
```

### 3.1 Где считать

| Когда | Где |
|--|--|
| Case detail / API suggest | один раз на запрос; бандл уже на case |
| Batch deep eval | после `attach_bundle_to_case`, до/вместе с suggest |
| Night LLM | только читает готовый `protocol_suggest`, историю сам не собирает |

Флаги: `CASE_PROTOCOL_SUGGEST=1` (есть),  
`MO_KP_EPISODE_FROM_HISTORY=1` (новый, default on после merge).

### 3.2 Модули

| Файл | Роль |
|--|--|
| `clinical_knowledge/mo_dx_episode.py` | `resolve_dx_episode_for_suggest` |
| `clinical_knowledge/case_protocol_suggest.py` | принять episode query / history |
| `clinical_knowledge/mo_kp_suggest_golden_eval.py` | Hit@k / reject eval |
| `tests/fixtures/mo_kp_suggest_golden.jsonl` | эталоны верно/неверно |
| `tests/test_mo_kp_suggest_golden.py` | CI |

### 3.3 №55 (фаза B, после golden)

Когда top suggest `match_kind=clinical` и score ≥ порога:

- критерии alignment КП (39.2 / 39.5 / 39.8) сравнивают план/обследование с выбранным КП;
- если clinical hit нет → `n/a` или 0.5 «план есть, КП не верифицирован» (как сейчас честно).

Не блокирует фазу A (episode + golden).

## 4. Метрики

| Метрика | Было | Цель |
|--|--|--|
| Flatfoot child → КП детс ортопедо №109 | specialty filler | Hit@1 clinical |
| Flatfoot reject позвоночник/огнестрел | попадали в top | reject_path |
| ПВУС + история полного текста | пусто/filler | Hit@1 тот же КП |
| Новый Dx (ОРВИ) при старом flatfoot в истории | риск старого КП | top не детс-ортопедо |
| Golden pass rate (обязательные rows) | n/a | 100% на CI |

## 5. Шаги

### Фаза A - фундамент (этот коммит/ветка)

- [x] Suggest v4 (accuracy plan)
- [x] План episode+golden (этот файл)
- [x] `mo_dx_episode.py` + wire в suggest
- [x] Golden fixture + eval + pytest
- [x] Негативные rows (reject) + episode rows

### Фаза B - прод

- [x] API `/protocol-suggest` + review-pack: бандл → episode → suggest
- [x] Env `MO_KP_EPISODE_FROM_HISTORY` (default on)
- [x] Рубрика МЗ plan: reason/score учитывает clinical-hit suggest
- [ ] Полная текстовая сверка плана с PDF КП (после merge)
- [x] BUILD_VERSION + PR (smoke 3605554 на **GCP** после merge + `deploy_to_gce.sh`)

### Фаза C - расширение golden

- [ ] 20+ rows по частым специальностям МО (урология, педиатрия, кардио)
- [ ] Отчёт `scripts/eval_mo_kp_suggest_golden.py` (опционально)

## 6. Контракт golden-строки

```json
{
  "id": "flatfoot_child_ortho",
  "kind": "positive",
  "clinical": {"clinical_diagnosis": "...", "patient_age_years": 7},
  "record": {"specialty": "Ортопед-травматолог"},
  "history_visits": [],
  "expect_match_kind": "clinical",
  "expected_path_contains_all": ["детс", "ортопедо"],
  "reject_path_contains_any": ["огнестрельн", "позвоночник"],
  "min_top_score": 50
}
```

`kind`:

- `positive` - top должен попасть в expected; reject не должен быть в top-K;
- `negative` - expected пустой / expect_no_clinical / reject обязателен в смысле «не эти пути»;
- `episode` - есть `history_visits`; проверяем, что query/эпизод ведёт к правильному КП.

## 7. Риски

- Omnibus-КП: Hit по path-фрагментам, не по точному filename hash.
- Педиатр без rubric slug - поиск шире; reject-правила обязательны.
- Пороги episode similarity - калибровать на golden, не на одном кейсе.
- Не тащить PHI в jsonl.

## 8. Критерий «готово» для фазы A

1. План в индексе `docs/plans/README.md`.
2. `pytest tests/test_mo_kp_suggest_golden.py tests/test_mo_dx_episode.py` green.
3. Flatfoot + reject + episode ПВУС + «новый Dx не берёт старый КП» в fixture.
