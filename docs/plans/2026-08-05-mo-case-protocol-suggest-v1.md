# Case → Protocol Suggest (v1)

Дата: 2026-08-05  
Статус: active (recall по МКБ superseded: `2026-08-07-mo-dx-text-suggest-icd-directory-eval-v1.md`)  
Автор: агент + владелец (по разбору МО Смирнова / KZ)  
Связанные:

- `2026-07-28-mo-daily-bi-platform-v1.md` - МО / КЗ контур;
- `2026-08-03-mo-rubric-mz-scoring-viz-v1.md` - оценка качества оформления (отдельно);
- `clinical_knowledge/protocol_match.py` - текущий подбор карточек по фактам КЗ (база, не замена этой спеки);
- эталон разбора: `Downloads/KZ/smirnova.pdf` (хромота 3 мес, отёк колена, M60).

---

## 1. Контекст и граница продукта

Нужен модуль **подсказки клинических протоколов МЗ РБ по случаю** (МО/КЗ):

- что открыть методисту / врачу рядом с кейсом;
- какие КП покрывают жалобы, статус, DDx и код МКБ;
- где диагноз/план **не закрывают** найденные находки.

Это **не** L1/L2 scorer качества оформления и **не** итоговый балл соответствия протоколу.

| | L1 / deep scorer | Case → Protocol Suggest |
|--|--|--|
| Цель | оценить, насколько КЗ/МО оформлен и полон | предложить релевантные КП МЗ |
| Выход | score, findings, severity | ranked list протоколов + reasons |
| Ошибка | ложный penalty врачу | нерелевантный КП в топе |
| Зависимость | rubric / rules | corpus cards + ICD profiles + case facts |

Текущий `match_protocol_cards()` - близкий фундамент (ICD/specialty/population). Эта спека добавляет:

1. явный **case fact graph** (жалобы, статус, разрывы);
2. **DDx-aware** ранжирование (не только код врача);
3. фильтры audience / care_setting / false positives;
4. UI-контракт для карточки МО.

---

## 2. Цель и метрики

### Цель MVP

Для кейса МО/КЗ за <2 с (без LLM) или <8 с (с optional LLM-rerank) вернуть top-3 КП с объяснением «почему».

### Метрики

| Метрика | Было | Цель MVP | Цель v2 |
|--|--|--|--|
| Hit@3 эталонного набора (ручная разметка, n≥30) | нет модуля | ≥70% | ≥85% |
| Доля ложных adult КП на pediatric case | н/д | <10% | <5% |
| Случаи с пустым suggest при наличии ICD в корпусе | н/д | <15% | <5% |
| Latency p95 без LLM | н/д | <2 с | <1 с |
| Согласие методиста «полезно» (thumbs) | н/д | ≥60% | ≥75% |

Эталон для калибровки v1: кейс Смирнова → ожидаемый порядок:

1. детский иммуновоспалительный/ревматический КП 2024 (M08 / ЮДМ);
2. детский ортопедо-травматологический КП 2024 (в т.ч. M91.1 Пертес);
3. КП с M60 (инфекция мягких тканей) - только как **слабый / code-only** match.

---

## 3. Вход: CaseFactGraph

Нормализованный JSON из МО/КЗ (те же слоты, что для анализа КЗ; без сырого `result` целиком).

```json
{
  "case_id": "visit:…",
  "audience": "pediatric|adult|unknown",
  "age_years": 9,
  "specialty": {"id": null, "slug": "travmatologiya-ortopediya", "label": "Ортопед-травматолог"},
  "care_setting": "outpatient|inpatient|unknown",
  "complaints": ["хромота на правую ногу"],
  "duration": {"value": 3, "unit": "month"},
  "anamnesis_flags": {"trauma": false, "allergy": false, "fever": null},
  "findings": [
    {"system": "knee", "side": "right", "sign": "edema", "severity": "P1"},
    {"system": "thigh_muscle", "side": "right", "structure": "rectus_femoris", "sign": "tenderness", "severity": "P2"}
  ],
  "diagnoses": [
    {"icd": "M60", "text": "Миозит прямой мышцы правого бедра", "role": "primary"}
  ],
  "plan": {
    "drugs": ["ibuprofen_ped"],
    "procedures": ["massage_thighs"],
    "follow_up": "on_worsening_only"
  },
  "gaps": []
}
```

### Чеклист извлечения фактов (детерминированно)

- [ ] Возраст / audience (`pediatric` если <18).
- [ ] Специальность врача (из визита / текста).
- [ ] Жалобы + длительность (regex/слоты).
- [ ] Анамнез-флаги: травма / лихорадка / аллергия / нагрузка (unknown, если нет текста).
- [ ] Находки status localis / суставы: отёк, боль, ограничение ROM, сторона.
- [ ] Диагнозы: ICD + free text; primary vs secondary.
- [ ] План: обследования / лекарства / контроль / «только при ухудшении».
- [ ] **Gaps** (авто): находка без отражения в диагнозе; длительность ≥4 недель без обследований в плане; pediatric limp без DDx-hints.

Правило gaps (пример из эталона):

```text
IF finding.sign == edema AND joint AND NOT diagnosis covers joint/synovitis/arthritis
  → gap.code = finding_not_in_diagnosis
IF duration_days >= 28 AND audience == pediatric AND complaint has limp
  AND plan has no imaging/labs AND follow_up == on_worsening_only
  → gap.code = underworkup_chronic_pediatric_limp
```

---

## 4. Алгоритм suggest (MVP, без LLM)

```text
CaseFactGraph
   │
   ├─(A) Candidate recall
   │     A1 ICD exact/root против protocol_icd_profiles + cards
   │     A2 specialty_slug / rubric path
   │     A3 filename/title lexical (миозит, ювенильн, колен, ортопед, детс_)
   │     A4 gap-driven DDx seeds (см. §4.2)
   │
   ├─(B) Hard filters
   │     B1 audience mismatch → drop или score*=0.05
   │     B2 administrative / unrelated rubric blacklist
   │     B3 adult joint replacement / endoscopy при pediatric → drop
   │
   ├─(C) Score
   │     C1 icd_fit
   │     C2 complaint/finding lexical overlap
   │     C3 specialty fit
   │     C4 population/audience fit
   │     C5 care_setting fit (outpatient vs «стац»)
   │     C6 gap_coverage: КП закрывает gap? (boost)
   │     C7 code_only_penalty: ICD hit без clinical overlap → demote
   │
   └─(D) Output top-K with reasons + confidence band
```

### 4.1 Веса score (стартовые, калибровать на эталоне)

| Компонент | Вес | Примечание |
|--|--|--|
| ICD fit | 0.30 | exact > root3 > chapter |
| Finding/complaint overlap | 0.20 | токены/словарь признаков |
| Gap coverage / DDx seed | 0.20 | закрывает ли КП выявленный разрыв |
| Specialty | 0.15 | path slug |
| Audience | 0.10 | pediatric/adult |
| Care setting | 0.05 | soft |

Нормализация 0-100. Порог показа: score ≥ 35 или top-1 всегда, если candidates > 0.

### 4.2 DDx seeds от gaps (таблица v1)

| Gap / паттерн | Seed ICD / темы | Пример КП |
|--|--|--|
| Pediatric limp ≥4 нед | M91.1, ортопед детский | 01КП детс ортопедо-травма 2024 |
| Joint edema + chronic symptoms, child | M08*, ювенильн артрит, ревматолог детский | КП иммуновоспал. ревм. заб. дет 2024 |
| Muscle pain + pediatric systemic flags | ЮДМ / миозит в ревмо-КП | тот же ревмо-КП |
| ICD M60 only, infection signs absent | soft-tissue infection КП | показать с `match_kind=code_only` |
| Trauma + joint | adult/ped trauma КП по audience | травма ОДА |

### 4.3 Reasons (обязательны в API)

Каждый hit:

```json
{
  "protocol_id": "…",
  "source_path": "minzdrav_protocols/…",
  "title": "…",
  "score": 78,
  "match_kind": "clinical|code_only|ddx|specialty",
  "reasons": [
    {"code": "gap_joint_edema", "text": "Закрывает отёк колена / маршрут ЮА (M08)"},
    {"code": "audience_pediatric", "text": "Детское население"}
  ],
  "covered_gaps": ["finding_not_in_diagnosis"],
  "warnings": ["care_setting_inpatient_protocol"]
}
```

---

## 5. API / UI контракт

### API (черновик)

`POST /api/methodist/mo/cases/{id}/protocol-suggest`  
или общий `POST /api/protocol/suggest` с телом CaseFactGraph.

Ответ:

```json
{
  "ok": true,
  "case_id": "…",
  "gaps": [{"code": "finding_not_in_diagnosis", "detail": "отёк правого колена"}],
  "items": [ /* top 3-5 */ ],
  "engine": "case_protocol_suggest_v1",
  "build_version": "…"
}
```

Не смешивать с полями L1 `overall_pct` / `deep.findings`.

### UI (карточка МО / КЗ)

- Блок «Протоколы к случаю» под диагнозом.
- Top-3: title, match_kind badge (`Клиника` / `Только код` / `DDx`), 1-2 reasons.
- Клик → существующий protocol viewer / PDF.
- Thumbs up/down → analytics (методстат).

---

## 6. Чеклист реализации

### Фаза S0 - спека и эталон (эта итерация docs)

- [x] Описать границу vs L1 scorer.
- [x] Зафиксировать CaseFactGraph + gaps.
- [x] Зафиксировать алгоритм recall → filter → score → reasons.
- [x] Эталон Смирнова (ожидаемый порядок КП).
- [ ] Собрать ещё ≥10 кейсов в `eval/case_protocol_suggest/` (без PHI в git: синтетика или хэш ID).

### Фаза S1 - engine MVP (код)

- [ ] `clinical_knowledge/case_protocol_suggest.py` (чистая функция от CaseFactGraph).
- [ ] Переиспользовать registry cards + `protocol_icd_profiles` + куски `protocol_match.py`.
- [ ] Не тащить scoring rubric L1.
- [ ] Unit-тесты на эталон Смирнова (fixtures с обезличенным fact graph).
- [ ] Endpoint + feature flag `CASE_PROTOCOL_SUGGEST=1`.

### Фаза S2 - UI методиста

- [ ] Блок в case detail МО.
- [ ] Badges match_kind + warnings.
- [ ] Feedback thumbs.

### Фаза S3 - optional LLM rerank

- [ ] Только после детерминированного top-20.
- [ ] Промпт: ранжировать и объяснить, **не** выдумывать пути PDF вне списка.
- [ ] Abort, если LLM предлагает path не из candidates.

---

## 7. Что изменено в проде

Пока: **ничего** (только план/спека).  
После S1/S2: новый endpoint + UI-блок; scorer и warehouse schema не меняются.

---

## 8. Риски

| Риск | Митигация |
|--|--|
| Путают suggest с L1 score | Разные API/UI; явный `engine` id |
| Adult КП на детях | hard filter audience |
| Code-only M60 выглядит как «лучший» | `match_kind=code_only` + demote |
| Стационарный КП на амбулаторный МО | warning `care_setting_*`, не drop |
| PHI в eval | только fact graph / synthetic |
| Конфликт с доработкой `protocol_match` | один владелец файла suggest; match остаётся библиотекой |

---

## 9. Владение файлами

Suggest владеет:

- `clinical_knowledge/case_protocol_suggest.py` (новый);
- `tests/test_case_protocol_suggest.py`;
- `eval/case_protocol_suggest/` (fixtures);
- UI-блок suggest в methodist case detail.

Не трогать без согласования: L1 batch, `mo_daily.assess_completeness`, publish pipeline.

---

## 10. Definition of Done MVP

1. Эталон Смирнова даёт порядок §2 (ревмо-дет ≥ ортопед-дет > M60 code-only).
2. API отдаёт gaps + reasons без L1 score.
3. Feature flag; выключено по умолчанию в проде до приёмки методистом.
4. Hit@3 на первом eval-наборе ≥70% или явный отчёт с разбором ошибок.

---

## 11. Первая безопасная команда после утверждения спеки

```bash
scripts/ops/git_task_start.sh case-protocol-suggest --pc=pc1 \
  --branch=codex/case-protocol-suggest-agent1-pc1
# затем S1: skeleton case_protocol_suggest.py + fixture Смирнова
```
