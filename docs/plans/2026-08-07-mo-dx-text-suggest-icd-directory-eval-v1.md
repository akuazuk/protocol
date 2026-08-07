# МО: КП по тексту диагноза; МКБ - отдельная оценка по справочнику (v1)

Дата: 2026-08-07  
Статус: active  
Источник: владелец (МО Аналитика - приоритет установленного диагноза).  
Связанные / частично superseded:

- `2026-08-05-mo-case-protocol-suggest-v1.md` - suggest; **recall по МКБ снимается**
- `2026-08-06-mo-protocol-suggest-titles-search-v1.md` - titles/search UI
- `2026-08-06-mo-icd-full-document-search-v1.md` - full-doc МКБ остаётся для **оценки наличия кода**, не для suggest
- `2026-08-05-mo-eval-smirnova-concordance-v1.md` - concordance / support Dx
- `icd_mkb.py`, `consult_criteria_enrichment.diagnosis_assessment_lines`

---

## 1. Продуктовое правило (источник истины)

В **МО Аналитика** два разных контура. Их нельзя смешивать.

| Контур | Вход | Цель | МКБ как ключ поиска |
|--|--|--|--|
| **A. Подбор КП** | установленный **клинический диагноз** (свободный текст) | какие протоколы МЗ открыть | **нет** - по МКБ КП не искать |
| **B. Оценка МКБ** | текст диагноза ± код в документе | есть ли такая нозология / код в **справочнике МКБ**, согласуется ли формулировка | да - сверка со справочником, **отдельный балл / finding** |

Дополнительно (не менять смысл без явного запроса):

- **Наличие/формат кода** в МО (`B_icd_invalid`, reg55 `icd10_present`, full-doc resolve) - остаётся **документационной** проверкой «код указан и похож на МКБ-10».
- **Согласованность клиники** (`B_dx_no_support`, shadow concordance) - «диагноз следует из жалоб/статуса», не справочник.
- **MIS agreement** (`B_icd_mismatch_mis`) - сверка с МИС, не каталог КП.

---

## 2. Почему сейчас ломается смысл

| Сейчас | Проблема |
|--|--|
| Suggest: `_WEIGHT_ICD = 0.40`, tier `clinical`/`code_only` требуют `icd_fit` | В топ попадают КП «по коду», даже если формулировка диагноза про другое |
| Fact graph сеет `diagnoses[].icd` из full-document resolve | Код из любого слота тянет чужие карточки |
| UI: «Подбор по МКБ, жалобам…» | Методист думает, что КП = МКБ |
| Справочник RU (`icd_mkb`) и `_title_match_score` живут в consult-enrichment | В МО Analytics нет отдельной оценки «диагноз ↔ справочник» |
| `B_icd_invalid` | Только формат/presence, не «такой диагноз есть в справочнике» |

Эталонный риск (Смирнова и др.): код M60 → soft-tissue КП, хотя текст диагноза / клиника требуют другого маршрута. По новому правилу КП ищутся **по тексту диагноза**; M60 проверяется отдельно в блоке МКБ.

---

## 3. Целевая модель оценок (МО Analytics)

Разделить оси в UI и в score/findings:

```text
1. Оформление / полнота блоков     (A_*)
2. Клиническая согласованность     (B_dx_*, concordance shadow)
3. МКБ-справочник (НОВОЕ)          (B_icd_dir_* / KPI)
4. План / безопасность             (C_*, medication)
5. Подбор КП (не балл врача)       protocol_suggest - только текст Dx
```

### 3.1 Новая оценка «Диагноз ↔ справочник МКБ»

Вход: текст установленного диагноза (`clinical_diagnosis` / `mis_diagnos`); код, если есть (full-doc), **не** обязателен для прохождения текста.

Выход:

| Поле | Смысл |
|--|--|
| `directory_hit` | текст диагноза находит кандидатов в RU-справочнике |
| `code_in_directory` | указанный код есть в справочнике |
| `text_rubric_fit` | overlap формулировки с `ru_title(code)` |
| `verdict` | `ok` / `review` / `fail` |
| `score_pct` | 0-100, отдельный KPI |

Findings: `B_icd_dir_no_match`, `B_icd_dir_code_unknown`, `B_icd_dir_text_mismatch`.

Helper: `clinical_knowledge/mo_icd_directory_eval.py`.  
Пороги: ok ≥ **0.35**, review ≥ **0.25**. Shadow default; primary только `MO_ICD_DIR_IN_PRIMARY=1`.

### 3.2 Подбор КП - только диагноз

Engine `case_protocol_suggest_v3`: `icd10=[]`, `use_icd=False`, reasons `diagnosis_fit` / specialty. UI: «Подбор по установленному диагнозу».

---

## 4. Метрики

| Метрика | Было | Стало | Цель v1 |
|--|--|--|--|
| Reasons `icd_fit` в suggest | да | **0** | 0% |
| Engine | v2 ICD-first | v3 text Dx | ok |
| Directory findings shadow | нет | `B_icd_dir_*` | ok |
| Overall без флага | не трогали | не трогаем | без `MO_ICD_DIR_IN_PRIMARY` |

---

## 5. Шаги реализации

### Фаза A - контракт и docs

- [x] Зафиксировать правило: КП ≠ МКБ; МКБ = отдельная оценка справочника
- [x] Пороги `text_rubric_fit` / `directory_hit`: 0.35 / 0.25 (consult), shadow default
- [x] UI-копирайт в `mo-app.js`

### Фаза B - Protocol Suggest (P0)

- [x] `build_case_fact_graph`: без ICD в diagnoses
- [x] `match_protocol_cards_by_diagnosis_text` / `use_icd=False`
- [x] `_match_kind` / `_rank_rows` без `icd_fit`; без `code_only`
- [x] Reasons `diagnosis_fit`; тесты + UI

### Фаза C - Оценка справочника МКБ (P1)

- [x] Helper `evaluate_diagnosis_against_icd_directory`
- [x] deep shadow + live merge в case detail
- [x] Labels RU; shadow default
- [x] Unit-тесты

### Фаза D - Калибровка (P2)

- [x] Unit-эталоны: цистит ok / мусор fail / text mismatch; suggest text-over-wrong-ICD
- [ ] Ручная проверка ≥20 кейсов методистом после деплоя
- [ ] soft-fill warehouse KPI - позже при необходимости

---

## 6. Риски

| Риск | Митигация |
|--|--|
| Пустой Dx → пустой suggest | fallback complaints/specialty + search_url |
| Путаница двух оценок МКБ | shadow badge + source `mo_icd_directory_v1` |
| Регресс code-centric Hit@3 | принято по продукту |

---

## 7. Definition of Done v1

1. Suggest без МКБ reasons/score - **done (код)**
2. Shadow оценка справочника - **done (код)**
3. Планы помечены - **done**
4. Тесты + deploy - в этом релизе
