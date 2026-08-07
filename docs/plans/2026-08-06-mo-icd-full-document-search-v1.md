# МО: МКБ искать по всему документу, не только в графе «Диагноз» (v1)

Дата: 2026-08-06  
Статус: active  
Источник требования: владелец (разбор gold-кейса `3650612` / экспертный review pack).  
Связанные: `2026-08-05-mo-eval-smirnova-concordance-v1.md`,
`2026-08-06-mo-case-findings-clarity-v1.md`, `2026-08-05-mo-methodist-review-pack-v1.md`.

---

## 1. Правило (источник истины)

При оценке **наличия / валидности кода МКБ-10** в МО (и КЗ, если тот же пайплайн):

1. Искать код **по всему тексту медицинского осмотра / консультативного заключения**,  
   а не только в поле / графе «Диагноз» (`clinical_diagnosis`, `diagnosis_short`,
   `mkb_code_main` из структурированного слота диагноза).
2. Код, найденный в жалобах, анамнезе, статусе, обследованиях, рекомендациях,
   манипуляциях или свободном тексте `result` - **считается присутствующим**
   для критериев «МКБ указан» / `B_icd_invalid` / `icd10_present` (reg55),
   если формат валиден.
3. Поле диагноза по-прежнему предпочтительно для **основного** кода и сверки
   с МИС; полный документ - обязательный fallback и источник для «код есть».
4. Не штрафовать `B_icd_invalid` / reg55 `icd10_present`, если валидный МКБ
   есть где-то в МО, даже при пустой графе диагноза.

Это продуктовое правило на **все будущие** правки оценки МО.

---

## 2. Где сейчас узко (аудит)

| Место | Сейчас | Нужно |
|--|--|--|
| `reg55_criteria._icd10_present` | только `diagnosis_short` | скан всех клинических слотов + raw blob |
| `kz_deep_eval` B2 / `kz_evaluation_engine` | `mkb_code_main` | resolve: main → extract по полному тексту МО |
| `mo_daily` / export `diagnosis_code` | слот диагноза МИС | optional fill из full-text, если слот пуст |
| LLM action-judge / промпты | часто смотрит блок диагноза | явно: «МКБ может быть в любом разделе» |
| Protocol Suggest | раньше тянул ICD в match | **не для suggest** - см. `2026-08-07-mo-dx-text-suggest-icd-directory-eval-v1.md`; full-doc только для оценки кода |

Вспомогательное уже есть: `extract_icd10` / `extract_mkb_code` / consult_parser
по полному тексту - переиспользовать, не плодить regex.

---

## 3. Метрики

| Метрика | Было | Цель |
|--|--|--|
| False `B_icd_invalid` когда код в не-диагнозном разделе | встречается | 0 на gold / ручной выборке ≥20 |
| False fail reg55 `icd10_present` при коде вне диагноза | да (по коду) | 0 |
| Регресс: пропуск реально пустого МКБ | - | не ухудшить recall на кейсах без кода нигде |

---

## 4. Шаги

- [x] P0: общий helper `resolve_icd_codes_from_mo(case) -> {main, all, sources}` (full-text)
- [x] P1: `_icd10_present` + deep/engine `B_icd_invalid` на helper
- [x] P2: тесты - код только в objective / recommendations → pass; нигде → fail
- [ ] P3: подтянуть `mkb_code_main` в warehouse/export только как soft-fill (не ломая MIS agreement)
- [ ] P4: одна строка в промптах LLM judge / methodist AI про полный документ

---

## 5. Риски

| Риск | Митигация |
|--|--|
| Ложный код из шаблона («см. MKB…») | тот же `_ICD10_RE` / `extract_mkb_code`; отсечь явные плейсхолдеры |
| Несколько кодов | `all` + main = приоритет слота диагноза, иначе первый валидный |
| Расхождение с МИС | `B_icd_mismatch_mis` отдельно; full-text не отменяет сверку |

---

## 6. Definition of Done

1. Правило зафиксировано здесь и в индексе планов. **done**
2. Helper + P1/P2 в коде с тестами. **done** (`mo_icd_resolve`, deep/engine/reg55/suggest)
3. На кейсе с МКБ вне графы диагноза нет ложного `B_icd_invalid`. **covered by unit tests**
4. P3/P4 (warehouse soft-fill, LLM prompts) - отдельно.
