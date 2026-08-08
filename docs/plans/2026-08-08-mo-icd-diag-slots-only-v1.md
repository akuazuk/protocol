# МО: МКБ и диагноз только из слотов «Клинический диагноз» / «Диагноз МИС»

Статус: **active**  
Дата: 2026-08-08  
Преемник для: `2026-08-06-mo-icd-full-document-search-v1.md` (архивировать)

## Контекст

Full-document поиск МКБ подхватывал коды из анамнеза / статуса / плана
(например Z98.8 «состояние после…»), из‑за чего в разборе и KPI оказывались
чужие диагнозы. Владелец: искать код и формулировку **только** там, где UI
связывает замечания - «Клинический диагноз» и «Диагноз МИС».

## Правило

1. Текст диагноза: `clinical_diagnosis`, `mis_diagnos` (+ алиасы `mis_diagnosis`,
   `diagnosis_mis`, `diagnosis_main_text`).
2. Код МКБ: явные `mkb_code_main` / `diagnosis_code` **или** код, извлечённый
   из этих же слотов диагноза.
3. Не сканировать complaints / anamnesis / objective / exam / treatment / raw_text.
4. Нет кода в слотах при наличии текста Dx - по-прежнему **не** дефект
   (`assess_icd_code_requirement`).

## Метрики

| | Было | Цель |
|--|--|--|
| Код из objective/plan в `resolve` main/all | да | нет |
| Ложный Dx/МКБ в чипе / suggest от чужого раздела | встречается | 0 на выборке |
| Код в clinical_diagnosis / mis_diagnos | ок | ок |

## Шаги

1. [x] План + индекс (archive full-doc)
2. [x] `mo_icd_resolve` slots-only + soft_fill
3. [x] reg55 how_checked текст
4. [x] Тесты
5. [ ] PR → GCE
