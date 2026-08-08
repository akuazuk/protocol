# МО: неклинические визиты полностью вне таблиц и оценок

Статус: **active**  
Дата: 2026-08-08  
Преемник уточнения для: `2026-08-08-mo-clinical-visit-only-v1.md`

## Контекст

Если итог «Не оценивается: не клинический приём (процедура / диагностика /
профосмотр / стоматология)» - такие случаи:

1. **не включать в таблицы** случаев;
2. **не оценивать** (№55, МКБ, findings, рубрика);
3. **не подбирать КП**;
4. **не гонять LLM**.

Раньше фильтр «Только оцениваемые» был мягким (можно снять) и live case-detail
всё равно считал ICD/рубрику/suggest.

## Метрики

| | Было | Стало / цель |
|--|------|------|
| Таблица случаев | soft filter | только `clinical_visit` (жёстко) |
| Case detail analyzers | бегут для всех | skip если не clinical_visit |
| Protocol suggest | без гейта | unavailable для non-clinical |
| Warehouse soft-fill МКБ | для всех | только clinical_visit |
| L1 batch | fallback kz/certificate | document_kind → is_scored_document_kind |

## Шаги

1. [x] План
2. [x] Hard API/UI table gate
3. [x] Skip live ICD/№55/rubric/LLM/suggest
4. [x] Align mo_case_document + split_kz_rows + warehouse soft-fill
5. [x] Tests → PR (deploy после merge)

## Риски

- Воронка «типы документов» по-прежнему показывает excluded volume (это не таблица случаев).
- Прямой URL case_id non-clinical: документ виден, оценок нет.
- Warehouse может хранить non-clinical строки для ops; в case table API их нет.
