# МО: подгрузка визита + поиск по visit_id / patient_id

## Контекст

Нужен приём `3468853` / patient `532264` (июнь) в МО Аналитике и поиск
по ID визита и пациента, даже если выбран период «вчера».

## Изменения

1. API/фильтры: `visit_id`, `patient_id`; числовой `q` ищет visit/mis/patient_key.
2. ID-поиск не режется `date_from`/`date_to`.
3. UI: placeholder + `visit_id:` / `patient_id:` / голый числовой запрос.
4. Ops: `scripts/ingest_mo_visit_from_mis.py` (только GCE).

## Метрики

| | Было | Цель |
|--|--|--|
| Поиск 3468853 при period=yesterday | 0 строк | 1 строка |
| patient_id 532264 | нет | находит визит(ы) |

## Статус

- [x] код поиска
- [ ] ingest 3468853 на GCE
- [ ] PR
