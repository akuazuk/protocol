# Handoff: сводка Лекарства/Анализы падала на складе

Дата: 2026-09-05

## Причина

`/api/methodist/mo/drugs-labs-kpis` читал `fact_mo_case.doctor_name`. Такой колонки нет
(есть `doctor_key` + `dim_doctor.doctor_fio`). SQLite давал 500, UI писал
«не удалось загрузить сводку».

В образ GCE не копировался `data/mo_finding_families/` - даже после фикса SQL
тайлы семейств были бы пустые.

## Сделано

- JOIN `dim_doctor`, чтение только `mode=ro`.
- COPY реестра семейств и аномалий в `deploy/gcp-app/Dockerfile`.
- Тест на каноническую схему склада без `doctor_name`.
