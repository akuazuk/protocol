# Handoff: MO dashboard nav cleanup (D6)

Дата: 2026-08-09
Branch: `cursor/mo-dashboard-nav-cleanup-pc1`
Plan: `docs/plans/2026-08-09-mo-dashboards-zones-first-v2.md`

## Сделано

- Удалены из DOM страницы: Специальности, Диагнозы, Безопасность, Кабинет врача, Журнал доступа, Качество данных.
- Меню: 6 видимых + hidden `settings` (accounts/admin).
- Журнал доступа и качество данных - secondary на экране Отчёты.
- Старые `?page=` редиректятся через `REMOVED_PAGES`.

## Координация

| Работа | Действие |
|--|--|
| Draft #77 | не трогали |
| Auth/settings | settings page retained |
| Doctor-cabinet API | без UI-страницы; API тесты не трогали |

## Тесты

`pytest tests/test_mo_dashboard_nav_cleanup.py tests/test_mo_frontend_structure.py::test_mo_dashboard_has_complete_crm_navigation tests/test_mo_dimensions.py::test_phase5_frontend_uses_real_echarts_and_selected_doctor_action_flow`
