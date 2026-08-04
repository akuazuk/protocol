# Frontend

Frontend-контур проекта (web + patient UI).

## Текущий статус

Канонические HTML/CSS/JS лежат в `frontend/web/{doctor,methodist,patient,shared}`.
Прод URL (`/`, `/search-flow.css`, `/protocol-logo.svg`, `/methodist/mo`, …)
сохраняются через явные маршруты и `backend/frontend_paths.py`.

## Структура

- `frontend/web/doctor/` - doctor workspace
- `frontend/web/methodist/` - methodist dashboards
- `frontend/web/patient/` - patient PWA
- `frontend/web/shared/` - общие CSS/JS, brand assets, МО chrome, vendor

План: `docs/plans/2026-08-04-repo-sections-archive-v2.md` (фаза 2).
Предшественник cleanup: `docs/plans/2026-07-29-repo-structure-cleanup-v1.md`.
