# Handoff: methodist cabinet MO style

Дата: 2026-08-09

## Git

| | |
|--|--|
| branch | `cursor/methodist-cabinet-mo-style-pc1` |
| worktree | `/private/tmp/protocol-task-methodist-cabinet-mo-style-pc1` |
| BUILD_VERSION | `2026-08-09-100801Z-methodist-cabinet-style` |

## Сделано

- `methodist-cabinet.css`: full width, MO tokens scoped to methodist-mode, buttons/forms, hide doctor chrome + footer clutter + B2C/онко tabs
- Nav: Очередь / Анализ / Обзор / Поиск / Учётки + CTA «МО Аналитика»
- Absolute links: `/patient.html`, `/onco-risk.html`, `/docs/...`
- Plan + tests

## Следующее

Merge PR → `SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh` → smoke `/methodist/overview`
