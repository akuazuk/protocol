# Handoff: protocol navigator hotfix (CSS + corpus)

Дата: 2026-08-09

## Что было не так

1. `/mo-protocol-viewer.css` → **404** (HTML без стилей МО).
2. На GCE в контейнере **не было** `minzdrav_protocols/` и `data/protocol_summaries/` → brief пустой, PDF 404 → ощущение «просто картинка / всё не так».

## Что сделано

- PR #98: route для CSS (`2026-08-09-090337Z-mo-proto-nav-css`).
- Ops: corpus ~289MB в `/var/data/protocol_corpus`, mount в `protocol-web`.
- Smoke: brief `available=True` (5 секций), PDF 200, CSS 200.
- Deploy script: `deploy/gcp-app/sync_protocol_corpus.sh` + mounts в `deploy_to_gce.sh`.

## Проверка

Hard refresh → разбор → **Открыть протокол**: разделы слева, пункты, стиль МО, PDF по ссылке.
