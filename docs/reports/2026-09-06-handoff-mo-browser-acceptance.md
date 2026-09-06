# Handoff: постоянные браузерные проверки МО

2026-09-06; akuazuk/protocol; agent1 / pc1.
Branch codex/mo-browser-acceptance-agent1-pc1.
Worktree /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/mo-browser-acceptance (locked).
Base e15ac9cfceac46e9eed51efb65ab3850390a99e1; HEAD — commit с этим handoff, SHA в PR.

Добавлен tests/e2e/mo-smoke.spec.ts, автоматически обнаруживается существующим
Playwright CI. Runtime и CI-конфигурация не меняются. 14 проверок:
11 страниц МО с настоящими HTML/assets/CSP, ECharts числа 70/20/10 и 78/82%,
мобильный family drill с period/finding_family, явный API failure без ложного 0.
API ответы synthetic; реальные записи и внешние модели не используются.

14 passed локально (16.8s), Python 3.11 uvicorn, Chrome, production CSP env из
штатной конфигурации. Локальный временный config менял только путь Python/Chrome,
порт 8223 и reuseExistingServer=false; он не коммитится. CI использует штатный
Chromium из Playwright. git diff --check passed. Полный CI ожидается.

Это первый постоянный MO smoke, не полная role/clinical acceptance. RBAC,
устаревшие ответы при быстрых фильтрах и полный dataset→API→UI parity остаются.
Browser check новых unknown/confidence/keyboard из #212 дополнительно проверен
в совместной диагностической сборке; данный набор совместим с текущим main.

BUILD_VERSION не требуется (только тесты/docs), merge/deploy не выполнялись.
Последний independently verified production a592d588. Держим только новый spec
и этот handoff. После обновления main повторить новый и прежний E2E; не снижать
required checks. Изменения runtime соседних PR не переносить в эту test-ветку.

Следующая безопасная команда:

```bash
gh pr list --repo akuazuk/protocol --state open
```
