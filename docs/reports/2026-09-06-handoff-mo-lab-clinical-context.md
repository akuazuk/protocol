# Handoff: МО — числовой лабораторный контекст, этап 1b

2026-09-06; akuazuk/protocol; agent1 / pc1.
Branch: codex/mo-lab-clinical-context-agent1-pc1.
Worktree: /private/tmp/protocol-task-mo-lab-clinical-context-pc1.
Base SHA: a592d588fdd7eb428161024ad13e4e3948bb3754.
HEAD: commit с этим handoff; опубликованный SHA доступен в PR.

## Изменение

Uncapped numerical projection отдельно от value-free reconcile. Основной lab
entrypoint использует числовой контекст для abnormal evaluator; display cap
не скрывает результат. Клиническая projection ограничена датой визита,
а evaluator повторно проверяет cutoff. Неизвестные единицы/несовместимые
масштабы не принимаются, числа с < > или текстом не превращаются в точные.
Adult seed не применяется к детям и неизвестному возрасту. Нулевое число
разбирается корректно. В payload добавлен abnormal_check: ограниченный охват,
unknown/applicability, точность даты и ошибка вычисления.

## Ограничения

Источники содержат дату, не доказанное available_at время; внутри дня нужна
проверка человеком. Seed-референсы не валидированы против локальной лаборатории;
первичным медицинским заключением не являются. Primary flags не менялись.
Нужна отдельная валидация референсов (включая TSH), клинической значимости,
acknowledgement и отдельных единиц с конверсией. Данный PR не закрывает A04 целиком.
Неизвестные единицы снижают coverage; это предпочтительнее ложного сравнения,
но coverage по каждому показателю предстоит реализовать.

## Проверки

44 passed: test_mo_lab_clinical_context, test_mo_lab_bundle, test_mo_lab_shadow,
test_lab_abnormal_and_formulary, test_mo_lab_dx_evidence. Все данные синтетические.
Проверены полный SQLite→entrypoint, value-free reconcile, display cap,
post-visit/foreign exclusion, adult guard, strict dimensions/numbers.
Ruff и py_compile изменённых модулей, diff --check пройдены.
Полный CI ожидается в PR; глобальные baseline failures не установлены.

BUILD_VERSION: 2026-09-06-085153Z-mo-lab-clinical-context.
Merge/deploy на момент первого commit не выполнены. Data/schema migration нет.
Последняя отдельно проверенная production версия: a592d588 / 2026-09-06-073651Z-deploy-lock-allowlist.
Не запускать production backfill или Gemini с Mac.

## Синхронизация

Не менять параллельно clinical_knowledge/mo_lab_bundle.py, mo_lab_shadow.py,
lab_abnormal_findings.py и tests/test_mo_lab_clinical_context.py,
tests/test_lab_abnormal_and_formulary.py. Эти файлы свободны по PR dashboard.
С #205 пересечение rag_server.py только BUILD_VERSION нашей ветки — soft;
после merge #205 штатный rebase, затем актуальные тесты и версия. Не перезаписывать
новые API-маршруты #205. Другие клинические модули отдельными PR.

Следующая безопасная команда:

```bash
gh pr list --repo akuazuk/protocol --state open
```

## Синхронизация после #205

Main fe0734a8 (calendar API) включён merge-коммитом без переписывания опубликованной
истории. Единственный конфликт BUILD_VERSION разрешён штатным pr_isolation helper;
API маршруты #205 сохранены. Новая версия 2026-09-06-090744Z-mo-lab-clinical-context.
Повторные локальные проверки и CI выполняются на обновлённом HEAD; deploy ещё нет.
