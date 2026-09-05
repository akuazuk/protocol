# Handoff: GCE sync семейств/аномалий + стык вкладок 2026-09-05

## Repo

- repo: `akuazuk/protocol`
- branch: `cursor/gce-sync-family-data-agent1-pc1`
- worktree: `/private/tmp/protocol-task-gce-sync-family-data-pc1`
- base: `origin/main` `fe86e74c` (#190)

## Сделано соседними вкладками (уже в main)

- #187 лекарства/анализы, #185 lab → Endpoint C, #188 RAG corpus paths, #189 изоляция CI, #190 KPI `doctor_fio`.
- GCE уже на `36bb2b3f` (#189): RAG 478 путей, `gastro_1` L2 HTTP 200, `/methodist/mo` с разделами Лекарства/Анализы, `MO_LAB_IN_PRIMARY=0`.
- Повторный deploy `fe86e74c` упал: Dockerfile копирует `data/mo_finding_families` и `data/mo_anomalies`, а `deploy_to_gce.sh` их не клал в tar. Контейнер не снимали, прод остался на `36bb2b3f`.

## Этот PR

- Tar GCE включает те же data-пути, что `COPY data/` в Dockerfile.
- Тест `test_gce_sync_includes_dockerfile_data_copies` ловит расхождение в будущем.

## Не сделано до merge этого PR

- Повторный GCE deploy `fe86e74c`+этот SHA.
- Merge красного #186 и старых docs-PR.

Не трогать параллельно: `clinical_knowledge/mo_backend.py` (закрытый #190).
