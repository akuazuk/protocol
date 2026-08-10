# Handoff: MO case-detail latency minimal package

Дата: 2026-08-10  
Ветка: `cursor/mo-case-detail-latency-pc1`  
План: `docs/plans/2026-08-10-mo-case-detail-latency-v1.md`

## Сделано

- Live concordance/ICD на `GET /cases/{id}` только при пустых findings или `?live=1`
- Prior CSV-скан выключен по умолчанию (`?prior=1` / `MO_CASE_DETAIL_PRIOR=1`)
- Zones path больше не гоняет sync suggest + prior
- Protocol suggest: `attach_history=0` по умолчанию; prewarm при старте контейнера
- Тесты: `tests/test_mo_case_detail_latency.py`

## Не сделано

- Warehouse-индекс prior
- Background enrich в UI после первого paint

## Следующая команда

После merge:

```bash
SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
```

Smoke: открыть 3600047; `/api/version` = `*-case-detail-latency`; profile detail &lt;500 ms.
