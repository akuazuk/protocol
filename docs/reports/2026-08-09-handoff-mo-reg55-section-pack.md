# Handoff: mo-reg55-section-pack (warehouse + filters)

- repo: akuazuk/protocol
- branch: `cursor/mo-reg55-section-pack-agent1-pc1`
- worktree: `/private/tmp/protocol-task-mo-reg55-section-pack-pc1`
- PR: https://github.com/akuazuk/protocol/pull/87
- параллельно open: #88 dashboards plan, #89 auth admin (не пересекаются по hot path №55)

## Сделано в этом коммите
- Warehouse: `fact_mo_case.reg55_*` + daily `avg_reg55_section_pct` / `n_band_*`
- Publish: `evaluate_reg55_section` → `warehouse_reg55_columns` в `upsert_warehouse`
- API filters: `reg55_point`, `reg55_band`, `reg55_pack` на `/cases`
- UI: pill градации в documents/queue, row-tint, chips/URL, drill с Обзора

## Нужен backfill
Колонки появятся при следующем `initialize_warehouse` + recompute дней. До backfill list API деградирует к axis `regulatory` без band.

```bash
# после merge / на GCE или warehouse host
# scripts/recompute_mo_days.py за нужный диапазон
```

## Не сделано
- `fact_mo_reg55_criterion`, donut/heatmap/trend band shares
- toolbar filter-pop «Градация №55»
- полный выпил legacy binary/rubric hot path

## Next
Backfill warehouse → smoke queue filter `reg55_band=noncompliant`.
