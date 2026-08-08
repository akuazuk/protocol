# Handoff: МО МКБ шаги 1-4 (merge + GCE)

Дата: 2026-08-08  
Production GCE: https://protocol.kravira.by  
BUILD_VERSION: `2026-08-08-073820Z-dx-fulltext-fallback`  
main SHA: `f983d93c` (#50), ранее #49 `ff17fe87`

## Сделано

1-2. PR #49: name_only/directory, `B_dx_absent`, чип `icd_visit_status` в mo-app  
3. Merge #49 + deploy GCE  
4. PR #50: `resolve_diagnosis_text_from_mo` (слоты → label → near code) + deploy GCE  

## В аналитике сейчас

- Чип у диагноза в таблице / KPI в разборе  
- Shadow findings: нет Dx / не в справочнике / слабо  
- Текст Dx ищется и вне графы диагноза  

## Дальше

5. Калибровка → primary (влияют на балл)  
6. История пациента (отдельный план)

## Deploy

`bash deploy/gcp-app/deploy_to_gce.sh` from origin/main; smoke `/api/version` + `/health/live`.
