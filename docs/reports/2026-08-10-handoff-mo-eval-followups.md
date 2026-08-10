# Handoff: MO eval quality followups (A–D) + DDInter mount

Дата: 2026-08-10  
Production SHA: `0b38d15d` (`origin/main`)  
`BUILD_VERSION`: `2026-08-10-175213Z-drug-safety-sync`  
URL: https://protocol.kravira.by

## Merged PRs
- #130 feat followups A–D (drug-norm, deep-rescore, priority donut)
- #131 mount DDInter + guard deep-rescore without pairs
- #132 sync drug_safety seeds in GCE tarball
- #129 closed as superseded by #130

## Visit 3600047 after deep-rescore (with DDInter)
- overall **60**, status **review**
- P1 Major DDI: эсциталопрам + суматриптан (surface+INN)
- P2 Moderate DDI: эсциталопрам + **мелоксикам** (не diclofenac)
- P3 missing exams/follow-up; P2 reg55
- `has_diclofenac=False`

## Deploy note
`ddinter_pairs.json` живёт на `/var/data/drug_safety` (gitignored) и монтируется в контейнер.

## Next
- UI smoke: Обзор → кольцо приоритетов → фильтр
- Фаза E (клинические сигналы) - бэклог
