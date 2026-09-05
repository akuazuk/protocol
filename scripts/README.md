## scripts/

Script layout by domain. Prefer canonical entrypoints; flat `scripts/*.py` remain
compatible during migration.

| Domain | Path | Meaning |
|--------|------|---------|
| Ops / git / Render | `scripts/ops/` | safe-start, deploy guards, hygiene, PR overlap / BUILD_VERSION rebase |
| Deploy data | `scripts/deploy/` | Render disk / mis_protocol upload helpers |
| Data / QA waves | `scripts/data/` | corpus QA batch shells + legacy root `*.py` wrappers |
| Dev helpers | `scripts/dev/` | launchd / telegram / diagnostics |
| Methodist / МО | `scripts/mo/` | daily ETL, warehouse, publish |
| MIS | `scripts/mis/` | MariaDB export + L1 batch |
| Corpus | `scripts/corpus/` | chunks / catalogs / vector index |

See plan: `docs/plans/2026-08-04-repo-sections-archive-v2.md` (фаза 2b).
