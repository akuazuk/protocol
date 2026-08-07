# Services (границы контуров)

Канон: `docs/plans/2026-08-07-by-home-gcp-llm-split-v1.md`.

| Сервис | Образ | Эпохи | Владеет |
|--|--|--|--|
| [api](api/README.md) | `protocol-gcp-app` / позже `by-home` | E1-E3 | HTTP UI/API |
| [mo_pipeline](mo_pipeline/README.md) | `protocol-gcp-app` | E1-E3 | score, recompute, warehouse |
| [mis_bridge](mis_bridge/README.md) | `protocol-mis-bridge` | E1 Mac → E2 GCP → E3 BY | только extract из MariaDB |
| [llm_worker](llm_worker/README.md) | `protocol-gcp-llm` | E1-E3 | платный LLM grade/judge |

Не класть сюда PDF corpus, ML dumps, PHI. Новые entrypoint - только thin wrappers над `scripts/` / `clinical_knowledge/`.
