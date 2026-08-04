## scripts/corpus

PDF corpus → chunks → catalogs → vector index pipelines.

### Canonical entrypoints (preferred)

- `scripts/corpus/py/build_rich_chunks.py` → `scripts/build_rich_chunks.py`
- `scripts/corpus/py/build_vector_index.py` → `scripts/build_vector_index.py`
- `scripts/corpus/py/build_protocol_summaries.py` → `scripts/build_protocol_summaries.py`
- `scripts/corpus/py/build_protocol_catalog.py` → `scripts/build_protocol_catalog.py`
- `scripts/corpus/py/build_chunk_embeddings.py` → `scripts/build_chunk_embeddings.py`
- `scripts/corpus/py/build_protocol_icd_index.py` → `scripts/build_protocol_icd_index.py`
- `scripts/corpus/py/enrich_rich_chunk_tags.py` → `scripts/enrich_rich_chunk_tags.py`

Legacy flat paths under `scripts/*.py` remain supported. Physical moves are deferred
until call-sites and launchd/CI are migrated (plan phase 2b+).

### Domain inventory (still flat in `scripts/`)

- `scripts/apply_chunk_rule_fixes.py`
- `scripts/audit_chunk_quality.py`
- `scripts/audit_chunk_tags.py`
- `scripts/batch_llm_protocol_summaries.py`
- `scripts/build_catalog_full.py`
- `scripts/build_catalog_llm_enrichment.py`
- `scripts/build_catalog_rules.py`
- `scripts/build_chunk_embeddings.py`
- `scripts/build_chunk_qa_queue.py`
- `scripts/build_chunk_qa_queue_tiered.py`
- `scripts/build_corpus_path_manifest.py`
- `scripts/build_kz_weak_chunk_qa_queue.py`
- `scripts/build_path_lex_shards.py`
- `scripts/build_protocol_catalog.py`
- `scripts/build_protocol_icd_index.py`
- `scripts/build_protocol_summaries.py`
- `scripts/build_rich_chunk_id_sidecar.py`
- `scripts/build_rich_chunks.py`
- `scripts/build_vector_index.py`
- `scripts/build_vector_index_sidecar.py`
- `scripts/catalog_rules_coverage_report.py`
- `scripts/corpus_manifest.py`
- `scripts/enrich_rich_chunk_tags.py`
- `scripts/export_chunk_qa_dataset.py`
- `scripts/export_protocol_summary_rag.py`
