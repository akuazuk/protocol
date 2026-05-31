# Protocol Summary Workflow

1. **Build drafts** (478 PDF → 1 YAML each):
   ```bash
   python -m scripts.build_protocol_summaries
   ```
2. **Validate**:
   ```bash
   python -m scripts.validate_protocol_summaries
   ```
3. **Export rules / RAG**:
   ```bash
   python -m scripts.export_protocol_summary_rules
   python -m scripts.export_protocol_summary_rag
   ```
4. **Analyze KZ**:
   ```bash
   PROTOCOL_SUMMARY_ENABLED=1 PROTOCOL_SUMMARY_MODE=hybrid \
     python -m scripts.analyze_consultation --file kz.txt --mode hybrid
   ```

Paths: `data/protocol_summaries/{drafts,yaml,json,validation_reports}/`

Default mode remains **legacy** (`PROTOCOL_SUMMARY_ENABLED=0`).
