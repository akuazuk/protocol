# protocol-gcp-app: web API + mo_pipeline (epochs E1/E2).
# No PHI COPY; runtime mounts $MO_DATA_ROOT.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-rag.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_server.py ./
COPY env_load.py icd_mkb.py retrieval_bm25.py gemini_verify.py consult_review_pipeline.py ./
COPY download_minzdrav_protocols.py ./
COPY backend/ backend/
COPY frontend/ frontend/
COPY clinical_knowledge/ clinical_knowledge/
COPY corpus_pipeline/ corpus_pipeline/
COPY config/ config/
COPY services/__init__.py services/__init__.py
COPY services/README.md services/README.md
COPY services/api/ services/api/
COPY services/mo_pipeline/ services/mo_pipeline/
COPY scripts/ scripts/
COPY data/catalog/ data/catalog/
# Drug safety seeds (DDInter pairs gitignored - mount /var/data/drug_safety at runtime)
COPY data/drug_safety/high_alert.json data/drug_safety/high_alert.json
COPY data/drug_safety/stopp_start_beers.json data/drug_safety/stopp_start_beers.json
# RU МКБ справочник для mo_icd_* / icd_mkb (без xlsx/WHO dump)
COPY data/icd_reference/icd10_ru_mkb10su.json data/icd_reference/icd10_ru_mkb10su.json
COPY data/icd_reference/icd10_ru_mkb10su.meta.json data/icd_reference/icd10_ru_mkb10su.meta.json
COPY data/icd_reference/dx_aliases_ru.json data/icd_reference/dx_aliases_ru.json
COPY data/icd_reference/icd10_who_2016_terminal_codes.json data/icd_reference/icd10_who_2016_terminal_codes.json
# Пост. МЗ №55 / №127 - критерии для evaluate_reg55 и оси regulatory
COPY data/regulations/ data/regulations/
# Карточки КП для case_protocol_suggest / protocol_match (без PDF)
COPY output/registry/protocol_cards.jsonl output/registry/protocol_cards.jsonl

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn rag_server:app --host 0.0.0.0 --port ${PORT}"]
