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
COPY backend/ backend/
COPY frontend/ frontend/
COPY clinical_knowledge/ clinical_knowledge/
COPY config/ config/
COPY services/__init__.py services/__init__.py
COPY services/README.md services/README.md
COPY services/api/ services/api/
COPY services/mo_pipeline/ services/mo_pipeline/
COPY scripts/ scripts/
COPY data/catalog/ data/catalog/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn rag_server:app --host 0.0.0.0 --port ${PORT}"]
