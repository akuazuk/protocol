# protocol-by-home (E3 stub): same app surface as gcp-app for now.
# Cutover will add mis_bridge on BY and drop warehouse leader from GCP.
FROM python:3.14-slim

WORKDIR /app

LABEL org.opencontainers.image.description="BY home stub - epoch E3; build-only until cutover"

COPY requirements.txt requirements-rag.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_server.py ./
COPY backend/ backend/
COPY frontend/ frontend/
COPY clinical_knowledge/ clinical_knowledge/
COPY config/ config/
COPY services/ services/
COPY scripts/ scripts/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    RUN_HOST=by

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn rag_server:app --host 0.0.0.0 --port ${PORT}"]
