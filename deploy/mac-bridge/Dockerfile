# protocol-mis-bridge: MariaDB extract (Mac E1 / GCP E2 / BY E3).
# Image name historically under mac-bridge; same binary on all hosts.
FROM python:3.11-slim

WORKDIR /app

COPY requirements-mis-bridge.txt ./
RUN pip install --no-cache-dir -r requirements-mis-bridge.txt

COPY services/__init__.py services/__init__.py
COPY services/mis_bridge/ services/mis_bridge/
COPY scripts/export_mis_protocol_month.py scripts/export_mis_protocol_month.py
COPY deploy/mac-bridge/extract-contract.md deploy/mac-bridge/extract-contract.md

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    RUN_HOST=mac

ENTRYPOINT ["python", "-m", "services.mis_bridge.extract_day"]
CMD ["--help"]
