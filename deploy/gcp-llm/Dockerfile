# protocol-gcp-llm: Gemini grade/judge jobs only (epochs E1-E3).
FROM python:3.14-slim

WORKDIR /app

COPY requirements-llm-worker.txt ./
RUN pip install --no-cache-dir -r requirements-llm-worker.txt

COPY services/__init__.py services/__init__.py
COPY services/llm_worker/ services/llm_worker/
COPY scripts/grade_kz_llm.py scripts/grade_kz_llm.py
COPY scripts/run_mo_action_queue_llm_judge.py scripts/run_mo_action_queue_llm_judge.py
COPY deploy/gcp-llm/job-contract.md deploy/gcp-llm/job-contract.md

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "services.llm_worker.grade_day"]
CMD ["--help"]
