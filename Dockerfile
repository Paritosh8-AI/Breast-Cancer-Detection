# ==============================================================================
# Breast Cancer Sentinel 2.0 - Production Multi-Stage Dockerfile
# ==============================================================================

FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    DASHBOARD_PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY artifacts/ ./artifacts/
COPY assets/ ./assets/
COPY data/ ./data/
COPY cancer_ai/ ./cancer_ai/
COPY app/ ./app/
COPY model/ ./model/

RUN pip install --no-cache-dir -e .

RUN useradd -m -u 1000 sentinel && \
    chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "cancer_ai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
