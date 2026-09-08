FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create runtime directories
RUN mkdir -p models_store data/raw data/processed logs

# Non-root user for security
RUN useradd -m -u 1000 finai && chown -R finai:finai /app
USER finai

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-level", "info"]
