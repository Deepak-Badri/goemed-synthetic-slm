# ── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment variables ──────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    TOKENIZERS_PARALLELISM=false \
    MODEL_PATH=/app/model \
    PORT=8000

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
COPY requirements-api.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements-api.txt

# ── Copy application code ──────────────────────────────────────────────────
COPY src/api/main.py .

# ── Health check ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Expose port ────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start command ──────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]