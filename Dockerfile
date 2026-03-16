# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency manifest first (layer-cache friendly)
COPY pyproject.toml ./

# Install all production dependencies into /app/.venv
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python -e "." --no-cache

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Default: inference API on port 8000
EXPOSE 8000
CMD ["uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
