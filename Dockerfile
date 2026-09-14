FROM python:3.12-slim

WORKDIR /app

# curl required for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir --prefer-binary -r requirements.txt

RUN addgroup --system vituser && adduser --system --group vituser

# Application source (must be copied before seed_models.py so the import
# of app.services.rating_shim resolves correctly at build time)
COPY app /app/app

# Data files for model seeding
COPY data /app/data

# Seed scripts
COPY scripts /app/scripts

# Seed all 16 VIT ensemble model artifacts at build time.
# PYTHONPATH=/app ensures "from app.services.rating_shim import RatingShim"
# resolves correctly when the seed script runs inside the build context.
# The seed step exits non-zero on failure so a broken image is never pushed.
ENV MODEL_DIR=/app/models
ENV PYTHONPATH=/app
RUN python /app/scripts/seed_models.py

RUN chown -R vituser:vituser /app

USER vituser

ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Shell form so $PORT expands at runtime — Render injects PORT dynamically
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
