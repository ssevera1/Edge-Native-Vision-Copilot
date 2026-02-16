# ============================================================
# Factory Safety Monitor — Edge-Optimised Container
# Target: resource-constrained edge devices (≤2 GB RAM)
#
# Stages:
#   base  → runtime OS deps + Python packages
#   test  → adds pytest, runs tests (CI only — not shipped)
#   prod  → final production image (default)
# ============================================================

# ---- base: shared foundation ----
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal OS packages OpenCV needs at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (shared by test + prod)
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/

# ---- test: CI stage (not shipped to edge) ----
FROM base AS test

COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

COPY tests/ tests/

CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

# ---- prod: final production image (default target) ----
FROM base AS prod

ENTRYPOINT ["python", "-u", "src/inference.py"]
