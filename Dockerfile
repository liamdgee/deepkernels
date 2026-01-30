# --- Stage 1: Base Build ---
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off

WORKDIR /app

# Install system dependencies (useful for specialized ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install .

# --- Stage 2: EXAGGERATED TESTING & VALIDATION ---
# This stage will fail the build if your ML logic doesn't hold up
FROM base AS test
COPY . .

# 1. Static Analysis (Linter/Formatter)
RUN black --check src/ && flake8 src/

# 2. Functional Unit Tests
# Verifying cleaner.py and harmoniser.py logic
RUN pytest tests/unit/test_cleaner.py

# 3. ML Logic Validation (The "Exaggerated" Part)
# Here you run scripts that check your specific model constraints:
# - Does lasso.py actually produce sparse coefficients?
# - Does hierarchical_dirichlet_clustering.py handle empty clusters?
# - Does mlops_preprocessing.py handle missing socio_corr data?
RUN python -m pytest tests/ml_logic/test_dirichlet_convergence.py
RUN python -m pytest tests/ml_logic/test_lasso_sparsity.py
RUN python -m pytest tests/ml_logic/test_data_harmonisation_integrity.py

# 4. Security Scan
# Check for vulnerabilities in your dependencies
RUN pip install safety && safety check

# --- Stage 3: Production Image ---
# We start fresh to keep the image lean, copying only the essentials
FROM base AS production

# Copy only the source code and production artifacts
COPY ./src ./src
COPY ./config ./config

# Set up a non-root user for security (Standard for MLOps)
RUN useradd -m mluser
USER mluser

# Metadata
LABEL maintainer="Your Name"
LABEL version="0.1.0"
LABEL description="ML Inference Service for Demographic Inference"

# Expose port for inference.py if you are running an API
EXPOSE 8080

# The command to start your inference service
ENTRYPOINT ["python", "src/pipeline/inference.py"]
