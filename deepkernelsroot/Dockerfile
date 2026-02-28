FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#-builder-#
FROM base AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
RUN mkdir -p src/deepkernels && touch src/deepkernels/__init__.py

RUN pip install --upgrade pip && \
    pip install .[dev]

#-test-#
FROM builder AS test

COPY . .
RUN black --check src/
RUN pytest tests/unit/test_cleaner.py
RUN pip install safety && safety check --output bare

#-prod-#
FROM base AS production

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYKEOPS_BUILD_DIR='/app/.cache/pykeops' \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

COPY src/ /app/src/
COPY pyproject.toml /app/

RUN useradd -m deepkerneluser && \
    mkdir -p /app/.cache/pykeops && \
    chown -R deepkerneluser:deepkerneluser /app

USER deepkerneluser

LABEL maintainer="Liam Douglas Giles"
LABEL version="0.1.0"
LABEL description="deepkernels: Gaussian Process Regression Framework in O(N)"

ENTRYPOINT ["python", "-m", "deepkernels"]