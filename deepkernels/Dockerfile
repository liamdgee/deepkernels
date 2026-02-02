#-base (shared env)-#
FROM python:3.10-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:$PATH"
WORKDIR /app

#-builder-#
FROM base as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN mkdir -p src/deepkernels && touch src/deepkernels/__init__.py

RUN pip install --user --upgrade pip && \
    pip install --user .[dev]

#-test-#
FROM builder as test

COPY . .
RUN black --check src/
RUN pytest tests/unit/test_cleaner.py
RUN pip install --user safety && /root/.local/bin/safety check --output bare

#-prod-#
FROM base AS production

COPY --from=builder /root/.local /root/.local
COPY src/ /app/src/
COPY pyproject.toml /app/

RUN useradd -m deepkerneluser && chown -R deepkerneluser /app
USER deepkerneluser

LABEL maintainer="Liam Douglas Giles"
LABEL version="0.1.0"
LABEL description="deepkernels: Gaussian Process Regression Framework in O(N)"

ENTRYPOINT ["python", "-m", "deepkernels.orchestrate.predict"]
