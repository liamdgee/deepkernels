FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel AS production

# --- Environment Variables ---
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYKEOPS_BUILD_DIR='/app/.cache/pykeops' \
    PYKEOPS_CACHE_DIR='/app/.cache/pykeops' \
    KEOPS_BIN_DIR='/app/.cache/pykeops' \
    KEOPS_EXTRA_FLAGS='-O3 -frounding-math -ffp-contract=off' \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PORT=7860
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ninja-build \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ /app/src/ 

RUN pip install --upgrade pip && \
    pip install .
# --- Application Code & Artifacts ---
#--- copies app.py, app_pipe.py, mlflow.db, the .pkl, and the .pth files
COPY . /app/

# --- Security & Permissions ---
#---- ** KeOps WILL crash if it cannot write its compiled C++ files to the cache dir ** ----#
RUN useradd -m deepkerneluser && \
    mkdir -p /app/.cache/pykeops && \
    chown -R deepkerneluser:deepkerneluser /app

USER deepkerneluser

LABEL maintainer="Liam Douglas Giles (Machine Learning Engineer -- Sydney)"
LABEL description="deepkernels -- A generative state space 'dreamer' model of systemic biases across US credit markets"
LABEL version="1.0.1"

# --- Networking ---
EXPOSE 7860

#--- Server Entrypoint ---#
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]