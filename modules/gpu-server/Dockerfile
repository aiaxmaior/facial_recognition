# GPU Server Dockerfile
# Facial Embedding (DeepFace/ArcFace) + Emotion Analysis (vLLM)
# Single container running both FastAPI app and vLLM server

# Use NVIDIA CUDA devel image (needed for flash-attn compilation)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

# Accept build arguments from docker compose / build script
ARG MAX_JOBS=6
ARG MAKEFLAGS="-j6"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=${MAX_JOBS}
ENV MAKEFLAGS=${MAKEFLAGS}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================
# Install PyTorch + CUDA 12.8
# ============================================
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ============================================
# Install vLLM + flash-attention
# ============================================
RUN pip install --no-cache-dir vllm

# Limit flash-attn compilation: 6 CPU cores, no GPU access during build
# This lets you keep using your GPU for other work while Docker builds
ENV MAX_JOBS=6
ENV CUDA_VISIBLE_DEVICES=""

# Verify environment variables are set
RUN echo "========================================" && \
    echo "flash-attn build configuration:" && \
    echo "MAX_JOBS=${MAX_JOBS}" && \
    echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'" && \
    echo "========================================"

# Flash-attention is OPTIONAL - uncomment if needed (adds 20-30 min to build)
# Only required for very long context windows (8k+ tokens)
# For image processing tasks, the performance difference is negligible
# RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Re-enable GPU access for subsequent build steps
ENV CUDA_VISIBLE_DEVICES=0
RUN echo "GPU re-enabled: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ============================================
# Install DeepFace + TensorFlow
# ============================================
RUN pip install --no-cache-dir \
    "tensorflow[and-cuda]>=2.15.0" \
    tf-keras>=2.18.0 \
    deepface>=0.0.79 \
    opencv-python-headless>=4.9.0 \
    ultralytics>=8.0.0

# ============================================
# Install application dependencies
# ============================================
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================
# Runtime stage (smaller final image)
# ============================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy Python packages from builder
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create app directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY config.yaml .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create directories for models (mounted as volumes)
RUN mkdir -p /models

# Environment
ENV CONFIG_PATH=/app/config.yaml
ENV VLLM_MODEL_PATH=/models/Qwen3-VL-8B-Thinking

# Expose ports: 5000 (FastAPI), 8000 (vLLM internal)
EXPOSE 5000

# Health check against FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start both vLLM and FastAPI via entrypoint
ENTRYPOINT ["./entrypoint.sh"]
