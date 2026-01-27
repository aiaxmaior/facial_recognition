# GPU Server Dockerfile
# Facial Embedding & Emotion Analysis Service

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install TensorFlow with CUDA support (for DeepFace)
RUN pip install --no-cache-dir tensorflow[and-cuda]>=2.15.0

# Install additional face detection backends
RUN pip install --no-cache-dir \
    opencv-python-headless>=4.9.0 \
    ultralytics>=8.0.0

# Copy application code
COPY app/ ./app/
COPY config.yaml .

# Create directory for models (mount as volume in production)
RUN mkdir -p /models

# Set config path
ENV CONFIG_PATH=/app/config.yaml

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
