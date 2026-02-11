#!/bin/bash
set -e

# ============================================
# GPU Server Entrypoint
# Starts vLLM server + FastAPI application
# ============================================

VLLM_MODEL="${VLLM_MODEL_PATH:-/models/Qwen3-VL-8B-Thinking}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEMORY="${VLLM_GPU_MEMORY_FRACTION:-0.6}"
FASTAPI_PORT="${FASTAPI_PORT:-5000}"

echo "=========================================="
echo " GPU Server Starting"
echo "=========================================="
echo " vLLM Model: $VLLM_MODEL"
echo " vLLM Port:  $VLLM_PORT"
echo " FastAPI Port: $FASTAPI_PORT"
echo "=========================================="

# ============================================
# Start vLLM server in background
# ============================================
if [ -d "$VLLM_MODEL" ]; then
    echo "[entrypoint] Starting vLLM server..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --gpu-memory-utilization "$VLLM_GPU_MEMORY" \
        --max-model-len 4096 \
        --trust-remote-code \
        2>&1 | sed 's/^/[vllm] /' &

    VLLM_PID=$!
    echo "[entrypoint] vLLM started with PID $VLLM_PID"

    # Wait for vLLM to be ready
    echo "[entrypoint] Waiting for vLLM to be ready..."
    MAX_WAIT=300  # 5 minutes max
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
            echo "[entrypoint] vLLM is ready (waited ${WAITED}s)"
            break
        fi
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $((WAITED % 30)) -eq 0 ]; then
            echo "[entrypoint] Still waiting for vLLM... (${WAITED}s)"
        fi
    done

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[entrypoint] WARNING: vLLM did not become ready within ${MAX_WAIT}s, starting FastAPI anyway"
    fi
else
    echo "[entrypoint] WARNING: Model not found at $VLLM_MODEL"
    echo "[entrypoint] vLLM will NOT be started. Emotion analysis will use mock mode."
fi

# ============================================
# Start FastAPI application (foreground)
# ============================================
echo "[entrypoint] Starting FastAPI server on port $FASTAPI_PORT..."
exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$FASTAPI_PORT" \
    --workers 1
