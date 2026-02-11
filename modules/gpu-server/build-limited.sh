#!/bin/bash
# Limited Resource Docker Build Script
# Prevents system freeze during flash-attn compilation

set -e

echo "=========================================="
echo "GPU Server - Limited Resource Build"
echo "=========================================="
echo "CPU Cores Available: $(nproc)"
echo "Memory Available: $(free -h | grep Mem | awk '{print $7}')"
echo ""
echo "Build Limits:"
echo "  - CPU Cores: 6 (out of $(nproc))"
echo "  - RAM: 16GB"
echo "  - GPU: Hidden during build"
echo "=========================================="
echo ""

# Set environment variables for Docker build
export MAX_JOBS=6
export MAKEFLAGS="-j6"
export CUDA_VISIBLE_DEVICES=""
# Use legacy builder (lighter than BuildKit)
export DOCKER_BUILDKIT=0

# Limit CPU cores via taskset (runs docker with limited CPU affinity)
# This restricts Docker build to CPUs 0-5 (6 cores)
# Adjust the range based on your needs: 0-5 = 6 cores, 0-11 = 12 cores, etc.
echo "Limiting build to 6 CPU cores (0-5)..."

taskset -c 0-5 docker compose build \
  --build-arg MAX_JOBS=6 \
  --build-arg MAKEFLAGS="-j6" \
  --progress=plain

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
