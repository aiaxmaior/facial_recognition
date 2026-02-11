# GPU Server Build Instructions

## Problem: Build Freezes System

The flash-attention compilation can freeze your system by using all CPU/GPU resources.

## Solution 1: Use the Build Script (Recommended)

```bash
cd ~/facial_recognition/modules/gpu-server
./build-limited.sh
```

This script:
- Limits CPU to 6 cores (25% of your 24 cores)
- Limits RAM to 16GB
- Hides GPU during build
- Takes ~30-40 minutes

---

## Solution 2: Manual Build with Environment Variables

```bash
cd ~/facial_recognition/modules/gpu-server

# Set resource limits
export MAX_JOBS=6
export MAKEFLAGS="-j6"
export CUDA_VISIBLE_DEVICES=""
export DOCKER_BUILDKIT=1

# Build with CPU affinity (limits to 6 cores: 0-5)
taskset -c 0-5 docker compose build \
  --build-arg MAX_JOBS=6 \
  --build-arg MAKEFLAGS="-j6"
```

**Note:** `taskset -c 0-5` restricts Docker to use only CPUs 0-5 (6 cores).
Adjust based on your system: `0-11` = 12 cores, `0-17` = 18 cores, etc.

---

## Solution 3: Skip Flash-Attention (Fastest)

If you don't need flash-attention optimization:

1. Edit `Dockerfile` and comment out line 58:
   ```dockerfile
   # RUN pip install --no-cache-dir flash-attn --no-build-isolation
   ```

2. Build normally:
   ```bash
   docker compose build
   ```

vLLM will work fine without flash-attn, just slightly slower on very long contexts.

---

## Monitor During Build

In another terminal:

```bash
# Watch CPU usage
htop

# Watch GPU usage (should be 0% during flash-attn build)
watch -n 1 nvidia-smi

# Watch Docker build progress
docker ps -a | grep gpu-server
```

---

## If Build Still Freezes

**Emergency stop:**
```bash
# Press Ctrl+C in build terminal, then:
docker compose down
docker system prune -f

# Kill any stuck build processes
pkill -9 dockerd
sudo systemctl restart docker
```

**Then try Solution 3** (skip flash-attn) for fastest build.

---

## Expected Build Times

| Method | Time | Risk of Freeze |
|--------|------|----------------|
| Solution 1 (Limited) | 30-40 min | Low |
| Solution 2 (Manual) | 30-40 min | Low |
| Solution 3 (No flash-attn) | 15-20 min | None |
| Unlimited (default) | 15-25 min | **High** |

---

## Verify Build Success

After build completes:

```bash
docker compose up -d
curl http://localhost:5000/health
```

Should return: `{"status":"ok"}`
