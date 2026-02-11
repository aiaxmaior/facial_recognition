# Gateway Integration - Configuration Guide

## Key Information from Admin's Gateway Code

### 1. Gateway Root Path
```python
app = FastAPI(lifespan=lifespan, root_path="/galaxy/slmapi")
```

**This tells us:**
- The gateway uses `/galaxy/slmapi` as the base path
- All services behind this gateway are likely prefixed with this path
- Your GPU server should probably use a similar pattern

### 2. Endpoint Pattern
Admin's service uses:
- `/` - Health check
- `/logs` - Log retrieval API
- `/logs/view` - Web UI for logs
- `/ws/logs` - WebSocket log streaming
- `/v1/chat/completions` - Main API endpoint

**Your GPU server uses:**
- `/` - Health check (detailed)
- `/health` - Simple health check
- `/logs` - Log retrieval API ✅ Same pattern
- `/logs/view` - Web UI for logs ✅ Same pattern
- `/ws/logs` - WebSocket streaming ✅ Same pattern
- `/v1/vectorizer` - Facial embeddings
- `/v1/vlm` - Emotion analysis

---

## Recommended Configuration for GPU Server

### Option 1: Parallel Service (Recommended)

Your GPU server runs alongside the MCP service with its own prefix:

**Gateway Configuration:**
```
/galaxy/slmapi/       → MCP LLM Service (existing)
/galaxy/gpu/          → GPU Server (your service)
```

**Your docker-compose.yaml:**
```yaml
environment:
  - ROOT_PATH=/galaxy/gpu
```

**Access URLs:**
- External: `https://gateway.com/galaxy/gpu/v1/vectorizer`
- External: `https://gateway.com/galaxy/gpu/v1/vlm`
- External: `https://gateway.com/galaxy/gpu/health`

---

### Option 2: Nested Under Existing Service

If the GPU server is considered part of the MCP service:

**Gateway Configuration:**
```
/galaxy/slmapi/v1/chat/completions  → MCP LLM
/galaxy/slmapi/v1/vectorizer        → GPU Server
/galaxy/slmapi/v1/vlm               → GPU Server
```

**Your docker-compose.yaml:**
```yaml
environment:
  - ROOT_PATH=/galaxy/slmapi
```

**Access URLs:**
- External: `https://gateway.com/galaxy/slmapi/v1/vectorizer`
- External: `https://gateway.com/galaxy/slmapi/v1/vlm`

---

### Option 3: Separate Domain/Subdomain

If GPU server gets its own subdomain:

**Gateway Configuration:**
```
https://api.yourdomain.com/          → MCP LLM Service
https://gpu.yourdomain.com/          → GPU Server
```

**Your docker-compose.yaml:**
```yaml
environment:
  - ROOT_PATH=  # Empty, no prefix needed
```

**Access URLs:**
- External: `https://gpu.yourdomain.com/v1/vectorizer`
- External: `https://gpu.yourdomain.com/v1/vlm`

---

## Questions to Ask the Gateway Admin

1. **What ROOT_PATH should the GPU server use?**
   - Same as MCP (`/galaxy/slmapi`)?
   - Different prefix (`/galaxy/gpu`)?
   - No prefix (separate domain)?

2. **How should endpoints be accessed externally?**
   - Share routing with existing MCP service?
   - Get its own prefix?
   - Get its own subdomain?

3. **Does the gateway strip path prefixes?**
   - If YES → Leave `ROOT_PATH` empty
   - If NO → Set `ROOT_PATH` to match gateway prefix

4. **What domain/hostname will be used?**
   - For CORS configuration
   - For SSL certificate planning

---

## Your Current Configuration Status

### ✅ Already Configured (Ready for Gateway)

**In `app/main.py`:**
```python
# Line 185-191
ROOT_PATH = os.environ.get("ROOT_PATH", "")
app = FastAPI(
    title="GPU Processing Server",
    version="1.0.0",
    lifespan=lifespan,
    root_path=ROOT_PATH,  # ← Gateway integration ready!
)
```

**In `docker-compose.yaml`:**
```yaml
# Line 27
- ROOT_PATH=${ROOT_PATH:-}  # ← Configurable via environment
```

### ⚙️ To Configure (After Talking to Admin)

**Create `.env` file:**
```bash
# Based on admin's answer, use ONE of these:

# Option 1: Parallel service
ROOT_PATH=/galaxy/gpu

# Option 2: Nested under MCP
ROOT_PATH=/galaxy/slmapi

# Option 3: Separate domain
ROOT_PATH=
```

**Or set in docker-compose.yaml directly:**
```yaml
environment:
  - ROOT_PATH=/galaxy/gpu  # Replace with actual value
```

---

## Testing Gateway Integration

### 1. Test Locally Without Gateway
```bash
# No ROOT_PATH set
docker compose up

# Endpoints accessible at:
curl http://localhost:5000/health
curl http://localhost:5000/v1/vectorizer
curl http://localhost:5000/v1/vlm
```

### 2. Test Locally WITH Gateway Prefix
```bash
# Set ROOT_PATH to simulate gateway
export ROOT_PATH=/galaxy/gpu
docker compose up

# Endpoints now include prefix:
curl http://localhost:5000/galaxy/gpu/health
curl http://localhost:5000/galaxy/gpu/v1/vectorizer
curl http://localhost:5000/galaxy/gpu/v1/vlm
```

### 3. Test Through Actual Gateway (After Deploy)
```bash
# Through gateway
curl https://gateway.com/galaxy/gpu/health
curl https://gateway.com/galaxy/gpu/v1/vectorizer

# Verify OpenAPI docs work
open https://gateway.com/galaxy/gpu/docs
```

---

## Key Similarities with Admin's Code

Your GPU server already matches the admin's pattern:

| Feature | Admin's Code | Your Code | Status |
|---------|--------------|-----------|--------|
| **root_path support** | ✅ `/galaxy/slmapi` | ✅ Configurable | ✅ Compatible |
| **Logging pattern** | ✅ deque + WebSocket | ✅ Same pattern | ✅ Compatible |
| **Log endpoints** | ✅ `/logs`, `/logs/view` | ✅ Same | ✅ Compatible |
| **Health check** | ✅ `/` returns string | ✅ `/health` returns JSON | ✅ Compatible |
| **Lifespan manager** | ✅ asynccontextmanager | ✅ Same pattern | ✅ Compatible |
| **Error handling** | ⚠️ HTTPException | ✅ Same | ✅ Compatible |
| **CORS** | ❓ Not shown | ✅ Configured | ✅ Ready |

**Your code is already compatible with their gateway architecture!** 🎉

---

## Next Steps

1. **Contact gateway admin** and ask:
   ```
   "I'm deploying a GPU processing service with these endpoints:
   - /v1/facial_recognition (facial embeddings)
   - /v1/vlm (emotion analysis)
   - /v1/transcription (audio transcription - placeholder)
   - /health (health check)

   What ROOT_PATH should I configure? Should I use:
   - /galaxy/gpu (parallel to slmapi)
   - /galaxy/slmapi (nested under existing)
   - Something else?

   My service will run on port 5000."
   ```

2. **Update configuration** based on their answer:
   ```bash
   # In docker-compose.yaml or .env
   ROOT_PATH=/galaxy/gpu  # Or whatever they specify
   ```

3. **Deploy to AWS** and test:
   ```bash
   # Build on AWS (won't freeze like local)
   docker compose build

   # Start service
   docker compose up -d

   # Verify local access works
   curl http://localhost:5000/health
   ```

4. **Coordinate with gateway admin** for routing:
   - Provide them with your AWS instance IP/hostname
   - Provide endpoint list
   - Test through gateway once configured

---

## Summary

**Status: READY FOR GATEWAY INTEGRATION**

- ✅ Code supports `root_path` (already implemented)
- ✅ Endpoint structure matches gateway patterns
- ✅ Logging/monitoring compatible
- ✅ Docker configuration flexible
- ⏳ Just need admin to specify the ROOT_PATH value

**Action Required:**
1. Ask admin for ROOT_PATH value
2. Update one environment variable
3. Deploy to AWS
4. Test through gateway

That's it! Your code is already gateway-ready.
