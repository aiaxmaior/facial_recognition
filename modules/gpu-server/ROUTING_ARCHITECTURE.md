# Routing Architecture - How API Calls Flow Through Gateway

## Current Setup (Before GPU Server)

```
┌─────────────────┐
│ Edge Devices    │
│ Enrollment Modal│
└────────┬────────┘
         │
         ▼
   ┌─────────────┐
   │   Gateway   │──────► /galaxy/slmapi/v1/chat/completions
   │  (Existing) │              ▼
   └─────────────┘        ┌──────────────┐
                          │  MCP Service │
                          │  (LLM Only)  │
                          └──────────────┘
```

---

## New Setup (With GPU Server)

```
┌──────────────────┐     ┌──────────────────┐
│ Enrollment Modal │     │  Edge Devices    │
│  (Web App)       │     │  (Jetson)        │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         │ POST /v1/facial_recognition    │ POST /v1/vlm
         │                        │
         ▼                        ▼
    ┌────────────────────────────────┐
    │         Gateway Server         │
    │  (nginx/Kong/AWS API Gateway)  │
    └──────┬──────────────────┬──────┘
           │                  │
           │                  │
    ┌──────▼─────┐    ┌──────▼─────────┐
    │ MCP Service│    │  GPU Server    │
    │   (LLM)    │    │ (Your Service) │
    │            │    │                │
    │ /v1/chat/  │    │ /v1/facial_rec │
    │ completions│    │ /v1/vlm        │
    │            │    │ /v1/transcript │
    └────────────┘    └────────────────┘
```

---

## Routing Configuration (What Gateway Admin Needs to Add)

### Option A: Separate Prefix (Recommended)

**Gateway routes:**
```
/galaxy/slmapi/*           → MCP Service (existing)
/galaxy/gpu/*              → GPU Server (new)
```

**Examples:**
```
POST /galaxy/slmapi/v1/chat/completions    → MCP Service
POST /galaxy/gpu/v1/facial_recognition     → GPU Server
POST /galaxy/gpu/v1/vlm                    → GPU Server
POST /galaxy/gpu/v1/transcription          → GPU Server
```

**Your configuration:**
```yaml
# In docker-compose.yaml
environment:
  - ROOT_PATH=/galaxy/gpu
```

---

### Option B: Shared Prefix (Nested)

**Gateway routes:**
```
/galaxy/slmapi/v1/chat/completions         → MCP Service
/galaxy/slmapi/v1/facial_recognition       → GPU Server
/galaxy/slmapi/v1/vlm                      → GPU Server
/galaxy/slmapi/v1/transcription            → GPU Server
```

**Your configuration:**
```yaml
# In docker-compose.yaml
environment:
  - ROOT_PATH=/galaxy/slmapi
```

---

## Gateway Configuration Examples

### For nginx Gateway

```nginx
# MCP Service (existing)
location /galaxy/slmapi/v1/chat {
    proxy_pass http://mcp-service:8000/v1/chat;
}

# GPU Server (new - Option A)
location /galaxy/gpu/ {
    proxy_pass http://gpu-server:5000/;
}

# OR GPU Server (new - Option B - nested)
location /galaxy/slmapi/v1/facial_recognition {
    proxy_pass http://gpu-server:5000/v1/facial_recognition;
}

location /galaxy/slmapi/v1/vlm {
    proxy_pass http://gpu-server:5000/v1/vlm;
}

location /galaxy/slmapi/v1/transcription {
    proxy_pass http://gpu-server:5000/v1/transcription;
}
```

---

### For Kong Gateway

```bash
# Add GPU server as a service
curl -i -X POST http://localhost:8001/services/ \
  --data name=gpu-server \
  --data url='http://gpu-server-host:5000'

# Add routes for each endpoint
curl -i -X POST http://localhost:8001/services/gpu-server/routes \
  --data 'paths[]=/galaxy/gpu/v1/facial_recognition' \
  --data 'strip_path=true'

curl -i -X POST http://localhost:8001/services/gpu-server/routes \
  --data 'paths[]=/galaxy/gpu/v1/vlm' \
  --data 'strip_path=true'

curl -i -X POST http://localhost:8001/services/gpu-server/routes \
  --data 'paths[]=/galaxy/gpu/v1/transcription' \
  --data 'strip_path=true'
```

---

### For AWS API Gateway

In AWS Console:
1. Create new Resource: `/galaxy/gpu` or `/gpu`
2. Add child resources: `/v1/facial_recognition`, `/v1/vlm`, `/v1/transcription`
3. Configure integration:
   - Type: HTTP Proxy
   - Endpoint: `http://<gpu-server-ip>:5000/v1/facial_recognition`
   - Method: POST

---

## How Devices Make API Calls

### From Enrollment Modal (Facial Recognition)

**Before Gateway Configuration:**
```javascript
// Enrollment modal currently calls:
POST http://bridge-server/api/gpu/vectorizer
{
  "employee_id": "EMP123",
  "images": [...]
}
```

**After Gateway Configuration (update bridge-server):**
```javascript
// Bridge server forwards to:
POST https://gateway.com/galaxy/gpu/v1/facial_recognition
{
  "employee_id": "EMP123",
  "images": [...]
}
```

---

### From Edge Devices (Emotion Analysis)

**After Gateway Configuration:**
```python
# Edge device (Jetson) code
import requests

response = requests.post(
    "https://gateway.com/galaxy/gpu/v1/vlm",
    json={
        "event_id": "EV2-123",
        "employee_id": "EMP456",
        "images": [{"frame": 0, "data": "base64..."}]
    }
)
```

---

## Complete Request Flow Example

### Scenario: Enrollment Modal Needs Facial Embedding

```
1. User uploads photos in Enrollment Modal
   ▼
2. Modal sends to Bridge Server:
   POST /api/enrollment/process
   ▼
3. Bridge Server forwards to Gateway:
   POST https://gateway.com/galaxy/gpu/v1/facial_recognition
   {
     "employee_id": "EMP123",
     "images": [
       {"pose": "front", "data": "base64..."},
       {"pose": "left", "data": "base64..."}
     ]
   }
   ▼
4. Gateway routes to GPU Server:
   POST http://gpu-server:5000/v1/facial_recognition
   (Gateway strips /galaxy/gpu prefix if configured)
   ▼
5. GPU Server processes with DeepFace/ArcFace
   ▼
6. Returns embedding:
   {
     "employee_id": "EMP123",
     "enrollmentProcessedFile": "base64_embedding...",
     "embedding_dim": 512,
     "model": "ArcFace"
   }
   ▼
7. Gateway forwards response back
   ▼
8. Bridge Server stores in database
   ▼
9. Enrollment Modal shows success
```

---

## DNS/Network Requirements

### For AWS Deployment:

**Gateway needs to reach GPU Server:**
- Same VPC: Use private IP
- Different VPC: Use VPC peering or public IP with security groups
- DNS: Create internal DNS entry (e.g., `gpu-server.internal`)

**Security Groups:**
```
GPU Server Security Group:
- Inbound: Port 5000 from Gateway security group
- Outbound: All (for model downloads)

Gateway Security Group:
- Inbound: Port 443 from Internet (HTTPS)
- Outbound: Port 5000 to GPU Server
```

---

## What Gateway Admin Needs From You

Provide them with:

```yaml
Service Name: GPU Processing Server
Host/IP: <your-aws-instance-ip-or-hostname>
Port: 5000
Protocol: HTTP (gateway handles SSL)

Endpoints to Route:
  1. POST /v1/facial_recognition
     - Purpose: Generate facial embeddings
     - Expected traffic: Low (enrollment only)
     - Timeout: 30s

  2. POST /v1/vlm
     - Purpose: Emotion analysis
     - Expected traffic: Medium (per event)
     - Timeout: 60s

  3. POST /v1/transcription
     - Purpose: Audio transcription (placeholder)
     - Expected traffic: TBD
     - Timeout: TBD

Health Check:
  - Path: /health
  - Method: GET
  - Expected Response: {"status": "ok"}
  - Timeout: 10s

Logs (optional):
  - Path: /logs/view
  - Method: GET
  - Purpose: Web UI for monitoring
```

---

## Testing After Gateway Configuration

### 1. Test Direct to GPU Server (Before Gateway)
```bash
# From gateway server or same VPC
curl http://gpu-server-ip:5000/health
# Should return: {"status": "ok"}
```

### 2. Test Through Gateway (After Configuration)
```bash
# From anywhere
curl https://gateway.com/galaxy/gpu/health
# Should return: {"status": "ok"}
```

### 3. Test Full Request Flow
```bash
# Test facial recognition
curl -X POST https://gateway.com/galaxy/gpu/v1/facial_recognition \
  -H "Content-Type: application/json" \
  -d '{
    "employee_id": "TEST123",
    "images": [
      {"pose": "front", "data": "base64_test_image"}
    ]
  }'
```

---

## Summary

**Gateway Admin Needs To:**
1. Choose routing pattern (Option A or B)
2. Add routes for 3 endpoints
3. Configure health checks
4. Test connectivity

**You Need To:**
1. Deploy GPU server to AWS
2. Provide IP/hostname to gateway admin
3. Set ROOT_PATH based on their choice
4. Update bridge-server/edge-device code with new URLs

**Edge Devices/Enrollment Modal Need:**
1. Update API URLs to point to gateway
2. No other code changes required
