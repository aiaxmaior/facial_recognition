# GPU Server - AWS Deployment Checklist

## ✅ COMPLETED (Ready for AWS)

### 1. Docker Configuration
- ✅ **Dockerfile**: Python 3.10, multi-stage build, CUDA 12.8
- ✅ **.dockerignore**: Prevents secrets/bloat in image
- ✅ **docker-compose.yaml**: GPU support, health checks, DNS configured
- ✅ **Flash-attention**: Disabled by default (optional, can enable later)
- ✅ **Build scripts**: `build-limited.sh` with resource limits

### 2. Application Code
- ✅ **Gateway integration**: ROOT_PATH support for API gateway routing
- ✅ **API endpoints**:
  - `/v1/vectorizer` - Facial embeddings
  - `/v1/vlm` - Emotion analysis
  - `/health` - Health check
  - Legacy endpoints still work
- ✅ **Configuration**: `config.yaml` for model settings
- ✅ **Logging**: WebSocket log viewer, in-memory storage
- ✅ **Health checks**: Multiple levels (FastAPI, vLLM)

### 3. Processors
- ✅ **EmbeddingProcessor**: DeepFace/ArcFace for facial recognition
- ✅ **EmotionProcessor**: VLM support (vllm backend)
- ✅ **Fallback modes**: Mock processors for testing

### 4. Security (Basic)
- ✅ **CORS**: Configured (needs production domains)
- ✅ **Error handling**: Generic 500 errors
- ✅ **.dockerignore**: Prevents AWS keys from being copied

---

## ⚠️ NEEDS CONFIGURATION FOR AWS

### 1. Build the Docker Image
**Status**: Local builds freeze due to resource constraints

**Options**:
```bash
# Option A: Build on AWS directly (RECOMMENDED)
# SSH to AWS instance, then:
cd /path/to/gpu-server
docker compose build

# Option B: Build on local machine when system is idle
# Use overnight build:
nohup ./build-limited.sh > build.log 2>&1 &

# Option C: Use pre-built base image (fastest)
# Modify Dockerfile to use: FROM vllm/vllm-openai:latest
```

**Action Required**: Choose build strategy and execute

---

### 2. Gateway Integration
**Status**: Code ready, needs environment variable

**Current**: `ROOT_PATH=${ROOT_PATH:-}` (empty/flexible)

**Action Required**:
1. Ask gateway admin for path prefix
2. Set environment variable:
   ```bash
   # In docker-compose.yaml or .env file:
   ROOT_PATH=/api/v1/gpu  # Or whatever gateway specifies
   ```

**Example gateway routing**:
- External: `https://gateway.com/api/v1/gpu/v1/vectorizer`
- Internal: `http://gpu-server:5000/v1/vectorizer`

---

### 2b. Edge device access (reachable by IP, no SSH)

**Goal:** Enrollment modals, API servers, and other edge devices must call the vectorizer at `http://<AWS_INSTANCE_IP>:<PORT>/v1/vectorizer` (or `/vectorizer/generate`) without using SSH tunnels or shared keys.

**What to do:**

1. **Expose the container on all interfaces**  
   Docker’s default `ports: "5001:5000"` binds to `0.0.0.0:5001` on the host, so the instance IP (e.g. `172.31.16.30`) can receive connections. If you override with `127.0.0.1:5001:5000`, only localhost is reachable—avoid that for edge access.

2. **AWS Security Group**  
   Allow **inbound TCP** on the host port (e.g. **5001**) from:
   - The VPC CIDR (e.g. `172.31.0.0/16`) if edge devices are in the same VPC, or  
   - Specific IPs/CIDRs of enrollment servers and edge devices, or  
   - A load balancer security group if you put the GPU server behind an ALB/NLB.

3. **Host firewall (if enabled)**  
   If `ufw` or `iptables` is active on the instance, allow the same port:
   ```bash
   sudo ufw allow 5001/tcp && sudo ufw status
   ```

4. **Edge device configuration**  
   Point the enrollment API server (or other clients) at the instance URL, e.g.:
   - `http://172.31.16.30:5001/v1/vectorizer` (private IP, same VPC)
   - Or the instance’s **public IP** and port 5001 if security group allows the client’s source.

**Check from another machine in the same network:**  
`curl -X POST http://172.31.16.30:5001/health` (then try `/v1/vectorizer` with a small payload). If this fails, fix Security Group and/or host firewall before debugging the app.

---

### 3. Model Storage
**Current**: Local path `/data/gpu-server-facial-recognition/models`

**For AWS**: Use EFS or S3

**Option A: EFS (Recommended)**
```yaml
# In docker-compose.yaml
volumes:
  - efs-models:/models:ro

volumes:
  efs-models:
    driver: local
    driver_opts:
      type: nfs4
      o: addr=fs-xxxxx.efs.us-east-1.amazonaws.com,rw
      device: ":/models"
```

**Option B: S3 (Download on startup)**
```bash
# In entrypoint.sh (before starting services)
aws s3 sync s3://my-bucket/models/ /models/
```

**Action Required**:
1. Create EFS or S3 bucket
2. Upload Qwen3-VL-8B-Thinking model
3. Update volume mount

---

### 4. CORS Configuration
**Current**: `allow_origins=["*"]` (development only)

**For AWS**: Restrict to actual domains

**Action Required**: Update `app/main.py`:
```python
allow_origins=[
    "https://yourdomain.com",
    "https://enrollment.yourdomain.com",
    os.environ.get("ALLOWED_ORIGIN", "*")
]
```

---

### 5. Authentication/Authorization
**Current**: None (APIs are open)

**For AWS**: Add security layer

**Options**:
- **Option A**: VPC isolation (private subnet, security groups)
- **Option B**: API keys in headers
- **Option C**: AWS IAM authentication
- **Option D**: Gateway handles auth (pass through)

**Action Required**: Decide on auth strategy and implement

---

### 6. Logging & Monitoring
**Current**: In-memory logs (cleared on restart)

**For AWS**: CloudWatch integration

**Action Required**: Add to `app/main.py`:
```python
import watchtower
logger.add(
    watchtower.CloudWatchLogHandler(),
    format="{message}",
    level="INFO"
)
```

Also enable JSON logging:
```bash
# Environment variable
LOG_JSON=true
```

---

### 7. Resource Configuration
**Current**:
- `VLLM_GPU_MEMORY_FRACTION=0.6` (60% GPU memory)
- No RAM/CPU limits

**For AWS**: Set based on instance type

**Action Required**: Update docker-compose.yaml:
```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 32G
    reservations:
      cpus: '4'
      memory: 16G
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

### 8. Environment Variables Summary
**Current `.env` or `docker-compose.yaml` needs**:

```bash
# Gateway
ROOT_PATH=                          # Set based on gateway config

# Model paths
VLLM_MODEL_PATH=/models/Qwen3-VL-8B-Thinking

# GPU settings
VLLM_GPU_MEMORY_FRACTION=0.6       # Adjust based on workload
NVIDIA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_JSON=true                       # For CloudWatch
LOG_LEVEL=INFO

# CORS (if not hardcoded)
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Optional: API keys (if implementing auth)
API_KEY=your-secret-key
```

---

## 📋 AWS Infrastructure Needs

### 1. EC2 GPU Instance
- **Type**: g5.xlarge or better (NVIDIA A10G)
- **AMI**: Deep Learning AMI (Ubuntu)
- **Storage**: 100GB+ for Docker images
- **Security Group**:
  - Inbound: Port 5000 (from gateway/VPC only)
  - Outbound: All (for model downloads)

### 2. EFS (Optional but Recommended)
- **Purpose**: Store models (shared across instances)
- **Size**: 20GB+ for Qwen3-VL model
- **Mount**: `/models`

### 3. Application Load Balancer (Optional)
- **Purpose**: Health checks, SSL termination, routing
- **Target**: GPU server instance on port 5000
- **Health check**: `/health` endpoint

### 4. CloudWatch
- **Log Group**: `/aws/gpu-server`
- **Metrics**: GPU utilization, request latency

### 5. IAM Role
- **Purpose**: Access to S3/EFS/CloudWatch
- **Policies**:
  - S3 read (if using S3 for models)
  - CloudWatch write (for logs)
  - EFS mount (if using EFS)

---

## 🚀 Deployment Steps

### Step 1: Prepare AWS Resources
1. Launch GPU instance (g5.xlarge)
2. Create EFS filesystem (optional)
3. Upload models to EFS or S3
4. Configure security groups

### Step 2: Deploy Code
```bash
# Option A: Git clone on instance
ssh ec2-user@<instance-ip>
git clone <your-repo>
cd facial_recognition/modules/gpu-server

# Option B: SCP from local
scp -r modules/gpu-server ec2-user@<instance-ip>:~/
```

### Step 3: Configure Environment
```bash
# Create .env file with AWS-specific settings
cat > .env <<EOF
ROOT_PATH=/api/v1/gpu
VLLM_MODEL_PATH=/models/Qwen3-VL-8B-Thinking
LOG_JSON=true
ALLOWED_ORIGINS=https://yourproduction.com
EOF
```

### Step 4: Build Docker Image
```bash
# On AWS instance (won't freeze!)
docker compose build

# Or use nohup for background build
nohup docker compose build > build.log 2>&1 &
tail -f build.log
```

### Step 5: Start Service
```bash
docker compose up -d
docker logs -f gpu-server
```

### Step 6: Verify
```bash
# Health check
curl http://localhost:5000/health

# Detailed status
curl http://localhost:5000/

# Test vectorizer (with test image)
curl -X POST http://localhost:5000/v1/vectorizer \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### Step 7: Configure Gateway
- Provide gateway admin with:
  - Instance IP/hostname
  - Port: 5000
  - Endpoints: `/v1/vectorizer`, `/v1/vlm`, `/health`
  - Expected ROOT_PATH

---

## 🔧 Quick Commands Reference

```bash
# Build
docker compose build

# Start
docker compose up -d

# Logs
docker logs -f gpu-server

# Stop
docker compose down

# Restart
docker compose restart

# Shell access
docker exec -it gpu-server bash

# GPU check
docker exec -it gpu-server nvidia-smi

# Health check
curl http://localhost:5000/health

# View logs in browser
open http://localhost:5000/logs/view
```

---

## 📊 Monitoring Checklist

Once deployed, monitor:
- [ ] GPU utilization (should be < 80%)
- [ ] Memory usage (RAM and VRAM)
- [ ] Request latency
- [ ] Error rates
- [ ] Health check status
- [ ] Disk space (Docker images can grow)

---

## 🔒 Security Checklist

- [ ] VPC isolation or security groups configured
- [ ] CORS restricted to production domains
- [ ] Authentication implemented (if required)
- [ ] AWS_RSA_KEY removed from project directory
- [ ] No secrets in docker-compose.yaml (use .env)
- [ ] SSL/TLS via load balancer
- [ ] Regular security updates (rebuild image monthly)

---

## 📝 Notes

1. **Local builds freeze** - Build on AWS instead
2. **Flash-attention disabled** - Can enable later if needed
3. **vLLM requires ~8GB VRAM** - Plan GPU memory accordingly
4. **DeepFace + vLLM share GPU** - Monitor memory usage
5. **First startup slow** - Model loading takes 2-5 minutes
