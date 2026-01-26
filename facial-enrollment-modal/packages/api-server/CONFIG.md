# API Server Configuration

## Environment Variables

### Server
| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3003` | API server port |
| `NODE_ENV` | `development` | Environment (`development`, `staging`, `production`) |

### Python Backend (GPU Vectorizer)
| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHON_API_URL` | `http://localhost:5000` | GPU vectorizer service URL |

### Redis Transitory Store
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | - | Redis connection URL (e.g., `redis://localhost:6379`) |
| `ENROLLMENT_TTL_DAYS` | `30` | Days before incomplete enrollments expire |

**Redis Persistence:** Ensure Redis is configured with persistence (RDB + AOF) to survive restarts:
```
# redis.conf
appendonly yes
appendfsync everysec
save 900 1
save 300 10
save 60 10000
```

### IoT Broker
| Variable | Default | Description |
|----------|---------|-------------|
| `IOT_BROKER_URL` | - | IoT Broker Data Service URL |
| `IOT_TENANT_ID` | - | Tenant identifier for multi-tenant API |

### Bridge API
| Variable | Default | Description |
|----------|---------|-------------|
| `BRIDGE_API_URL` | - | Dashboard Bridge API URL |

### Graylog Logging

> **Note:** Graylog configuration is a placeholder. Update `GRAYLOG_HOST` and `GRAYLOG_PORT` 
> once your Graylog instance details are confirmed. The GELF UDP protocol is standard and
> should work with any Graylog setup.

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAYLOG_HOST` | - | Graylog server hostname/IP (TBD) |
| `GRAYLOG_PORT` | `12201` | Graylog GELF UDP port |
| `LOG_FACILITY` | `enrollment-api` | Application name in logs |
| `LOG_LEVEL` | `6` (INFO) | Minimum log level (0=EMERG, 7=DEBUG) |
| `LOG_CONSOLE` | `true` | Enable console output |

**Log Levels:**
| Level | Value | Description |
|-------|-------|-------------|
| EMERGENCY | 0 | System unusable |
| ALERT | 1 | Immediate action required |
| CRITICAL | 2 | Critical conditions |
| ERROR | 3 | Error conditions |
| WARNING | 4 | Warning conditions |
| NOTICE | 5 | Normal but significant |
| INFO | 6 | Informational |
| DEBUG | 7 | Debug messages |

## Example Configuration

```bash
# .env
PORT=3003
NODE_ENV=production
PYTHON_API_URL=http://localhost:5000

# Redis
REDIS_URL=redis://localhost:6379
ENROLLMENT_TTL_DAYS=30

# IoT Broker
IOT_BROKER_URL=https://hbss-bridgestg.qraie.ai
IOT_TENANT_ID=hbss

# Bridge
BRIDGE_API_URL=https://hbss-bridgestg.qraie.ai

# Graylog
GRAYLOG_HOST=graylog.internal.qraie.ai
GRAYLOG_PORT=12201
LOG_FACILITY=enrollment-api
LOG_LEVEL=6
```

## Fallback Behavior

- **No Redis:** Falls back to in-memory store (data lost on restart)
- **No IoT Broker:** Simulates publish (for development)
- **No Python Backend:** Uses mock embedding data (for development)
- **No Graylog:** Console output only

## Graylog Custom Fields

The following custom fields are sent with each log entry:

| Field | Description |
|-------|-------------|
| `_environment` | Deployment environment |
| `_employee_id` | Employee ID being processed |
| `_enrollment_status` | Simplified status (unenrolled/captured/published) |
| `_detailed_status` | Internal detailed status |
| `_action` | Action being performed |
| `_duration_ms` | Operation duration in milliseconds |
| `_devices_notified` | Number of IoT devices notified |
| `_error_type` | Type of error if applicable |
| `_request_id` | Unique request identifier |
| `_tenant_id` | Multi-tenant identifier |
