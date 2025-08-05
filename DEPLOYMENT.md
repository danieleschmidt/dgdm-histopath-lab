# DGDM Histopath Lab - Production Deployment Guide

This guide provides comprehensive instructions for deploying the Dynamic Graph Diffusion Models for Histopathology Analysis system in various environments.

## 🚀 Quick Start

### Local Development
```bash
# Clone and setup
git clone <repository-url>
cd dgdm-histopath-lab
./deploy/deploy.sh deploy-local
```

### Docker Deployment
```bash
# Build and deploy with Docker Compose
./deploy/deploy.sh deploy-docker -e production
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes cluster
./deploy/deploy.sh deploy-k8s -e production -v v1.0.0
```

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 8 cores
- RAM: 16GB
- GPU: NVIDIA GPU with 8GB VRAM (for training)
- Storage: 100GB SSD
- OS: Ubuntu 20.04+ / CentOS 8+ / Docker-compatible OS

**Recommended Production:**
- CPU: 16+ cores
- RAM: 64GB+
- GPU: NVIDIA V100/A100 with 32GB+ VRAM
- Storage: 1TB+ NVMe SSD
- Network: 10Gbps+ for large dataset transfers

### Software Dependencies

**Required:**
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+
- NVIDIA Docker (for GPU support)

**For Kubernetes Deployment:**
- kubectl 1.24+
- Helm 3.8+
- Kubernetes 1.24+ cluster with GPU support

**For Development:**
- Git 2.30+
- Pre-commit 2.15+

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Gateway   │    │  File Storage   │
│     (Nginx)     │◄──►│   (FastAPI)     │◄──►│   (MinIO/S3)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  DGDM App API   │    │ Training Worker │    │   Database      │
│  (FastAPI+ML)   │    │   (Celery)      │    │ (PostgreSQL)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │ Message Queue   │    │  Cache Layer    │
│ (Prometheus +   │    │    (Redis)      │    │    (Redis)      │
│   Grafana)      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🐳 Docker Deployment

### Environment Setup

1. **Create environment file:**
```bash
# .env
DGDM_SECRET_KEY=your-super-secret-key-change-in-production
GRAFANA_PASSWORD=your-grafana-password
DATABASE_PASSWORD=your-database-password
ENVIRONMENT=production
VERSION=latest
```

2. **Configure data directories:**
```bash
# Create required directories
mkdir -p {data,outputs,logs,cache,models}

# Set appropriate permissions
chmod 755 data outputs logs cache models
chown -R 1000:1000 data outputs logs cache models
```

### Deploy Services

```bash
# Production deployment
./deploy/deploy.sh deploy-docker -e production

# Development with Jupyter
./deploy/deploy.sh deploy-docker -e development
```

### Service URLs
- **Application API:** http://localhost:8000
- **Grafana Monitoring:** http://localhost:3000 (admin/your-password)
- **File Browser:** http://localhost:8080
- **Prometheus:** http://localhost:9090
- **Kibana Logs:** http://localhost:5601

## ☸️ Kubernetes Deployment

### Cluster Prerequisites

1. **GPU Node Pool:**
```yaml
# Example node pool configuration
nodePool:
  instanceType: Standard_NC24s_v3  # Azure
  # OR: p3.8xlarge (AWS)
  # OR: nvidia-tesla-v100 (GCP)
  accelerator: nvidia-tesla-v100
  minNodes: 1
  maxNodes: 5
  autoScaling: true
```

2. **NVIDIA Device Plugin:**
```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml
```

3. **Storage Classes:**
```bash
# For cloud providers, ensure fast SSD storage class
kubectl apply -f kubernetes/storage-class.yaml
```

### Deploy to Kubernetes

1. **Set environment variables:**
```bash
export DGDM_SECRET_KEY="your-secret-key"
export GRAFANA_PASSWORD="your-grafana-password"
export ENVIRONMENT="production"
export NAMESPACE="dgdm-prod"
```

2. **Deploy:**
```bash
./deploy/deploy.sh deploy-k8s -e production -n dgdm-prod -v v1.0.0
```

3. **Verify deployment:**
```bash
kubectl get pods -n dgdm-prod
kubectl get services -n dgdm-prod
kubectl get ingress -n dgdm-prod
```

### Scaling Configuration

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dgdm-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dgdm-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🏭 Production Configuration

### Security Configuration

1. **SSL/TLS Setup:**
```bash
# Generate certificates (Let's Encrypt recommended)
certbot certonly --webroot -w /var/www/certbot -d your-domain.com

# Or use existing certificates
cp your-cert.pem nginx/ssl/
cp your-key.pem nginx/ssl/
```

2. **Network Security:**
```yaml
# Network Policy Example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: dgdm-network-policy
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: dgdm-histopath
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
```

3. **Authentication:**
```bash
# Configure OAuth2/OIDC integration
export DGDM_AUTH_PROVIDER="oauth2"
export DGDM_OAUTH_CLIENT_ID="your-client-id"
export DGDM_OAUTH_CLIENT_SECRET="your-client-secret"
```

### Performance Optimization

1. **Database Optimization:**
```sql
-- PostgreSQL configuration
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
SELECT pg_reload_conf();
```

2. **Redis Configuration:**
```
# redis.conf
maxmemory 8gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

3. **Application Tuning:**
```yaml
# ConfigMap updates
DGDM_MAX_WORKERS: "8"
DGDM_BATCH_SIZE: "16"
DGDM_CACHE_SIZE_GB: "32"
DGDM_PREFETCH_FACTOR: "8"
```

## 📊 Monitoring and Observability

### Metrics Collection

The system automatically exposes metrics for:
- **Application Performance:** Request latency, throughput, error rates
- **Model Performance:** Training progress, inference time, accuracy
- **System Resources:** CPU, memory, GPU utilization
- **Business Metrics:** User activity, data processing volume

### Alerting Rules

```yaml
# Example Prometheus alerting rules
groups:
- name: dgdm-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      
  - alert: GPUMemoryHigh
    expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "GPU memory usage above 90%"
```

### Log Aggregation

Logs are automatically collected and indexed in Elasticsearch:
- **Application logs:** Structured JSON logs with correlation IDs
- **Access logs:** HTTP request/response logs with performance metrics
- **Audit logs:** Security events and user actions
- **System logs:** Infrastructure and container logs

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The repository includes a comprehensive CI/CD pipeline:

1. **Code Quality Gates:**
   - Code formatting (Black)
   - Linting (Flake8, MyPy)
   - Security scanning (Bandit)
   - Test coverage (>85%)

2. **Automated Testing:**
   - Unit tests
   - Integration tests
   - Performance benchmarks
   - Security tests

3. **Deployment Automation:**
   - Build and push Docker images
   - Deploy to staging environment
   - Run smoke tests
   - Deploy to production (with approval)

### Manual Deployment Steps

For manual deployments:

```bash
# 1. Build and test
./deploy/deploy.sh build -v v1.2.0
./deploy/deploy.sh test

# 2. Push to registry
./deploy/deploy.sh push -v v1.2.0

# 3. Deploy to staging
./deploy/deploy.sh deploy-k8s -e staging -v v1.2.0

# 4. Run integration tests
./deploy/deploy.sh test -e staging

# 5. Deploy to production
./deploy/deploy.sh deploy-k8s -e production -v v1.2.0
```

## 🛠️ Maintenance and Operations

### Backup Strategy

1. **Database Backups:**
```bash
# Automated daily backups
kubectl create cronjob postgres-backup \
  --image=postgres:15 \
  --schedule="0 2 * * *" \
  -- pg_dump -h postgres -U dgdm dgdm_db > /backup/backup-$(date +%Y%m%d).sql
```

2. **Model and Data Backups:**
```bash
# Sync to cloud storage
aws s3 sync /app/models s3://your-bucket/models/
aws s3 sync /app/outputs s3://your-bucket/outputs/
```

### Update Procedures

1. **Rolling Updates:**
```bash
# Update application
kubectl set image deployment/dgdm-app dgdm-app=dgdm-histopath:v1.2.0
kubectl rollout status deployment/dgdm-app

# Rollback if needed
kubectl rollout undo deployment/dgdm-app
```

2. **Database Migrations:**
```bash
# Run migrations
kubectl exec -it deployment/dgdm-app -- python -m alembic upgrade head
```

### Troubleshooting

#### Common Issues

1. **GPU Not Available:**
```bash
# Check GPU drivers
nvidia-smi
kubectl describe node <gpu-node>

# Verify device plugin
kubectl get pods -n kube-system | grep nvidia-device-plugin
```

2. **High Memory Usage:**
```bash
# Check memory usage
kubectl top pods -n dgdm-histopath
kubectl describe pod <pod-name>

# Adjust memory limits
kubectl patch deployment dgdm-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"dgdm-app","resources":{"limits":{"memory":"16Gi"}}}]}}}}'
```

3. **Slow Training:**
```bash
# Check GPU utilization
kubectl exec -it <worker-pod> -- nvidia-smi

# Verify data loading
kubectl logs <worker-pod> | grep "DataLoader"

# Check storage performance
kubectl exec -it <pod> -- dd if=/dev/zero of=/app/data/test bs=1M count=1000
```

#### Health Checks

```bash
# Application health
curl -f http://your-domain/health

# Database connectivity
kubectl exec -it deployment/dgdm-app -- python -c "
from dgdm_histopath.utils.database import check_connection
print('DB OK' if check_connection() else 'DB Error')
"

# GPU availability
kubectl exec -it deployment/dgdm-app -- python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
"
```

## 🔐 Security Best Practices

### Container Security
- Run containers as non-root user
- Use minimal base images
- Scan images for vulnerabilities
- Keep dependencies updated

### Network Security
- Use TLS for all communications
- Implement network policies
- Restrict ingress/egress traffic
- Use service mesh for zero-trust

### Data Security
- Encrypt data at rest and in transit
- Implement proper RBAC
- Audit all access to sensitive data
- Regular security assessments

### Secrets Management
- Use Kubernetes secrets or external secret managers
- Rotate secrets regularly
- Never commit secrets to version control
- Use least privilege access

## 📈 Performance Tuning

### Application Performance
- Enable GPU acceleration
- Optimize batch sizes
- Use mixed precision training
- Implement efficient data loading

### Infrastructure Performance
- Use SSD storage for databases
- Configure appropriate resource limits
- Enable horizontal pod autoscaling
- Use cluster autoscaling

### Network Performance
- Use dedicated network for GPU nodes
- Configure appropriate MTU sizes
- Use local storage for temporary data
- Implement efficient data pipelines

## 🆘 Support and Troubleshooting

### Getting Help
- Check the logs: `kubectl logs -f deployment/dgdm-app`
- Review monitoring dashboards
- Check GitHub Issues
- Contact support team

### Emergency Procedures
1. **Service Outage:**
   - Check health endpoints
   - Review recent deployments
   - Check resource utilization
   - Scale up if needed

2. **Data Loss:**
   - Stop all write operations
   - Restore from latest backup
   - Verify data integrity
   - Resume operations

3. **Security Incident:**
   - Isolate affected services
   - Review audit logs
   - Rotate compromised credentials
   - Document incident

For additional support, please refer to the project documentation or contact the development team.