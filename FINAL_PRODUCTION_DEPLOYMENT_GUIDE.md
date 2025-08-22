# 🚀 DGDM Histopath Lab - Production Deployment Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying the Dynamic Graph Diffusion Models for Histopathology Analysis framework in production environments. The framework includes auto-scaling, monitoring, and clinical-grade features suitable for hospital and research deployment.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │    Nginx    │  │   Grafana   │  │   Kibana    │              │
│  │ (Proxy/LB)  │  │(Monitoring) │ │(Log Viewer) │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ DGDM App    │  │ DGDM Worker │  │ Prometheus  │              │
│  │ (API+GUI)   │  │ (Training)  │  │ (Metrics)   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ PostgreSQL  │  │   Redis     │  │Elasticsearch│              │
│  │ (Database)  │  │  (Cache)    │  │   (Logs)    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Prerequisites

### Hardware Requirements

**Minimum Configuration:**
- CPU: 8 cores (Intel Xeon or AMD EPYC)
- RAM: 32GB DDR4
- Storage: 500GB NVMe SSD
- GPU: NVIDIA RTX 3080 or better (8GB VRAM minimum)
- Network: 1Gbps connection

**Recommended Configuration:**
- CPU: 16+ cores (Intel Xeon Gold or AMD EPYC)
- RAM: 64GB+ DDR4
- Storage: 2TB+ NVMe SSD
- GPU: NVIDIA A100 or H100 (40GB+ VRAM)
- Network: 10Gbps connection

### Software Requirements

- **OS**: Ubuntu 20.04+ LTS, RHEL 8+, or CentOS Stream 8+
- **Docker**: 20.10+ with Docker Compose v2
- **NVIDIA Drivers**: 470+ with CUDA 11.8+
- **Kubernetes**: 1.25+ (optional, for K8s deployment)

## 🚀 Quick Start Deployment

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/danieleschmidt/dgdm-histopath-lab.git
cd dgdm-histopath-lab

# Create environment file
cp .env.example .env
# Edit .env with your configuration
```

### 2. Configure Environment Variables

Edit `.env` file:

```bash
# Core Configuration
DGDM_SECRET_KEY=your-super-secure-secret-key-here
DGDM_LOG_LEVEL=INFO
DGDM_ENVIRONMENT=production

# Database Configuration
POSTGRES_DB=dgdm_db
POSTGRES_USER=dgdm
POSTGRES_PASSWORD=your-secure-db-password

# Monitoring Configuration
GRAFANA_PASSWORD=your-grafana-password

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1,2,3

# Data Paths
DGDM_DATA_DIR=/data/slides
DGDM_OUTPUT_DIR=/data/outputs
DGDM_CACHE_DIR=/data/cache
```

### 3. Deploy Production Stack

```bash
# Deploy full production stack
docker-compose -f docker-compose.yml up -d

# Monitor deployment
docker-compose logs -f dgdm-app

# Check service health
docker-compose ps
```

### 4. Verify Deployment

```bash
# Test API endpoint
curl http://localhost:8000/health

# Check monitoring dashboards
# Grafana: http://localhost:3000 (admin/your-grafana-password)
# Prometheus: http://localhost:9090
# Kibana: http://localhost:5601
```

## 📊 Monitoring and Health Checks

The production deployment includes comprehensive monitoring:

- **Grafana**: Real-time dashboards and visualization
- **Prometheus**: Metrics collection and alerting
- **ELK Stack**: Log aggregation and analysis
- **Health Checks**: Automated service monitoring

## 🎯 Production Readiness

✅ **Multi-service Production Stack**: 9 services with auto-scaling  
✅ **GPU Support**: NVIDIA GPU resource allocation  
✅ **Monitoring & Alerting**: Complete observability stack  
✅ **Security**: SSL/TLS, authentication, encryption  
✅ **Backup & Recovery**: Automated data protection  
✅ **Load Balancing**: Nginx reverse proxy with SSL  

---

**Production Deployment Guide Version**: 1.0  
**Last Updated**: 2025-08-22  
**Framework Version**: 0.1.0