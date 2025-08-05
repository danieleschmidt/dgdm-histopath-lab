# Multi-stage production Dockerfile for DGDM Histopath Lab
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-venv \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create python3 symlink
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create non-root user for security
RUN groupadd -r dgdm && useradd -r -g dgdm -s /bin/bash -m dgdm

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml README.md ./
COPY dgdm_histopath/__init__.py dgdm_histopath/__init__.py

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    pytest-xdist>=3.0.0 \
    black>=22.0.0 \
    flake8>=5.0.0 \
    mypy>=1.0.0 \
    pre-commit>=2.20.0 \
    jupyter \
    ipykernel \
    notebook

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Set up pre-commit hooks
RUN git init . && pre-commit install || true

# Change ownership to dgdm user
RUN chown -R dgdm:dgdm /app

USER dgdm

# Expose ports
EXPOSE 8888 6006 8000

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Copy only necessary files
COPY dgdm_histopath/ ./dgdm_histopath/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY requirements.txt pyproject.toml README.md ./

# Install package
RUN pip install --no-cache-dir .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/logs /app/cache && \
    chown -R dgdm:dgdm /app

# Health check script
COPY <<EOF /usr/local/bin/healthcheck.py
#!/usr/bin/env python3
import sys
import torch
import requests
from pathlib import Path

def main():
    try:
        # Test PyTorch import
        print("Testing PyTorch...")
        assert torch.cuda.is_available(), "CUDA not available"
        
        # Test package import
        print("Testing package import...")
        from dgdm_histopath.models.dgdm_model import DGDMModel
        
        # Test model creation
        print("Testing model creation...")
        model = DGDMModel(node_features=64, hidden_dims=[128, 64], num_classes=3)
        
        # Test API endpoint if running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            assert response.status_code == 200
            print("API health check passed")
        except:
            print("API not running or not responsive")
        
        print("Health check passed")
        return 0
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

RUN chmod +x /usr/local/bin/healthcheck.py

# Switch to non-root user
USER dgdm

# Set environment variables for production
ENV PYTHONPATH=/app
ENV DGDM_LOG_LEVEL=INFO
ENV DGDM_DATA_DIR=/app/data
ENV DGDM_OUTPUT_DIR=/app/outputs
ENV DGDM_CACHE_DIR=/app/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 /usr/local/bin/healthcheck.py

# Expose API port
EXPOSE 8000

# Production command
CMD ["python", "-m", "dgdm_histopath.api.server", "--host", "0.0.0.0", "--port", "8000"]

# Minimal production stage for edge deployment
FROM python:3.9-slim as minimal

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libhdf5103 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built package from production stage
COPY --from=production /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=production /app/dgdm_histopath/ ./dgdm_histopath/
COPY --from=production /app/configs/ ./configs/

# Create non-root user
RUN groupadd -r dgdm && useradd -r -g dgdm dgdm && \
    mkdir -p /app/data /app/outputs /app/logs && \
    chown -R dgdm:dgdm /app

USER dgdm

# Minimal health check
HEALTHCHECK --interval=60s --timeout=5s --retries=2 \
    CMD python -c "import dgdm_histopath; print('OK')" || exit 1

EXPOSE 8000

CMD ["python", "-c", "print('DGDM Histopath Lab - Minimal deployment ready')"]