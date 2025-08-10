"""
Production-ready deployment and health checking system for DGDM Histopath Lab.

Provides comprehensive deployment validation, health monitoring, and
production environment setup for clinical histopathology AI systems.
"""

import os
import sys
import time
import logging
import asyncio
import threading
import psutil
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import subprocess
import socket
import ssl
from contextlib import contextmanager
import warnings

try:
    import torch
    import torch.distributed as dist
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - some features disabled")

from dgdm_histopath.utils.logging import get_logger, setup_logging
from dgdm_histopath.utils.monitoring import AdvancedMetricsCollector
from dgdm_histopath.utils.security import SecurityAuditor
from dgdm_histopath.utils.validation import InputValidator
from dgdm_histopath.utils.exceptions import DeploymentError, HealthCheckError


@dataclass
class HealthStatus:
    """System health status information."""
    service: str
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    details: Dict[str, Any]
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    timeout_seconds: int = 30
    health_check_interval: int = 60
    enable_ssl: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    enable_metrics: bool = True
    metrics_port: int = 8080
    api_rate_limit: int = 1000  # requests per hour
    
    # Database settings
    database_url: Optional[str] = None
    database_pool_size: int = 10
    
    # Cache settings
    redis_url: Optional[str] = None
    cache_ttl: int = 3600
    
    # Security settings
    jwt_secret: Optional[str] = None
    cors_origins: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = []


class ProductionHealthChecker:
    """Comprehensive health checking for production deployments."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.metrics_collector = AdvancedMetricsCollector()
        self.last_check_time = None
        self.health_history = []
        self.checks_registry = {}
        
        # Register standard health checks
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("network_connectivity", self._check_network)
        if PYTORCH_AVAILABLE:
            self.register_check("gpu_status", self._check_gpu_status)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("process_health", self._check_process_health)
        
    def register_check(self, name: str, check_function: Callable) -> None:
        """Register a custom health check."""
        self.checks_registry[name] = check_function
        self.logger.info(f"Registered health check: {name}")
        
    async def run_health_check(self, check_name: str) -> HealthStatus:
        """Run a specific health check."""
        start_time = time.time()
        
        try:
            if check_name not in self.checks_registry:
                raise HealthCheckError(f"Unknown health check: {check_name}")
                
            check_function = self.checks_registry[check_name]
            result = await asyncio.get_event_loop().run_in_executor(
                None, check_function
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return HealthStatus(
                service=check_name,
                status="healthy" if result['healthy'] else "unhealthy",
                timestamp=datetime.now().isoformat(),
                details=result,
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Health check {check_name} failed: {e}")
            
            return HealthStatus(
                service=check_name,
                status="unhealthy",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)},
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    async def run_all_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks."""
        tasks = []
        for check_name in self.checks_registry:
            task = asyncio.create_task(self.run_health_check(check_name))
            tasks.append((check_name, task))
        
        results = {}
        for check_name, task in tasks:
            try:
                result = await task
                results[check_name] = result
            except Exception as e:
                self.logger.error(f"Failed to run health check {check_name}: {e}")
                results[check_name] = HealthStatus(
                    service=check_name,
                    status="unhealthy",
                    timestamp=datetime.now().isoformat(),
                    details={"error": str(e)},
                    error_message=str(e)
                )
        
        self.last_check_time = time.time()
        self.health_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": {k: asdict(v) for k, v in results.items()}
        })
        
        # Keep only last 100 health check results
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
            
        return results
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            # Define healthy thresholds
            healthy = (
                cpu_percent < 90 and
                memory.percent < 90 and
                load_avg[0] < psutil.cpu_count() * 2
            )
            
            return {
                "healthy": healthy,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "load_average": load_avg[0],
                "cpu_count": psutil.cpu_count()
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Require at least 10GB free space and less than 90% usage
            healthy = free_space_gb > 10 and used_percent < 90
            
            return {
                "healthy": healthy,
                "free_space_gb": free_space_gb,
                "used_percent": used_percent,
                "total_space_gb": disk_usage.total / (1024**3)
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            # Test DNS resolution
            socket.gethostbyname('google.com')
            
            # Test HTTP connectivity
            response = requests.get('https://httpbin.org/status/200', timeout=5)
            network_healthy = response.status_code == 200
            
            # Check network IO stats
            net_io = psutil.net_io_counters()
            
            return {
                "healthy": network_healthy,
                "dns_resolution": True,
                "http_connectivity": network_healthy,
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_gpu_status(self) -> Dict[str, Any]:
        """Check GPU status and availability."""
        if not PYTORCH_AVAILABLE:
            return {"healthy": True, "message": "PyTorch not available"}
            
        try:
            if not torch.cuda.is_available():
                return {"healthy": True, "message": "CUDA not available"}
            
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_cached = torch.cuda.memory_reserved(i)
                memory_total = props.total_memory
                
                gpu_info.append({
                    "device_id": i,
                    "name": props.name,
                    "memory_allocated_gb": memory_allocated / (1024**3),
                    "memory_cached_gb": memory_cached / (1024**3),
                    "memory_total_gb": memory_total / (1024**3),
                    "memory_usage_percent": (memory_allocated / memory_total) * 100
                })
            
            # Check if any GPU has critical memory usage
            max_memory_usage = max(gpu['memory_usage_percent'] for gpu in gpu_info) if gpu_info else 0
            healthy = max_memory_usage < 95
            
            return {
                "healthy": healthy,
                "gpu_count": gpu_count,
                "gpus": gpu_info,
                "max_memory_usage_percent": max_memory_usage
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage and detect memory leaks."""
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Check for memory leaks (simplified)
            if hasattr(self, '_last_memory_check'):
                memory_growth = memory_info.rss - self._last_memory_check
                growth_mb = memory_growth / (1024**2)
            else:
                growth_mb = 0
                
            self._last_memory_check = memory_info.rss
            
            # Alert if memory usage is high or growing rapidly
            healthy = (
                memory_percent < 80 and
                growth_mb < 100  # Less than 100MB growth per check
            )
            
            return {
                "healthy": healthy,
                "memory_usage_mb": memory_info.rss / (1024**2),
                "memory_percent": memory_percent,
                "memory_growth_mb": growth_mb,
                "gc_objects": len(gc.get_objects())
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_process_health(self) -> Dict[str, Any]:
        """Check process health and thread status."""
        try:
            process = psutil.Process()
            
            # Get process information
            cpu_percent = process.cpu_percent()
            num_threads = process.num_threads()
            num_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            
            # Check for thread leaks or resource exhaustion
            healthy = (
                num_threads < 100 and  # Reasonable thread limit
                num_fds < 1000 and     # File descriptor limit
                cpu_percent < 95       # Not consuming all CPU
            )
            
            return {
                "healthy": healthy,
                "pid": process.pid,
                "cpu_percent": cpu_percent,
                "num_threads": num_threads,
                "num_fds": num_fds,
                "status": process.status(),
                "create_time": process.create_time()
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def get_overall_health(self, results: Dict[str, HealthStatus]) -> str:
        """Determine overall system health from individual checks."""
        if not results:
            return "unknown"
        
        statuses = [result.status for result in results.values()]
        
        if all(status == "healthy" for status in statuses):
            return "healthy"
        elif any(status == "unhealthy" for status in statuses):
            # Check if critical services are down
            critical_services = ["system_resources", "disk_space", "memory_usage"]
            for service in critical_services:
                if service in results and results[service].status == "unhealthy":
                    return "unhealthy"
            return "degraded"
        else:
            return "degraded"


class ProductionDeploymentManager:
    """Manage production deployment lifecycle and monitoring."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.health_checker = ProductionHealthChecker(config)
        self.security_auditor = SecurityAuditor()
        self.is_running = False
        self.monitoring_thread = None
        
        # Setup logging for production
        setup_logging(
            level=config.log_level,
            log_file="logs/production.log",
            enable_security_audit=True,
            structured_logging=True
        )
    
    async def start_production_services(self) -> Dict[str, Any]:
        """Start all production services with health monitoring."""
        self.logger.info("Starting production deployment...")
        
        try:
            # Validate deployment environment
            await self._validate_deployment_environment()
            
            # Initialize security measures
            self._initialize_security()
            
            # Start health monitoring
            self._start_health_monitoring()
            
            # Validate initial system health
            health_results = await self.health_checker.run_all_checks()
            overall_health = self.health_checker.get_overall_health(health_results)
            
            if overall_health == "unhealthy":
                raise DeploymentError("System health check failed - cannot start production services")
            
            self.is_running = True
            
            deployment_status = {
                "status": "started",
                "timestamp": datetime.now().isoformat(),
                "health": overall_health,
                "environment": self.config.environment,
                "config": asdict(self.config)
            }
            
            self.logger.info(f"Production services started successfully: {deployment_status}")
            return deployment_status
            
        except Exception as e:
            self.logger.error(f"Failed to start production services: {e}")
            raise DeploymentError(f"Production startup failed: {e}")
    
    async def shutdown_production_services(self) -> Dict[str, Any]:
        """Gracefully shutdown production services."""
        self.logger.info("Shutting down production services...")
        
        try:
            self.is_running = False
            
            # Stop monitoring
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
            
            # Cleanup resources
            self._cleanup_resources()
            
            shutdown_status = {
                "status": "shutdown",
                "timestamp": datetime.now().isoformat(),
                "environment": self.config.environment
            }
            
            self.logger.info("Production services shutdown completed")
            return shutdown_status
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise DeploymentError(f"Shutdown failed: {e}")
    
    async def _validate_deployment_environment(self) -> None:
        """Validate the deployment environment is ready."""
        self.logger.info("Validating deployment environment...")
        
        # Check required directories
        required_dirs = ["logs", "data", "models", "temp"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
        
        # Check SSL certificates if enabled
        if self.config.enable_ssl:
            if not self.config.ssl_cert_path or not Path(self.config.ssl_cert_path).exists():
                raise DeploymentError("SSL enabled but certificate not found")
            if not self.config.ssl_key_path or not Path(self.config.ssl_key_path).exists():
                raise DeploymentError("SSL enabled but private key not found")
        
        # Validate environment variables
        required_env_vars = ["DGDM_ENV", "DGDM_LOG_LEVEL"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            self.logger.warning(f"Missing environment variables: {missing_vars}")
        
        # Check resource availability
        health_results = await self.health_checker.run_all_checks()
        overall_health = self.health_checker.get_overall_health(health_results)
        
        if overall_health == "unhealthy":
            critical_failures = [
                name for name, result in health_results.items()
                if result.status == "unhealthy"
            ]
            raise DeploymentError(f"Environment validation failed: {critical_failures}")
        
        self.logger.info("Deployment environment validation completed")
    
    def _initialize_security(self) -> None:
        """Initialize security measures for production."""
        self.logger.info("Initializing security measures...")
        
        # Log security initialization
        self.security_auditor.log_security_event(
            'system_startup',
            {
                'environment': self.config.environment,
                'ssl_enabled': self.config.enable_ssl,
                'debug_mode': self.config.debug_mode
            }
        )
        
        # Security warnings for production
        if self.config.debug_mode:
            self.logger.warning("DEBUG MODE ENABLED IN PRODUCTION - SECURITY RISK")
        
        if not self.config.enable_ssl and self.config.environment == "production":
            self.logger.warning("SSL DISABLED IN PRODUCTION - SECURITY RISK")
        
        self.logger.info("Security initialization completed")
    
    def _start_health_monitoring(self) -> None:
        """Start background health monitoring thread."""
        def monitoring_loop():
            while self.is_running:
                try:
                    # Run health checks
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    health_results = loop.run_until_complete(
                        self.health_checker.run_all_checks()
                    )
                    loop.close()
                    
                    overall_health = self.health_checker.get_overall_health(health_results)
                    
                    if overall_health == "unhealthy":
                        self.logger.error("System health check failed!")
                        # Could trigger alerts here
                    elif overall_health == "degraded":
                        self.logger.warning("System health is degraded")
                    
                    time.sleep(self.config.health_check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(60)  # Back off on errors
        
        self.monitoring_thread = threading.Thread(
            target=monitoring_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self.monitoring_thread.start()
        self.logger.info("Health monitoring started")
    
    def _cleanup_resources(self) -> None:
        """Clean up resources during shutdown."""
        try:
            # Clean up temporary files
            temp_dir = Path("temp")
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                temp_dir.mkdir(exist_ok=True)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Resource cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")


# Singleton instance for global access
_deployment_manager: Optional[ProductionDeploymentManager] = None

def get_deployment_manager(config: Optional[DeploymentConfig] = None) -> ProductionDeploymentManager:
    """Get or create the global deployment manager instance."""
    global _deployment_manager
    
    if _deployment_manager is None:
        if config is None:
            config = DeploymentConfig()
        _deployment_manager = ProductionDeploymentManager(config)
    
    return _deployment_manager


async def production_startup_check() -> Dict[str, Any]:
    """Run comprehensive production startup validation."""
    config = DeploymentConfig()
    manager = ProductionDeploymentManager(config)
    
    return await manager.start_production_services()


if __name__ == "__main__":
    # CLI interface for deployment management
    import argparse
    
    parser = argparse.ArgumentParser(description="DGDM Production Deployment Manager")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--health-check", action="store_true", help="Run health check only")
    parser.add_argument("--start", action="store_true", help="Start production services")
    parser.add_argument("--shutdown", action="store_true", help="Shutdown services")
    
    args = parser.parse_args()
    
    if args.health_check:
        async def run_health_check():
            config = DeploymentConfig()
            manager = ProductionDeploymentManager(config)
            results = await manager.health_checker.run_all_checks()
            print(json.dumps({k: asdict(v) for k, v in results.items()}, indent=2))
        
        asyncio.run(run_health_check())
    elif args.start:
        asyncio.run(production_startup_check())
    else:
        print("Use --help for available options")