"""
Production Orchestrator for DGDM Histopath Lab.

This module provides comprehensive production deployment orchestration
including container management, service discovery, health monitoring,
and automated failover capabilities.
"""

import os
import sys
import json
import time
import logging
import subprocess
import threading
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import socket
import requests


class ServiceStatus(Enum):
    """Service status enumeration."""
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    name: str
    image: str
    port: int
    health_check_path: str = "/health"
    environment: Dict[str, str] = None
    volumes: List[str] = None
    depends_on: List[str] = None
    replicas: int = 1
    max_memory: str = "2g"
    max_cpu: str = "1.0"


@dataclass
class ServiceInstance:
    """Running service instance information."""
    service_name: str
    instance_id: str
    container_id: Optional[str]
    pid: Optional[int]
    port: int
    status: ServiceStatus
    started_at: datetime
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0
    restart_count: int = 0


class HealthChecker:
    """Service health checking with intelligent failure detection."""
    
    def __init__(self, check_interval: int = 30, failure_threshold: int = 3):
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self.health_history: Dict[str, List[bool]] = {}
        self.running = True
        
        # Start health check thread
        self.health_thread = threading.Thread(target=self._health_worker, daemon=True)
        self.health_thread.start()
    
    def register_service(self, instance: ServiceInstance):
        """Register a service for health checking."""
        if instance.service_name not in self.health_history:
            self.health_history[instance.service_name] = []
    
    def check_service_health(self, instance: ServiceInstance) -> bool:
        """Check health of a specific service instance."""
        try:
            # HTTP health check
            if instance.port:
                url = f"http://localhost:{instance.port}/health"
                response = requests.get(url, timeout=10)
                healthy = response.status_code == 200
            else:
                # Process-based health check
                if instance.pid:
                    import psutil
                    process = psutil.Process(instance.pid)
                    healthy = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                else:
                    healthy = False
            
            # Update health history
            if instance.service_name in self.health_history:
                self.health_history[instance.service_name].append(healthy)
                # Keep only last 20 checks
                if len(self.health_history[instance.service_name]) > 20:
                    self.health_history[instance.service_name].pop(0)
            
            # Update instance health check info
            instance.last_health_check = datetime.now()
            if not healthy:
                instance.health_check_failures += 1
            else:
                instance.health_check_failures = 0
            
            return healthy
            
        except Exception as e:
            logging.warning(f"Health check failed for {instance.service_name}: {e}")
            instance.health_check_failures += 1
            return False
    
    def _health_worker(self):
        """Background worker for health checking."""
        while self.running:
            try:
                # This would be called by the orchestrator for each service
                time.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Health check worker error: {e}")
                time.sleep(60)  # Back off on errors
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all services."""
        summary = {}
        for service_name, history in self.health_history.items():
            if history:
                recent_checks = history[-10:]  # Last 10 checks
                health_rate = (sum(recent_checks) / len(recent_checks)) * 100
                summary[service_name] = {
                    "health_rate": health_rate,
                    "recent_checks": len(recent_checks),
                    "current_status": "healthy" if recent_checks[-1] else "unhealthy"
                }
        return summary


class LoadBalancer:
    """Simple round-robin load balancer for service instances."""
    
    def __init__(self):
        self.service_instances: Dict[str, List[ServiceInstance]] = {}
        self.current_index: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def register_instance(self, instance: ServiceInstance):
        """Register a service instance for load balancing."""
        with self.lock:
            if instance.service_name not in self.service_instances:
                self.service_instances[instance.service_name] = []
                self.current_index[instance.service_name] = 0
            
            self.service_instances[instance.service_name].append(instance)
    
    def remove_instance(self, service_name: str, instance_id: str):
        """Remove a service instance."""
        with self.lock:
            if service_name in self.service_instances:
                self.service_instances[service_name] = [
                    inst for inst in self.service_instances[service_name]
                    if inst.instance_id != instance_id
                ]
    
    def get_next_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Get next healthy instance using round-robin."""
        with self.lock:
            instances = self.service_instances.get(service_name, [])
            healthy_instances = [inst for inst in instances if inst.status == ServiceStatus.HEALTHY]
            
            if not healthy_instances:
                return None
            
            # Round-robin selection
            current_idx = self.current_index.get(service_name, 0)
            instance = healthy_instances[current_idx % len(healthy_instances)]
            self.current_index[service_name] = (current_idx + 1) % len(healthy_instances)
            
            return instance
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            stats = {}
            for service_name, instances in self.service_instances.items():
                healthy_count = sum(1 for inst in instances if inst.status == ServiceStatus.HEALTHY)
                stats[service_name] = {
                    "total_instances": len(instances),
                    "healthy_instances": healthy_count,
                    "load_distribution": "round_robin"
                }
            return stats


class ProductionOrchestrator:
    """Main production orchestrator for DGDM Histopath Lab."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.services: Dict[str, ServiceConfig] = {}
        self.running_instances: Dict[str, List[ServiceInstance]] = {}
        
        self.health_checker = HealthChecker()
        self.load_balancer = LoadBalancer()
        
        self.shutdown_event = threading.Event()
        self.management_thread = None
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        if config_path and config_path.exists():
            self.load_configuration(config_path)
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def setup_logging(self):
        """Setup orchestrator logging."""
        self.logger = logging.getLogger("dgdm_orchestrator")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def load_configuration(self, config_path: Path):
        """Load service configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for service_data in config_data.get("services", []):
                service_config = ServiceConfig(**service_data)
                self.services[service_config.name] = service_config
                
            self.logger.info(f"Loaded configuration for {len(self.services)} services")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
    
    def register_service(self, config: ServiceConfig):
        """Register a service for orchestration."""
        self.services[config.name] = config
        self.running_instances[config.name] = []
        self.logger.info(f"Registered service: {config.name}")
    
    def start_service(self, service_name: str) -> bool:
        """Start a service with all its replicas."""
        if service_name not in self.services:
            self.logger.error(f"Service not registered: {service_name}")
            return False
        
        config = self.services[service_name]
        
        # Check dependencies
        for dep in config.depends_on or []:
            if not self._is_service_healthy(dep):
                self.logger.error(f"Dependency {dep} not healthy for {service_name}")
                return False
        
        # Start replicas
        success_count = 0
        for i in range(config.replicas):
            if self._start_service_instance(config, i):
                success_count += 1
        
        self.logger.info(f"Started {success_count}/{config.replicas} instances of {service_name}")
        return success_count > 0
    
    def _start_service_instance(self, config: ServiceConfig, replica_id: int) -> bool:
        """Start a single service instance."""
        instance_id = f"{config.name}-{replica_id}"
        port = config.port + replica_id  # Offset port for replicas
        
        try:
            # Create service instance
            instance = ServiceInstance(
                service_name=config.name,
                instance_id=instance_id,
                container_id=None,
                pid=None,
                port=port,
                status=ServiceStatus.STARTING,
                started_at=datetime.now()
            )
            
            # Start container or process
            if self._is_containerized():
                container_id = self._start_container(config, instance_id, port)
                instance.container_id = container_id
            else:
                pid = self._start_process(config, port)
                instance.pid = pid
            
            # Register instance
            if config.name not in self.running_instances:
                self.running_instances[config.name] = []
            
            self.running_instances[config.name].append(instance)
            self.health_checker.register_service(instance)
            self.load_balancer.register_instance(instance)
            
            # Wait for service to be ready
            self._wait_for_service_ready(instance)
            
            self.logger.info(f"Started instance {instance_id} on port {port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start instance {instance_id}: {e}")
            return False
    
    def _is_containerized(self) -> bool:
        """Check if running in containerized environment."""
        return os.path.exists("/.dockerenv") or os.environ.get("CONTAINER_MODE") == "true"
    
    def _start_container(self, config: ServiceConfig, instance_id: str, port: int) -> str:
        """Start a Docker container."""
        cmd = [
            "docker", "run", "-d",
            "--name", instance_id,
            "-p", f"{port}:{config.port}",
            "--memory", config.max_memory,
            "--cpus", config.max_cpu
        ]
        
        # Add environment variables
        for key, value in (config.environment or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        
        # Add volumes
        for volume in (config.volumes or []):
            cmd.extend(["-v", volume])
        
        cmd.append(config.image)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Docker run failed: {result.stderr}")
        
        return result.stdout.strip()
    
    def _start_process(self, config: ServiceConfig, port: int) -> int:
        """Start a regular process."""
        # This would typically start the DGDM service process
        env = os.environ.copy()
        env.update(config.environment or {})
        env["PORT"] = str(port)
        
        # Example command - adjust based on actual service
        cmd = ["python", "-m", "dgdm_histopath.cli.serve", "--port", str(port)]
        
        process = subprocess.Popen(cmd, env=env)
        return process.pid
    
    def _wait_for_service_ready(self, instance: ServiceInstance, timeout: int = 60):
        """Wait for service to become ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.health_checker.check_service_health(instance):
                instance.status = ServiceStatus.HEALTHY
                return
            
            time.sleep(2)
        
        instance.status = ServiceStatus.FAILED
        self.logger.warning(f"Service {instance.instance_id} failed to become ready")
    
    def _is_service_healthy(self, service_name: str) -> bool:
        """Check if a service has at least one healthy instance."""
        instances = self.running_instances.get(service_name, [])
        return any(inst.status == ServiceStatus.HEALTHY for inst in instances)
    
    def stop_service(self, service_name: str):
        """Stop all instances of a service."""
        instances = self.running_instances.get(service_name, [])
        
        for instance in instances:
            self._stop_service_instance(instance)
        
        self.running_instances[service_name] = []
        self.logger.info(f"Stopped service: {service_name}")
    
    def _stop_service_instance(self, instance: ServiceInstance):
        """Stop a single service instance."""
        instance.status = ServiceStatus.STOPPING
        
        try:
            if instance.container_id:
                subprocess.run(["docker", "stop", instance.container_id], timeout=30)
                subprocess.run(["docker", "rm", instance.container_id])
            elif instance.pid:
                os.kill(instance.pid, signal.SIGTERM)
                time.sleep(5)  # Give it time to shutdown gracefully
                try:
                    os.kill(instance.pid, 0)  # Check if still running
                    os.kill(instance.pid, signal.SIGKILL)  # Force kill if needed
                except ProcessLookupError:
                    pass  # Already stopped
            
            instance.status = ServiceStatus.STOPPED
            self.load_balancer.remove_instance(instance.service_name, instance.instance_id)
            
        except Exception as e:
            self.logger.error(f"Error stopping instance {instance.instance_id}: {e}")
            instance.status = ServiceStatus.FAILED
    
    def restart_service(self, service_name: str):
        """Restart a service (rolling restart)."""
        if service_name not in self.services:
            self.logger.error(f"Service not registered: {service_name}")
            return
        
        self.logger.info(f"Restarting service: {service_name}")
        
        # For rolling restart, start new instances before stopping old ones
        config = self.services[service_name]
        old_instances = self.running_instances.get(service_name, []).copy()
        
        # Start new instances
        self.start_service(service_name)
        
        # Wait for new instances to be healthy
        time.sleep(10)
        
        # Stop old instances
        for instance in old_instances:
            self._stop_service_instance(instance)
    
    def start_management(self):
        """Start the orchestration management loop."""
        self.management_thread = threading.Thread(target=self._management_worker, daemon=True)
        self.management_thread.start()
        self.logger.info("Production orchestrator started")
    
    def _management_worker(self):
        """Main management loop."""
        while not self.shutdown_event.is_set():
            try:
                # Check service health and restart failed instances
                self._check_and_restart_failed_services()
                
                # Cleanup stopped containers/processes
                self._cleanup_stopped_instances()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Management loop error: {e}")
                time.sleep(60)
    
    def _check_and_restart_failed_services(self):
        """Check for failed services and restart them."""
        for service_name, instances in self.running_instances.items():
            for instance in instances:
                if instance.status in [ServiceStatus.FAILED, ServiceStatus.UNHEALTHY]:
                    if instance.health_check_failures >= self.health_checker.failure_threshold:
                        self.logger.warning(f"Restarting failed instance: {instance.instance_id}")
                        self._stop_service_instance(instance)
                        
                        # Start replacement instance
                        config = self.services[service_name]
                        replica_id = int(instance.instance_id.split('-')[-1])
                        self._start_service_instance(config, replica_id)
    
    def _cleanup_stopped_instances(self):
        """Remove stopped instances from tracking."""
        for service_name in list(self.running_instances.keys()):
            self.running_instances[service_name] = [
                inst for inst in self.running_instances[service_name]
                if inst.status != ServiceStatus.STOPPED
            ]
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "services": {
                name: {
                    "config": asdict(config),
                    "instances": [
                        {
                            "instance_id": inst.instance_id,
                            "status": inst.status.value,
                            "port": inst.port,
                            "started_at": inst.started_at.isoformat(),
                            "health_failures": inst.health_check_failures,
                            "restart_count": inst.restart_count
                        }
                        for inst in self.running_instances.get(name, [])
                    ]
                }
                for name, config in self.services.items()
            },
            "health_summary": self.health_checker.get_health_summary(),
            "load_balancing": self.load_balancer.get_load_balancing_stats(),
            "management_active": self.management_thread and self.management_thread.is_alive()
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of all services."""
        self.logger.info("Starting graceful shutdown...")
        
        # Stop management loop
        self.shutdown_event.set()
        
        # Stop all services
        for service_name in list(self.services.keys()):
            self.stop_service(service_name)
        
        self.logger.info("Shutdown complete")


def create_default_config() -> Dict[str, Any]:
    """Create default production configuration."""
    return {
        "services": [
            {
                "name": "dgdm-api",
                "image": "dgdm-histopath:latest",
                "port": 8000,
                "health_check_path": "/health",
                "environment": {
                    "PYTHONPATH": "/app",
                    "LOG_LEVEL": "INFO"
                },
                "replicas": 2,
                "max_memory": "4g",
                "max_cpu": "2.0"
            },
            {
                "name": "dgdm-worker",
                "image": "dgdm-histopath:latest",
                "port": 8001,
                "health_check_path": "/health",
                "environment": {
                    "PYTHONPATH": "/app",
                    "WORKER_MODE": "true"
                },
                "depends_on": ["dgdm-api"],
                "replicas": 3,
                "max_memory": "8g",
                "max_cpu": "4.0"
            }
        ]
    }


if __name__ == "__main__":
    # Example usage
    orchestrator = ProductionOrchestrator()
    
    # Register services
    api_service = ServiceConfig(
        name="dgdm-api",
        image="dgdm-histopath:latest",
        port=8000,
        replicas=1
    )
    
    orchestrator.register_service(api_service)
    
    # Start services
    orchestrator.start_service("dgdm-api")
    
    # Start management
    orchestrator.start_management()
    
    # Print status
    status = orchestrator.get_orchestration_status()
    print("=" * 80)
    print("PRODUCTION ORCHESTRATOR STATUS")
    print("=" * 80)
    print(json.dumps(status, indent=2))
    
    try:
        # Keep running
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        orchestrator.shutdown()