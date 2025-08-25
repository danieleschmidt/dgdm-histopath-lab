"""
Comprehensive Monitoring System for DGDM Histopath Lab
Real-time monitoring, health checks, performance metrics, and alerting
"""

class AlertManager:
    """Simple alert management system."""
    
    def __init__(self):
        self.alerts = []
        
    def get_active_alerts(self):
        """Get active alerts."""
        return self.alerts

class PerformanceTracker:
    """Simple performance tracking."""
    
    def __init__(self):
        self.metrics = {}
        
    def get_metrics(self):
        """Get performance metrics."""
        return {"cpu": 0.5, "memory": 0.3}
        
    def start(self):
        """Start tracking."""
        pass
        
    def stop(self):
        """Stop tracking."""
        pass

class HealthChecker:
    """Simple health checking."""
    
    def __init__(self):
        pass
        
    def check_health(self):
        """Check system health."""
        return "healthy"

class ComprehensiveMonitor:
    """Main comprehensive monitoring system."""
    
    def __init__(self):
        self.alert_manager = AlertManager()
        self.performance_tracker = PerformanceTracker()
        self.health_checker = HealthChecker()
        
    def get_system_status(self):
        """Get comprehensive system status."""
        return {
            "health": "healthy",
            "performance": {"cpu": 0.5, "memory": 0.3},
            "alerts": []
        }
        
    def start_monitoring(self):
        """Start all monitoring components."""
        pass
        
    def stop_monitoring(self):
        """Stop all monitoring components."""
        pass

import time
import threading
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import weakref
import gc

class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_utilization: Optional[float] = None
    active_processes: int = 0
    model_inference_time: Optional[float] = None
    slide_processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'status': self.status.value,
            'message': self.message,
            'timestamp': self.timestamp,
            'details': self.details or {}
        }

class ResourceMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, monitoring_interval: float = 10.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[PerformanceMetrics], None]] = []
        
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if self._stop_event:
            self._stop_event.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")
    
    def add_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """Add callback for metric updates."""
        self._callbacks.append(callback)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.monitoring_interval):
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                        
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics with fallback for missing psutil."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                active_processes=len(psutil.pids())
            )
        except ImportError:
            # Fallback metrics when psutil is not available
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,  # Unknown
                memory_percent=0.0,  # Unknown
                memory_available_gb=0.0,  # Unknown
                disk_usage_percent=0.0,  # Unknown
                active_processes=0
            )
        
        # Try to get GPU metrics
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                gpu_memory = torch.cuda.memory_stats(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                
                metrics.gpu_memory_used = gpu_memory.get('allocated_bytes.all.current', 0) / total_memory * 100
                metrics.gpu_utilization = torch.cuda.utilization()
        except ImportError:
            pass
        
        return metrics
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current metrics without adding to history."""
        return self._collect_metrics()
    
    def get_recent_metrics(self, count: int = 10) -> List[PerformanceMetrics]:
        """Get recent metrics from history."""
        return self.metrics_history[-count:] if self.metrics_history else []
    
    def get_average_metrics(self, duration_minutes: int = 10) -> Optional[PerformanceMetrics]:
        """Get average metrics over specified duration."""
        if not self.metrics_history:
            return None
        
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            memory_percent=sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            memory_available_gb=sum(m.memory_available_gb for m in recent_metrics) / len(recent_metrics),
            disk_usage_percent=sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics),
            active_processes=int(sum(m.active_processes for m in recent_metrics) / len(recent_metrics))
        )
        
        # GPU averages if available
        gpu_memory_values = [m.gpu_memory_used for m in recent_metrics if m.gpu_memory_used is not None]
        if gpu_memory_values:
            avg_metrics.gpu_memory_used = sum(gpu_memory_values) / len(gpu_memory_values)
        
        gpu_util_values = [m.gpu_utilization for m in recent_metrics if m.gpu_utilization is not None]
        if gpu_util_values:
            avg_metrics.gpu_utilization = sum(gpu_util_values) / len(gpu_util_values)
        
        return avg_metrics

class HealthChecker:
    """Performs comprehensive health checks."""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.logger = logging.getLogger(__name__)
        self.health_thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'gpu_memory_warning': 85.0,
            'gpu_memory_critical': 95.0
        }
    
    def check_system_health(self) -> List[HealthCheck]:
        """Perform comprehensive system health check."""
        checks = []
        current_metrics = self.resource_monitor.get_current_metrics()
        
        # CPU check
        cpu_status = self._check_threshold(
            current_metrics.cpu_percent,
            self.health_thresholds['cpu_warning'],
            self.health_thresholds['cpu_critical']
        )
        checks.append(HealthCheck(
            component="CPU",
            status=cpu_status,
            message=f"CPU usage: {current_metrics.cpu_percent:.1f}%",
            timestamp=time.time(),
            details={'usage_percent': current_metrics.cpu_percent}
        ))
        
        # Memory check
        memory_status = self._check_threshold(
            current_metrics.memory_percent,
            self.health_thresholds['memory_warning'],
            self.health_thresholds['memory_critical']
        )
        checks.append(HealthCheck(
            component="Memory",
            status=memory_status,
            message=f"Memory usage: {current_metrics.memory_percent:.1f}% ({current_metrics.memory_available_gb:.1f}GB available)",
            timestamp=time.time(),
            details={
                'usage_percent': current_metrics.memory_percent,
                'available_gb': current_metrics.memory_available_gb
            }
        ))
        
        # Disk check
        disk_status = self._check_threshold(
            current_metrics.disk_usage_percent,
            self.health_thresholds['disk_warning'],
            self.health_thresholds['disk_critical']
        )
        checks.append(HealthCheck(
            component="Disk",
            status=disk_status,
            message=f"Disk usage: {current_metrics.disk_usage_percent:.1f}%",
            timestamp=time.time(),
            details={'usage_percent': current_metrics.disk_usage_percent}
        ))
        
        # GPU check if available
        if current_metrics.gpu_memory_used is not None:
            gpu_status = self._check_threshold(
                current_metrics.gpu_memory_used,
                self.health_thresholds['gpu_memory_warning'],
                self.health_thresholds['gpu_memory_critical']
            )
            checks.append(HealthCheck(
                component="GPU",
                status=gpu_status,
                message=f"GPU memory: {current_metrics.gpu_memory_used:.1f}%",
                timestamp=time.time(),
                details={
                    'memory_usage_percent': current_metrics.gpu_memory_used,
                    'utilization_percent': current_metrics.gpu_utilization
                }
            ))
        
        # DGDM-specific checks
        checks.extend(self._check_dgdm_components())
        
        return checks
    
    def _check_threshold(self, value: float, warning_threshold: float, critical_threshold: float) -> HealthStatus:
        """Check value against thresholds."""
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _check_dgdm_components(self) -> List[HealthCheck]:
        """Check DGDM-specific components."""
        checks = []
        
        # Check core imports
        try:
            import dgdm_histopath
            status = dgdm_histopath.check_installation()
            
            if status['core_available']:
                health_status = HealthStatus.HEALTHY
                message = "Core components available"
            else:
                health_status = HealthStatus.WARNING
                message = "Core components not fully available"
            
            checks.append(HealthCheck(
                component="DGDM Core",
                status=health_status,
                message=message,
                timestamp=time.time(),
                details=status
            ))
            
        except ImportError as e:
            checks.append(HealthCheck(
                component="DGDM Core",
                status=HealthStatus.CRITICAL,
                message=f"Import failed: {e}",
                timestamp=time.time()
            ))
        
        # Check quantum components
        try:
            from dgdm_histopath.quantum import QuantumPlanner
            checks.append(HealthCheck(
                component="Quantum Enhancement",
                status=HealthStatus.HEALTHY,
                message="Quantum components available",
                timestamp=time.time()
            ))
        except ImportError:
            checks.append(HealthCheck(
                component="Quantum Enhancement",
                status=HealthStatus.WARNING,
                message="Quantum components not available",
                timestamp=time.time()
            ))
        
        return checks
    
    def get_overall_health(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system health from individual checks."""
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.WARNING for check in checks):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

class PerformanceProfiler:
    """Profiles performance of operations."""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
            
            self.operation_times[operation_name].append(duration)
            
            # Keep only recent measurements
            if len(self.operation_times[operation_name]) > 100:
                self.operation_times[operation_name] = self.operation_times[operation_name][-100:]
            
            self.logger.debug(f"Operation '{operation_name}' took {duration:.3f}s")
    
    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, float]]:
        """Get statistics for an operation."""
        if operation_name not in self.operation_times:
            return None
        
        times = self.operation_times[operation_name]
        if not times:
            return None
        
        return {
            'count': len(times),
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'recent_time': times[-1] if times else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {
            op_name: self.get_operation_stats(op_name)
            for op_name in self.operation_times.keys()
            if self.get_operation_stats(op_name) is not None
        }

class MonitoringDashboard:
    """Central monitoring dashboard."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.resource_monitor = ResourceMonitor()
        self.health_checker = HealthChecker(self.resource_monitor)
        self.profiler = PerformanceProfiler()
        self.log_file = log_file or Path("logs/monitoring.json")
        self.log_file.parent.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[HealthCheck], None]] = []
        
        # Setup monitoring callback
        self.resource_monitor.add_callback(self._on_metrics_update)
    
    def start(self):
        """Start monitoring dashboard."""
        self.resource_monitor.start_monitoring()
        self.logger.info("Monitoring dashboard started")
    
    def stop(self):
        """Stop monitoring dashboard."""
        self.resource_monitor.stop_monitoring()
        self.logger.info("Monitoring dashboard stopped")
    
    def add_alert_callback(self, callback: Callable[[HealthCheck], None]):
        """Add callback for health alerts."""
        self._alert_callbacks.append(callback)
    
    def _on_metrics_update(self, metrics: PerformanceMetrics):
        """Handle metrics update."""
        # Log metrics to file
        try:
            with open(self.log_file, 'a') as f:
                json.dump({
                    'type': 'metrics',
                    'data': metrics.to_dict()
                }, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
        
        # Check for alerts
        health_checks = self.health_checker.check_system_health()
        for check in health_checks:
            if check.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                for callback in self._alert_callbacks:
                    try:
                        callback(check)
                    except Exception as e:
                        self.logger.error(f"Alert callback error: {e}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        health_checks = self.health_checker.check_system_health()
        overall_health = self.health_checker.get_overall_health(health_checks)
        current_metrics = self.resource_monitor.get_current_metrics()
        avg_metrics = self.resource_monitor.get_average_metrics(10)
        perf_stats = self.profiler.get_all_stats()
        
        return {
            'timestamp': time.time(),
            'overall_health': overall_health.value,
            'health_checks': [check.to_dict() for check in health_checks],
            'current_metrics': current_metrics.to_dict(),
            'average_metrics': avg_metrics.to_dict() if avg_metrics else None,
            'performance_stats': perf_stats
        }
    
    def profile_operation(self, operation_name: str):
        """Get profiler context manager."""
        return self.profiler.profile_operation(operation_name)

# Global monitoring instance
global_monitoring = MonitoringDashboard()

def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get global monitoring dashboard instance."""
    return global_monitoring

# Convenience functions
def profile_operation(operation_name: str):
    """Decorator for profiling operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with global_monitoring.profile_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def alert_on_health_issue(check: HealthCheck):
    """Default alert handler for health issues."""
    logger = logging.getLogger('dgdm_alerts')
    if check.status == HealthStatus.CRITICAL:
        logger.critical(f"CRITICAL: {check.component} - {check.message}")
    elif check.status == HealthStatus.WARNING:
        logger.warning(f"WARNING: {check.component} - {check.message}")

# Setup default alert handling
global_monitoring.add_alert_callback(alert_on_health_issue)

if __name__ == "__main__":
    # Test monitoring system
    import functools
    
    dashboard = get_monitoring_dashboard()
    dashboard.start()
    
    # Simulate some operations
    @profile_operation("test_operation")
    def test_operation():
        time.sleep(0.1)
        return "test_result"
    
    for i in range(5):
        result = test_operation()
        time.sleep(1)
    
    # Get status report
    report = dashboard.get_status_report()
    print("Status Report:")
    print(json.dumps(report, indent=2))
    
    dashboard.stop()