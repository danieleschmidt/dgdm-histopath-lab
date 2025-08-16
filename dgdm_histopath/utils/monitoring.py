"""
Advanced real-time monitoring and alerting system for DGDM Histopath Lab.

Provides comprehensive monitoring of system performance, model behavior,
and clinical deployment metrics with automated alerting capabilities.
"""

import time
import psutil
import threading
import logging
import json
import os
import gc
import queue
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict, deque
import warnings
import hashlib

from dgdm_histopath.utils.exceptions import (
    PerformanceError, ResourceError, global_exception_handler
)


@dataclass
class SystemMetrics:
    """Enhanced system performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    gpu_memory_used: Optional[int] = None
    gpu_memory_total: Optional[int] = None
    gpu_utilization: Optional[float] = None
    network_io_sent: Optional[int] = None
    network_io_recv: Optional[int] = None
    load_average: Optional[float] = None
    open_files: Optional[int] = None
    

@dataclass  
class PerformanceMetrics:
    """Application performance metrics."""
    operation: str
    duration: float
    memory_before: int
    memory_after: int
    success: bool
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class AdvancedMetricsCollector:
    """Advanced metrics collector with real-time alerting and analytics."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.system_metrics: List[SystemMetrics] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        self.custom_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.alert_queue = queue.Queue()
        self.collection_callbacks = []
        
        # Enhanced GPU availability check
        self.gpu_available = False
        self.gpu_count = 0
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.torch = torch
                self.gpu_count = torch.cuda.device_count()
        except ImportError:
            pass
        
        # Initialize network IO baseline
        self.network_io_baseline = psutil.net_io_counters()
        
        # Performance thresholds for alerts
        self.alert_thresholds = {
            'cpu_percent': {'warning': 80, 'critical': 95},
            'memory_percent': {'warning': 85, 'critical': 95},
            'disk_usage_percent': {'warning': 85, 'critical': 95},
            'gpu_memory_percent': {'warning': 85, 'critical': 95}
        }
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        try:
            # Basic system metrics
            vm = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network IO metrics
            current_net_io = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=vm.percent,
                memory_available=vm.available,
                disk_usage_percent=(disk.used / disk.total) * 100,
                network_io_sent=current_net_io.bytes_sent - self.network_io_baseline.bytes_sent,
                network_io_recv=current_net_io.bytes_recv - self.network_io_baseline.bytes_recv,
                load_average=os.getloadavg()[0] if hasattr(os, 'getloadavg') else None,
                open_files=len(psutil.Process().open_files()) if hasattr(psutil.Process(), 'open_files') else None
            )
            
            # Enhanced GPU metrics for multiple GPUs
            if self.gpu_available and self.gpu_count > 0:
                try:
                    total_gpu_memory = 0
                    total_gpu_used = 0
                    gpu_utilizations = []
                    
                    for gpu_id in range(self.gpu_count):
                        memory_info = self.torch.cuda.mem_get_info(gpu_id)
                        total_gpu_memory += memory_info[1]
                        total_gpu_used += memory_info[1] - memory_info[0]
                        
                        # Try to get utilization if pynvml is available
                        try:
                            import pynvml
                            if not hasattr(self, 'nvml_initialized'):
                                pynvml.nvmlInit()
                                self.nvml_initialized = True
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_utilizations.append(util.gpu)
                        except ImportError:
                            pass
                    
                    metrics.gpu_memory_total = total_gpu_memory
                    metrics.gpu_memory_used = total_gpu_used
                    if gpu_utilizations:
                        metrics.gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations)
                        
                except Exception as e:
                    self.logger.debug(f"Failed to collect GPU metrics: {e}")
            
            # Check for alert conditions
            self._check_metric_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            global_exception_handler.handle_exception(e, reraise=False)
            raise
    
    def record_system_metrics(self):
        """Record current system metrics."""
        try:
            metrics = self.collect_system_metrics()
            
            with self.lock:
                self.system_metrics.append(metrics)
                
                # Maintain max history
                if len(self.system_metrics) > self.max_history:
                    self.system_metrics = self.system_metrics[-self.max_history:]
                    
        except Exception as e:
            self.logger.error(f"Failed to record system metrics: {e}")
    
    def record_performance_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self.lock:
            self.performance_metrics.append(metrics)
            
            # Maintain max history
            if len(self.performance_metrics) > self.max_history:
                self.performance_metrics = self.performance_metrics[-self.max_history:]
    
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_system = [
                m for m in self.system_metrics
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            recent_performance = [
                m for m in self.performance_metrics
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
        return {
            'system_metrics': [asdict(m) for m in recent_system],
            'performance_metrics': [asdict(m) for m in recent_performance],
            'summary': self._compute_summary(recent_system, recent_performance)
        }
    
    def _compute_summary(self, system_metrics: List[SystemMetrics], performance_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Compute enhanced summary statistics with trend analysis."""
        summary = {
            'period_minutes': 5,
            'system_health': 'unknown',
            'avg_cpu_percent': 0,
            'avg_memory_percent': 0,
            'max_cpu_percent': 0,
            'max_memory_percent': 0,
            'total_operations': len(performance_metrics),
            'failed_operations': 0,
            'success_rate': 0,
            'avg_operation_duration': 0,
            'p95_operation_duration': 0,
            'network_throughput_mbps': 0,
            'gpu_utilization': 0,
            'trends': {}
        }
        
        if system_metrics:
            cpu_values = [m.cpu_percent for m in system_metrics]
            memory_values = [m.memory_percent for m in system_metrics]
            
            summary['avg_cpu_percent'] = np.mean(cpu_values)
            summary['avg_memory_percent'] = np.mean(memory_values)
            summary['max_cpu_percent'] = np.max(cpu_values)
            summary['max_memory_percent'] = np.max(memory_values)
            
            # Network throughput calculation
            if len(system_metrics) > 1 and system_metrics[-1].network_io_sent:
                time_diff = (datetime.fromisoformat(system_metrics[-1].timestamp) - 
                           datetime.fromisoformat(system_metrics[0].timestamp)).total_seconds()
                if time_diff > 0:
                    bytes_diff = (system_metrics[-1].network_io_sent + system_metrics[-1].network_io_recv)
                    summary['network_throughput_mbps'] = (bytes_diff * 8) / (time_diff * 1024 * 1024)
            
            # GPU utilization
            gpu_utils = [m.gpu_utilization for m in system_metrics if m.gpu_utilization is not None]
            if gpu_utils:
                summary['gpu_utilization'] = np.mean(gpu_utils)
            
            # Trend analysis (simple linear trend)
            if len(system_metrics) >= 3:
                cpu_trend = self._calculate_trend(cpu_values)
                memory_trend = self._calculate_trend(memory_values)
                summary['trends'] = {
                    'cpu_trend': 'increasing' if cpu_trend > 0.1 else 'decreasing' if cpu_trend < -0.1 else 'stable',
                    'memory_trend': 'increasing' if memory_trend > 0.1 else 'decreasing' if memory_trend < -0.1 else 'stable'
                }
            
            # Enhanced system health determination
            if summary['max_cpu_percent'] > 95 or summary['max_memory_percent'] > 95:
                summary['system_health'] = 'critical'
            elif summary['avg_cpu_percent'] > 85 or summary['avg_memory_percent'] > 85:
                summary['system_health'] = 'warning'
            elif summary['trends'].get('cpu_trend') == 'increasing' and summary['avg_cpu_percent'] > 70:
                summary['system_health'] = 'degrading'
            else:
                summary['system_health'] = 'healthy'
                
        if performance_metrics:
            failed = sum(1 for m in performance_metrics if not m.success)
            summary['failed_operations'] = failed
            summary['success_rate'] = 1.0 - (failed / len(performance_metrics))
            
            durations = [m.duration for m in performance_metrics]
            summary['avg_operation_duration'] = np.mean(durations)
            summary['p95_operation_duration'] = np.percentile(durations, 95)
            
        return summary
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend slope."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(values)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        return slope
    
    def _check_metric_alerts(self, metrics: SystemMetrics) -> None:
        """Check metrics against alert thresholds."""
        alerts = []
        
        # CPU alerts
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']['critical']:
            alerts.append(('critical', f'CPU usage critical: {metrics.cpu_percent:.1f}%'))
        elif metrics.cpu_percent > self.alert_thresholds['cpu_percent']['warning']:
            alerts.append(('warning', f'CPU usage high: {metrics.cpu_percent:.1f}%'))
        
        # Memory alerts
        if metrics.memory_percent > self.alert_thresholds['memory_percent']['critical']:
            alerts.append(('critical', f'Memory usage critical: {metrics.memory_percent:.1f}%'))
        elif metrics.memory_percent > self.alert_thresholds['memory_percent']['warning']:
            alerts.append(('warning', f'Memory usage high: {metrics.memory_percent:.1f}%'))
        
        # GPU memory alerts
        if metrics.gpu_memory_used and metrics.gpu_memory_total:
            gpu_percent = (metrics.gpu_memory_used / metrics.gpu_memory_total) * 100
            if gpu_percent > self.alert_thresholds['gpu_memory_percent']['critical']:
                alerts.append(('critical', f'GPU memory critical: {gpu_percent:.1f}%'))
            elif gpu_percent > self.alert_thresholds['gpu_memory_percent']['warning']:
                alerts.append(('warning', f'GPU memory high: {gpu_percent:.1f}%'))
        
        # Queue alerts for processing
        for severity, message in alerts:
            try:
                self.alert_queue.put_nowait({'severity': severity, 'message': message, 'timestamp': metrics.timestamp})
            except queue.Full:
                self.logger.warning("Alert queue full, dropping alert")
    
    def add_collection_callback(self, callback: Callable[[SystemMetrics], None]) -> None:
        """Add callback for metric collection events."""
        self.collection_callbacks.append(callback)
    
    def record_custom_metric(self, name: str, value: Union[float, int], tags: Optional[Dict[str, str]] = None) -> None:
        """Record custom application metrics."""
        with self.lock:
            metric_data = {
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'tags': tags or {}
            }
            self.custom_metrics[name].append(metric_data)
    
    def get_custom_metric_stats(self, name: str, minutes: int = 5) -> Dict[str, Any]:
        """Get statistics for custom metrics."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_values = [
                entry for entry in self.custom_metrics[name]
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]
        
        if not recent_values:
            return {}
        
        values = [entry['value'] for entry in recent_values]
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
    
    def export_metrics(self, output_path: Path):
        """Export metrics to JSON file."""
        with self.lock:
            data = {
                'export_time': datetime.now().isoformat(),
                'system_metrics': [asdict(m) for m in self.system_metrics],
                'performance_metrics': [asdict(m) for m in self.performance_metrics]
            }
            
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


# Global enhanced metrics collector instance
metrics_collector = AdvancedMetricsCollector()


class HealthChecker:
    """System health monitoring and alerting."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alert_thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'error_rate': 0.1,  # 10% error rate
        }
        self.alert_callbacks: List[Callable] = []
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'alerts': []
        }
        
        try:
            # System resource checks
            system_metrics = metrics_collector.collect_system_metrics()
            
            # CPU check
            if system_metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
                health_report['checks']['cpu'] = 'warning'
                health_report['alerts'].append(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
            else:
                health_report['checks']['cpu'] = 'healthy'
            
            # Memory check
            if system_metrics.memory_percent > self.alert_thresholds['memory_percent']:
                health_report['checks']['memory'] = 'warning'  
                health_report['alerts'].append(f"High memory usage: {system_metrics.memory_percent:.1f}%")
            else:
                health_report['checks']['memory'] = 'healthy'
            
            # Disk check
            if system_metrics.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
                health_report['checks']['disk'] = 'warning'
                health_report['alerts'].append(f"High disk usage: {system_metrics.disk_usage_percent:.1f}%")
            else:
                health_report['checks']['disk'] = 'healthy'
            
            # GPU check (if available)
            if system_metrics.gpu_memory_used is not None:
                gpu_usage_percent = (system_metrics.gpu_memory_used / system_metrics.gpu_memory_total) * 100
                if gpu_usage_percent > 90:
                    health_report['checks']['gpu'] = 'warning'
                    health_report['alerts'].append(f"High GPU memory usage: {gpu_usage_percent:.1f}%")
                else:
                    health_report['checks']['gpu'] = 'healthy'
            
            # Performance checks
            recent_metrics = metrics_collector.get_recent_metrics(minutes=5)
            if recent_metrics['performance_metrics']:
                error_rate = recent_metrics['summary']['failed_operations'] / recent_metrics['summary']['total_operations']
                if error_rate > self.alert_thresholds['error_rate']:
                    health_report['checks']['error_rate'] = 'warning'
                    health_report['alerts'].append(f"High error rate: {error_rate:.1%}")
                else:
                    health_report['checks']['error_rate'] = 'healthy'
            
            # Determine overall status
            if any(status == 'warning' for status in health_report['checks'].values()):
                health_report['overall_status'] = 'warning'
            if health_report['alerts']:
                health_report['overall_status'] = 'degraded'
                
            # Trigger alerts if necessary
            if health_report['alerts']:
                for callback in self.alert_callbacks:
                    try:
                        callback("system_health_alert", health_report)
                    except Exception as e:
                        self.logger.error(f"Alert callback failed: {e}")
                        
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health_report['overall_status'] = 'error'
            health_report['alerts'].append(f"Health check failed: {e}")
            
        return health_report


# Global health checker instance
health_checker = HealthChecker()


@contextmanager
def monitor_operation(operation_name: str, auto_gc: bool = True):
    """Context manager to monitor operation performance."""
    logger = logging.getLogger(__name__)
    
    # Record initial state
    start_time = time.time()
    memory_before = psutil.Process().memory_info().rss
    
    success = True
    error_message = None
    
    try:
        logger.info(f"Starting monitored operation: {operation_name}")
        yield
        
    except Exception as e:
        success = False
        error_message = str(e)
        logger.error(f"Operation failed: {operation_name} - {e}")
        raise
        
    finally:
        # Record final state
        end_time = time.time()
        duration = end_time - start_time
        memory_after = psutil.Process().memory_info().rss
        
        # Optional garbage collection
        if auto_gc:
            gc.collect()
            
        # Record metrics
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration=duration,
            memory_before=memory_before,
            memory_after=memory_after,
            success=success,
            error_message=error_message
        )
        
        metrics_collector.record_performance_metrics(metrics)
        
        if success:
            logger.info(f"Operation completed: {operation_name} ({duration:.2f}s)")
        else:
            logger.error(f"Operation failed: {operation_name} ({duration:.2f}s)")


class ResourceLimiter:
    """Enforce resource limits and prevent resource exhaustion."""
    
    def __init__(self, max_memory_percent: float = 80.0, max_gpu_memory_percent: float = 90.0):
        self.max_memory_percent = max_memory_percent
        self.max_gpu_memory_percent = max_gpu_memory_percent
        self.logger = logging.getLogger(__name__)
        
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.max_memory_percent:
            self.logger.warning(f"Memory usage {memory_percent:.1f}% exceeds limit {self.max_memory_percent:.1f}%")
            return False
        return True
    
    def check_gpu_memory_limit(self) -> bool:
        """Check if GPU memory usage is within limits."""
        try:
            import torch
            if torch.cuda.is_available():
                memory_info = torch.cuda.mem_get_info()
                memory_used_percent = ((memory_info[1] - memory_info[0]) / memory_info[1]) * 100
                if memory_used_percent > self.max_gpu_memory_percent:
                    self.logger.warning(f"GPU memory usage {memory_used_percent:.1f}% exceeds limit {self.max_gpu_memory_percent:.1f}%")
                    return False
        except ImportError:
            pass
        return True
    
    def enforce_limits(self) -> bool:
        """Enforce all resource limits."""
        memory_ok = self.check_memory_limit()
        gpu_memory_ok = self.check_gpu_memory_limit()
        
        if not memory_ok:
            # Force garbage collection
            gc.collect()
            
        if not gpu_memory_ok:
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
                
        return memory_ok and gpu_memory_ok


def start_background_monitoring(interval_seconds: int = 30):
    """Start background system monitoring."""
    logger = logging.getLogger(__name__)
    
    def monitoring_loop():
        while True:
            try:
                metrics_collector.record_system_metrics()
                health_report = health_checker.check_system_health()
                
                if health_report['overall_status'] != 'healthy':
                    logger.warning(f"System health: {health_report['overall_status']}")
                    for alert in health_report['alerts']:
                        logger.warning(f"Health alert: {alert}")
                        
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Background monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                time.sleep(interval_seconds)
    
    # Start monitoring in background thread
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()
    logger.info("Background system monitoring started")
    
    return monitoring_thread


# Simple alert callback that logs to file
def file_alert_callback(alert_type: str, data: Dict[str, Any]):
    """Log alerts to file."""
    alert_log_path = Path("logs/alerts.log")
    alert_log_path.parent.mkdir(exist_ok=True)
    
    with open(alert_log_path, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {alert_type}: {json.dumps(data)}\n")


# Register default alert callback
health_checker.add_alert_callback(file_alert_callback)

# Enum for metric types
from enum import Enum

class MetricType(Enum):
    """Metric type enumeration for monitoring."""
    COUNTER = "counter"
    GAUGE = "gauge"
    TIMER = "timer"
    HISTOGRAM = "histogram"

class MonitoringScope(Enum):
    """Monitoring scope enumeration."""
    SYSTEM = "system"
    APPLICATION = "application"
    PERFORMANCE = "performance"
    CLINICAL = "clinical"

# Simple metrics collector for compatibility
def record_metric(name, value, metric_type=None, scope=None, **kwargs):
    """Record a metric with optional type and scope."""
    pass

# Global metrics collector instance
metrics_collector = AdvancedMetricsCollector()
