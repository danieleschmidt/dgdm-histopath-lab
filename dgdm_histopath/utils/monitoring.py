"""System monitoring and health check utilities."""

import time
import psutil
import threading
import logging
import json
import os
import gc
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import warnings


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    gpu_memory_used: Optional[int] = None
    gpu_memory_total: Optional[int] = None
    gpu_utilization: Optional[float] = None
    

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


class MetricsCollector:
    """Collect and store performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics: List[SystemMetrics] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # GPU availability check
        self.gpu_available = False
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.torch = torch
        except ImportError:
            pass
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
                memory_available=psutil.virtual_memory().available,
                disk_usage_percent=psutil.disk_usage('/').percent
            )
            
            # Add GPU metrics if available
            if self.gpu_available:
                try:
                    memory_info = self.torch.cuda.mem_get_info()
                    metrics.gpu_memory_total = memory_info[1]
                    metrics.gpu_memory_used = memory_info[1] - memory_info[0]
                    metrics.gpu_utilization = self.torch.cuda.utilization()
                except Exception as e:
                    self.logger.debug(f"Failed to collect GPU metrics: {e}")
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
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
        """Compute summary statistics."""
        summary = {
            'period_minutes': 5,
            'system_health': 'unknown',
            'avg_cpu_percent': 0,
            'avg_memory_percent': 0,
            'total_operations': len(performance_metrics),
            'failed_operations': 0,
            'avg_operation_duration': 0
        }
        
        if system_metrics:
            summary['avg_cpu_percent'] = sum(m.cpu_percent for m in system_metrics) / len(system_metrics)
            summary['avg_memory_percent'] = sum(m.memory_percent for m in system_metrics) / len(system_metrics)
            
            # Determine system health
            if summary['avg_cpu_percent'] > 90 or summary['avg_memory_percent'] > 90:
                summary['system_health'] = 'critical'
            elif summary['avg_cpu_percent'] > 70 or summary['avg_memory_percent'] > 70:
                summary['system_health'] = 'warning'
            else:
                summary['system_health'] = 'healthy'
                
        if performance_metrics:
            summary['failed_operations'] = sum(1 for m in performance_metrics if not m.success)
            summary['avg_operation_duration'] = sum(m.duration for m in performance_metrics) / len(performance_metrics)
            
        return summary
    
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


# Global metrics collector instance
metrics_collector = MetricsCollector()


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