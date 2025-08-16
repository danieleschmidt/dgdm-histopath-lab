"""
Advanced Monitoring and Observability for Medical AI Systems

Comprehensive monitoring, alerting, and observability framework
for production medical AI deployments with real-time health tracking.
"""

import time
import threading
import queue
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging
import warnings
from pathlib import Path
import hashlib
import statistics

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from dgdm_histopath.utils.exceptions import DGDMException


class MetricType(Enum):
    """Types of metrics for monitoring."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    DISTRIBUTION = "distribution"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringScope(Enum):
    """Scope of monitoring."""
    SYSTEM = "system"
    MODEL = "model"
    DATA = "data"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CLINICAL = "clinical"


@dataclass
class Metric:
    """Individual metric data structure."""
    name: str
    value: Union[float, int, str]
    metric_type: MetricType
    scope: MonitoringScope
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    name: str
    severity: AlertSeverity
    scope: MonitoringScope
    message: str
    triggered_by: str
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """System health status."""
    overall_status: str  # HEALTHY, DEGRADED, UNHEALTHY, CRITICAL
    component_statuses: Dict[str, str] = field(default_factory=dict)
    active_alerts: List[Alert] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """
    High-performance metrics collection system with buffering,
    aggregation, and real-time streaming capabilities.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        flush_interval: float = 10.0,
        enable_async: bool = True,
        retention_days: int = 30
    ):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.enable_async = enable_async
        self.retention_days = retention_days
        
        # Thread-safe metric storage
        self.metrics_buffer = queue.Queue(maxsize=buffer_size)
        self.aggregated_metrics = {}
        self.metric_history = {}
        
        # Threading components
        self._stop_event = threading.Event()
        self._flush_thread = None
        self._cleanup_thread = None
        
        # Start background threads
        if enable_async:
            self._start_background_threads()
        
        self.logger = logging.getLogger(__name__)
    
    def record_metric(
        self,
        name: str,
        value: Union[float, int, str],
        metric_type: MetricType = MetricType.GAUGE,
        scope: MonitoringScope = MonitoringScope.SYSTEM,
        tags: Optional[Dict[str, str]] = None,
        **metadata
    ):
        """Record a metric with thread-safe buffering."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            scope=scope,
            tags=tags or {},
            metadata=metadata
        )
        
        try:
            if self.enable_async:
                self.metrics_buffer.put_nowait(metric)
            else:
                self._process_metric_immediate(metric)
        except queue.Full:
            self.logger.warning(f"Metrics buffer full, dropping metric: {name}")
    
    def _process_metric_immediate(self, metric: Metric):
        """Process metric immediately (synchronous mode)."""
        metric_key = f"{metric.scope.value}.{metric.name}"
        
        # Store in aggregated metrics
        if metric_key not in self.aggregated_metrics:
            self.aggregated_metrics[metric_key] = []
        
        self.aggregated_metrics[metric_key].append({
            "value": metric.value,
            "timestamp": metric.timestamp.isoformat(),
            "tags": metric.tags,
            "metadata": metric.metadata
        })
        
        # Maintain history
        if metric_key not in self.metric_history:
            self.metric_history[metric_key] = []
        
        self.metric_history[metric_key].append(metric)
        
        # Limit history size
        if len(self.metric_history[metric_key]) > 1000:
            self.metric_history[metric_key] = self.metric_history[metric_key][-1000:]
    
    def _start_background_threads(self):
        """Start background threads for async processing."""
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        
        self._flush_thread.start()
        self._cleanup_thread.start()
    
    def _flush_worker(self):
        """Background thread for flushing metrics."""
        while not self._stop_event.is_set():
            try:
                # Process all queued metrics
                while not self.metrics_buffer.empty():
                    try:
                        metric = self.metrics_buffer.get_nowait()
                        self._process_metric_immediate(metric)
                    except queue.Empty:
                        break
                
                # Wait for next flush interval
                self._stop_event.wait(self.flush_interval)
                
            except Exception as e:
                self.logger.error(f"Error in metrics flush worker: {e}")
    
    def _cleanup_worker(self):
        """Background thread for cleaning up old metrics."""
        while not self._stop_event.is_set():
            try:
                cutoff_time = datetime.now() - timedelta(days=self.retention_days)
                
                # Clean up old metrics
                for metric_key in list(self.metric_history.keys()):
                    self.metric_history[metric_key] = [
                        m for m in self.metric_history[metric_key]
                        if m.timestamp > cutoff_time
                    ]
                    
                    if not self.metric_history[metric_key]:
                        del self.metric_history[metric_key]
                
                # Sleep for 1 hour before next cleanup
                self._stop_event.wait(3600)
                
            except Exception as e:
                self.logger.error(f"Error in metrics cleanup worker: {e}")
    
    def get_metrics(
        self,
        scope: Optional[MonitoringScope] = None,
        name_pattern: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve metrics with filtering options."""
        filtered_metrics = {}
        
        for metric_key, metric_list in self.aggregated_metrics.items():
            # Filter by scope
            if scope and not metric_key.startswith(scope.value):
                continue
            
            # Filter by name pattern
            if name_pattern and name_pattern not in metric_key:
                continue
            
            # Filter by time range
            if time_range:
                start_time, end_time = time_range
                filtered_list = [
                    m for m in metric_list
                    if start_time <= datetime.fromisoformat(m["timestamp"]) <= end_time
                ]
            else:
                filtered_list = metric_list
            
            if filtered_list:
                filtered_metrics[metric_key] = filtered_list
        
        return filtered_metrics
    
    def get_aggregated_stats(
        self,
        metric_name: str,
        scope: MonitoringScope,
        aggregation_window: timedelta = timedelta(minutes=5)
    ) -> Dict[str, float]:
        """Get aggregated statistics for a metric."""
        metric_key = f"{scope.value}.{metric_name}"
        
        if metric_key not in self.metric_history:
            return {}
        
        # Filter to aggregation window
        cutoff_time = datetime.now() - aggregation_window
        recent_metrics = [
            m for m in self.metric_history[metric_key]
            if m.timestamp > cutoff_time and isinstance(m.value, (int, float))
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        if not NUMPY_AVAILABLE:
            # Fallback to basic statistics
            return {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99))
        }
    
    def shutdown(self):
        """Gracefully shutdown the metrics collector."""
        self.logger.info("Shutting down metrics collector...")
        
        if self.enable_async:
            self._stop_event.set()
            
            if self._flush_thread and self._flush_thread.is_alive():
                self._flush_thread.join(timeout=5.0)
            
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5.0)
        
        # Final flush
        while not self.metrics_buffer.empty():
            try:
                metric = self.metrics_buffer.get_nowait()
                self._process_metric_immediate(metric)
            except queue.Empty:
                break


class AlertingSystem:
    """
    Intelligent alerting system with threshold-based rules,
    anomaly detection, and escalation policies.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_handlers: Optional[List[Callable[[Alert], None]]] = None,
        enable_anomaly_detection: bool = True
    ):
        self.metrics_collector = metrics_collector
        self.alert_handlers = alert_handlers or []
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Alert storage
        self.active_alerts = {}
        self.alert_history = []
        self.alert_rules = {}
        
        # Anomaly detection state
        self.baseline_metrics = {}
        self.anomaly_thresholds = {}
        
        self.logger = logging.getLogger(__name__)
    
    def add_alert_rule(
        self,
        rule_name: str,
        metric_name: str,
        scope: MonitoringScope,
        threshold: float,
        comparison: str = "greater",  # greater, less, equal, not_equal
        severity: AlertSeverity = AlertSeverity.WARNING,
        duration_seconds: float = 60.0,
        **metadata
    ):
        """Add a threshold-based alert rule."""
        self.alert_rules[rule_name] = {
            "metric_name": metric_name,
            "scope": scope,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "duration_seconds": duration_seconds,
            "metadata": metadata,
            "last_triggered": None,
            "violation_start": None
        }
        
        self.logger.info(f"Added alert rule: {rule_name}")
    
    def check_alerts(self):
        """Check all alert rules and trigger alerts if conditions are met."""
        for rule_name, rule in self.alert_rules.items():
            try:
                self._check_individual_rule(rule_name, rule)
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule_name}: {e}")
        
        # Check for anomalies if enabled
        if self.enable_anomaly_detection:
            self._check_anomalies()
    
    def _check_individual_rule(self, rule_name: str, rule: Dict[str, Any]):
        """Check an individual alert rule."""
        metric_name = rule["metric_name"]
        scope = rule["scope"]
        
        # Get recent metric values
        stats = self.metrics_collector.get_aggregated_stats(
            metric_name, scope, timedelta(minutes=1)
        )
        
        if not stats:
            return
        
        current_value = stats.get("mean", 0.0)
        threshold = rule["threshold"]
        comparison = rule["comparison"]
        
        # Check threshold violation
        violation = False
        if comparison == "greater" and current_value > threshold:
            violation = True
        elif comparison == "less" and current_value < threshold:
            violation = True
        elif comparison == "equal" and abs(current_value - threshold) < 1e-6:
            violation = True
        elif comparison == "not_equal" and abs(current_value - threshold) >= 1e-6:
            violation = True
        
        now = datetime.now()
        
        if violation:
            if rule["violation_start"] is None:
                rule["violation_start"] = now
            elif (now - rule["violation_start"]).total_seconds() >= rule["duration_seconds"]:
                # Duration threshold met, trigger alert
                if rule_name not in self.active_alerts:
                    alert = Alert(
                        alert_id=self._generate_alert_id(rule_name),
                        name=rule_name,
                        severity=rule["severity"],
                        scope=scope,
                        message=f"Metric {metric_name} {comparison} {threshold} (current: {current_value:.3f})",
                        triggered_by=f"threshold_rule_{rule_name}",
                        threshold_value=threshold,
                        current_value=current_value,
                        metadata=rule["metadata"]
                    )
                    
                    self._trigger_alert(alert)
                    rule["last_triggered"] = now
        else:
            # Reset violation start time
            rule["violation_start"] = None
            
            # Resolve alert if it was active
            if rule_name in self.active_alerts:
                self._resolve_alert(rule_name)
    
    def _check_anomalies(self):
        """Check for anomalous metric behavior using statistical methods."""
        # Get all recent metrics
        for scope in MonitoringScope:
            recent_metrics = self.metrics_collector.get_metrics(
                scope=scope,
                time_range=(datetime.now() - timedelta(minutes=10), datetime.now())
            )
            
            for metric_key, metric_data in recent_metrics.items():
                if len(metric_data) < 10:  # Need sufficient data
                    continue
                
                try:
                    self._detect_metric_anomaly(metric_key, metric_data)
                except Exception as e:
                    self.logger.debug(f"Anomaly detection error for {metric_key}: {e}")
    
    def _detect_metric_anomaly(self, metric_key: str, metric_data: List[Dict[str, Any]]):
        """Detect anomalies in a specific metric using statistical methods."""
        # Extract numeric values
        values = []
        for data_point in metric_data:
            try:
                value = float(data_point["value"])
                values.append(value)
            except (ValueError, TypeError):
                continue
        
        if len(values) < 5:
            return
        
        # Simple anomaly detection using z-score
        if NUMPY_AVAILABLE:
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            
            if std_val > 0:
                z_scores = np.abs((values_array - mean_val) / std_val)
                max_z_score = np.max(z_scores)
                
                # Trigger anomaly alert if z-score > 3 (very unusual)
                if max_z_score > 3.0:
                    anomaly_alert_id = f"anomaly_{metric_key}_{int(time.time())}"
                    
                    if anomaly_alert_id not in self.active_alerts:
                        alert = Alert(
                            alert_id=anomaly_alert_id,
                            name=f"Anomaly in {metric_key}",
                            severity=AlertSeverity.WARNING,
                            scope=MonitoringScope.SYSTEM,
                            message=f"Statistical anomaly detected in {metric_key} (z-score: {max_z_score:.2f})",
                            triggered_by="anomaly_detection",
                            current_value=float(values[-1]),
                            metadata={"z_score": float(max_z_score), "mean": float(mean_val), "std": float(std_val)}
                        )
                        
                        self._trigger_alert(alert)
    
    def _generate_alert_id(self, rule_name: str) -> str:
        """Generate unique alert ID."""
        timestamp = int(time.time())
        hash_input = f"{rule_name}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert and notify handlers."""
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        self.logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.message}")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def _resolve_alert(self, alert_identifier: str):
        """Resolve an active alert."""
        alert = self.active_alerts.get(alert_identifier)
        if alert:
            alert.resolved = True
            alert.resolution_timestamp = datetime.now()
            del self.active_alerts[alert_identifier]
            
            self.logger.info(f"ALERT RESOLVED: {alert.name}")
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        scope: Optional[MonitoringScope] = None
    ) -> List[Alert]:
        """Get currently active alerts with optional filtering."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if scope:
            alerts = [a for a in alerts if a.scope == scope]
        
        return alerts


class HealthMonitor:
    """
    System health monitoring with component status tracking,
    dependency checking, and overall health assessment.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alerting_system: AlertingSystem,
        check_interval: float = 30.0
    ):
        self.metrics_collector = metrics_collector
        self.alerting_system = alerting_system
        self.check_interval = check_interval
        
        # Component registry
        self.components = {}
        self.dependencies = {}
        self.health_checks = {}
        
        # System state
        self.system_start_time = datetime.now()
        self.last_health_check = None
        
        # Background monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        
        self.logger = logging.getLogger(__name__)
    
    def register_component(
        self,
        component_name: str,
        health_check: Callable[[], bool],
        dependencies: Optional[List[str]] = None,
        critical: bool = False
    ):
        """Register a component for health monitoring."""
        self.components[component_name] = {
            "health_check": health_check,
            "dependencies": dependencies or [],
            "critical": critical,
            "status": "unknown",
            "last_check": None,
            "consecutive_failures": 0
        }
        
        if dependencies:
            self.dependencies[component_name] = dependencies
        
        self.logger.info(f"Registered component for monitoring: {component_name}")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self._monitoring_active = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(min(self.check_interval, 30.0))  # Back off on errors
    
    def _perform_health_checks(self):
        """Perform health checks on all registered components."""
        self.last_health_check = datetime.now()
        
        for component_name, component_info in self.components.items():
            try:
                # Check dependencies first
                deps_healthy = self._check_dependencies(component_name)
                
                if deps_healthy:
                    # Run health check
                    health_check = component_info["health_check"]
                    is_healthy = health_check()
                    
                    if is_healthy:
                        component_info["status"] = "healthy"
                        component_info["consecutive_failures"] = 0
                    else:
                        component_info["status"] = "unhealthy"
                        component_info["consecutive_failures"] += 1
                else:
                    component_info["status"] = "dependency_failure"
                    component_info["consecutive_failures"] += 1
                
                component_info["last_check"] = self.last_health_check
                
                # Record metrics
                self.metrics_collector.record_metric(
                    f"component_health_{component_name}",
                    1.0 if component_info["status"] == "healthy" else 0.0,
                    MetricType.GAUGE,
                    MonitoringScope.SYSTEM
                )
                
            except Exception as e:
                self.logger.error(f"Health check failed for {component_name}: {e}")
                component_info["status"] = "check_failed"
                component_info["consecutive_failures"] += 1
    
    def _check_dependencies(self, component_name: str) -> bool:
        """Check if all dependencies of a component are healthy."""
        dependencies = self.dependencies.get(component_name, [])
        
        for dep in dependencies:
            if dep not in self.components:
                self.logger.warning(f"Unknown dependency {dep} for component {component_name}")
                return False
            
            if self.components[dep]["status"] != "healthy":
                return False
        
        return True
    
    def get_health_status(self) -> HealthStatus:
        """Get current system health status."""
        # Calculate overall status
        critical_components = [
            name for name, info in self.components.items()
            if info["critical"]
        ]
        
        critical_unhealthy = [
            name for name in critical_components
            if self.components[name]["status"] != "healthy"
        ]
        
        unhealthy_components = [
            name for name, info in self.components.items()
            if info["status"] != "healthy"
        ]
        
        # Determine overall status
        if critical_unhealthy:
            overall_status = "CRITICAL"
        elif len(unhealthy_components) > len(self.components) * 0.5:
            overall_status = "UNHEALTHY"
        elif unhealthy_components:
            overall_status = "DEGRADED"
        else:
            overall_status = "HEALTHY"
        
        # Component statuses
        component_statuses = {
            name: info["status"] for name, info in self.components.items()
        }
        
        # Active alerts
        active_alerts = self.alerting_system.get_active_alerts()
        
        # Performance metrics
        uptime = (datetime.now() - self.system_start_time).total_seconds()
        
        performance_metrics = {
            "uptime_seconds": uptime,
            "component_count": len(self.components),
            "healthy_components": len(self.components) - len(unhealthy_components),
            "active_alert_count": len(active_alerts),
            "critical_alert_count": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL])
        }
        
        return HealthStatus(
            overall_status=overall_status,
            component_statuses=component_statuses,
            active_alerts=active_alerts,
            uptime_seconds=uptime,
            performance_metrics=performance_metrics
        )


# Global instances for easy access
_global_metrics_collector = None
_global_alerting_system = None
_global_health_monitor = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def get_alerting_system() -> AlertingSystem:
    """Get global alerting system instance."""
    global _global_alerting_system, _global_metrics_collector
    if _global_alerting_system is None:
        if _global_metrics_collector is None:
            _global_metrics_collector = MetricsCollector()
        _global_alerting_system = AlertingSystem(_global_metrics_collector)
    return _global_alerting_system


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _global_health_monitor, _global_metrics_collector, _global_alerting_system
    if _global_health_monitor is None:
        if _global_metrics_collector is None:
            _global_metrics_collector = MetricsCollector()
        if _global_alerting_system is None:
            _global_alerting_system = AlertingSystem(_global_metrics_collector)
        _global_health_monitor = HealthMonitor(_global_metrics_collector, _global_alerting_system)
    return _global_health_monitor


# Convenience functions
def record_metric(name: str, value: Union[float, int, str], **kwargs):
    """Convenience function to record a metric."""
    get_metrics_collector().record_metric(name, value, **kwargs)


def add_alert_rule(rule_name: str, **kwargs):
    """Convenience function to add an alert rule."""
    get_alerting_system().add_alert_rule(rule_name, **kwargs)


def register_component(component_name: str, health_check: Callable[[], bool], **kwargs):
    """Convenience function to register a component for health monitoring."""
    get_health_monitor().register_component(component_name, health_check, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("Advanced Monitoring Framework Loaded")
    print("Monitoring capabilities:")
    print("- High-performance metrics collection with buffering")
    print("- Intelligent alerting with threshold rules and anomaly detection")
    print("- System health monitoring with component tracking")
    print("- Real-time observability and diagnostics")
    print("- Thread-safe operation for production environments")

# Global monitoring instance
global_monitor = get_health_monitor()

def start_monitoring():
    """Start global monitoring."""
    global_monitor.start_monitoring()

def stop_monitoring():
    """Stop global monitoring."""
    global_monitor.stop_monitoring()

def get_system_health():
    """Get system health."""
    return global_monitor.get_health_status()

def record_clinical_operation(**kwargs):
    """Record clinical operation."""
    pass  # Placeholder implementation
