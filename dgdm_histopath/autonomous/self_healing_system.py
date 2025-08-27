"""
Self-Healing System with Predictive Maintenance

Features:
- Automatic error detection and recovery
- Predictive failure analysis
- Dynamic resource adjustment
- Circuit breaker patterns
- Health monitoring and alerting
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque

from ..utils.logging import get_logger
from ..utils.monitoring import MetricsCollector

logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"


class RecoveryAction(Enum):
    """Types of recovery actions"""
    RESTART_SERVICE = "restart_service"
    SCALE_RESOURCES = "scale_resources"
    CLEAR_CACHE = "clear_cache"
    RESET_CONNECTIONS = "reset_connections"
    FAILOVER = "failover"
    CIRCUIT_BREAKER = "circuit_breaker"
    THROTTLE_REQUESTS = "throttle_requests"


@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    threshold: float
    status: HealthStatus
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class IncidentRecord:
    """Record of system incident and recovery"""
    incident_id: str
    start_time: datetime
    end_time: Optional[datetime]
    severity: HealthStatus
    root_cause: str
    recovery_actions: List[RecoveryAction]
    success: bool
    impact_duration: float
    metadata: Dict[str, Any] = None


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, 
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker"""
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class SelfHealingSystem:
    """Self-healing system with predictive maintenance"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.metrics_collector = MetricsCollector()
        
        # System state
        self.health_metrics: Dict[str, HealthMetric] = {}
        self.incident_history: List[IncidentRecord] = []
        self.recovery_strategies: Dict[str, List[Callable]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        self.alert_handlers: List[Callable] = []
        
        # Predictive models
        self._initialize_predictive_models()
        
        # Start background monitoring
        self._start_monitoring()
        
        logger.info("Self-healing system initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with intelligent defaults"""
        default_config = {
            "monitoring_interval": 30,
            "health_check_timeout": 10,
            "max_recovery_attempts": 3,
            "incident_retention_days": 30,
            "predictive_window_minutes": 60,
            "auto_recovery_enabled": True,
            "circuit_breaker_defaults": {
                "failure_threshold": 5,
                "recovery_timeout": 60
            },
            "thresholds": {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "error_rate": 5.0,
                "response_time": 2000.0
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def _initialize_predictive_models(self):
        """Initialize models for predictive maintenance"""
        try:
            # Simple anomaly detection using statistical methods
            self.baseline_metrics = defaultdict(deque)
            self.anomaly_thresholds = {}
            
            # Pattern recognition for failure prediction
            self.failure_patterns = {}
            
        except Exception as e:
            logger.error(f"Failed to initialize predictive models: {e}")
    
    def _start_monitoring(self):
        """Start background monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started background health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._perform_health_check()
                await self._analyze_trends()
                await self._predictive_maintenance()
                
                await asyncio.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            # System metrics
            system_metrics = await self._collect_system_metrics()
            
            # Application metrics
            app_metrics = await self._collect_application_metrics()
            
            # Update health status
            all_metrics = {**system_metrics, **app_metrics}
            for name, value in all_metrics.items():
                await self._update_health_metric(name, value)
            
            # Check for issues
            issues = self._detect_health_issues()
            if issues:
                await self._handle_health_issues(issues)
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        metrics = {}
        
        try:
            import psutil
            
            # CPU usage
            metrics["cpu_usage"] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics["memory_usage"] = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics["disk_usage"] = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            metrics["network_bytes_sent"] = network.bytes_sent
            metrics["network_bytes_recv"] = network.bytes_recv
            
        except ImportError:
            logger.warning("psutil not available for system metrics")
            # Fallback to basic metrics
            metrics = {
                "cpu_usage": 50.0,  # Placeholder
                "memory_usage": 60.0,
                "disk_usage": 30.0
            }
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            
        return metrics
    
    async def _collect_application_metrics(self) -> Dict[str, float]:
        """Collect application-specific metrics"""
        metrics = {}
        
        try:
            # Error rate (would integrate with actual error tracking)
            metrics["error_rate"] = await self._calculate_error_rate()
            
            # Response time
            metrics["avg_response_time"] = await self._calculate_avg_response_time()
            
            # Throughput
            metrics["requests_per_second"] = await self._calculate_throughput()
            
            # Queue depths
            metrics["task_queue_depth"] = await self._get_queue_depth()
            
        except Exception as e:
            logger.error(f"Application metrics collection failed: {e}")
            
        return metrics
    
    async def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        # Placeholder - would integrate with actual logging/monitoring
        return 2.5
    
    async def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        # Placeholder - would integrate with actual request tracking
        return 450.0
    
    async def _calculate_throughput(self) -> float:
        """Calculate current throughput"""
        # Placeholder - would integrate with actual request counting
        return 125.0
    
    async def _get_queue_depth(self) -> float:
        """Get current task queue depth"""
        # Placeholder - would integrate with actual queue monitoring
        return 15.0
    
    async def _update_health_metric(self, name: str, value: float):
        """Update health metric and determine status"""
        threshold = self.config["thresholds"].get(name, 100.0)
        
        # Determine health status
        if value > threshold * 1.2:
            status = HealthStatus.CRITICAL
        elif value > threshold:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        # Create metric record
        metric = HealthMetric(
            name=name,
            value=value,
            threshold=threshold,
            status=status,
            timestamp=datetime.now()
        )
        
        self.health_metrics[name] = metric
        
        # Update baselines for anomaly detection
        self.baseline_metrics[name].append(value)
        if len(self.baseline_metrics[name]) > 100:
            self.baseline_metrics[name].popleft()
    
    def _detect_health_issues(self) -> List[Dict[str, Any]]:
        """Detect current health issues"""
        issues = []
        
        for name, metric in self.health_metrics.items():
            if metric.status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
                issues.append({
                    "type": "threshold_exceeded",
                    "metric": name,
                    "value": metric.value,
                    "threshold": metric.threshold,
                    "severity": metric.status,
                    "timestamp": metric.timestamp
                })
        
        # Check for anomalies
        anomalies = self._detect_anomalies()
        issues.extend(anomalies)
        
        return issues
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in metrics"""
        anomalies = []
        
        for name, values in self.baseline_metrics.items():
            if len(values) < 10:  # Need sufficient history
                continue
            
            try:
                import numpy as np
                
                values_array = np.array(values)
                mean = np.mean(values_array[:-1])  # Exclude current value
                std = np.std(values_array[:-1])
                current = values_array[-1]
                
                # Check if current value is anomalous (2 standard deviations)
                if abs(current - mean) > 2 * std:
                    anomalies.append({
                        "type": "statistical_anomaly",
                        "metric": name,
                        "value": current,
                        "expected_range": [mean - 2*std, mean + 2*std],
                        "severity": HealthStatus.DEGRADED,
                        "timestamp": datetime.now()
                    })
                    
            except ImportError:
                # Fallback without numpy
                continue
            except Exception as e:
                logger.error(f"Anomaly detection failed for {name}: {e}")
        
        return anomalies
    
    async def _handle_health_issues(self, issues: List[Dict[str, Any]]):
        """Handle detected health issues"""
        for issue in issues:
            await self._trigger_recovery(issue)
    
    async def _trigger_recovery(self, issue: Dict[str, Any]):
        """Trigger appropriate recovery actions"""
        try:
            # Determine recovery strategy
            recovery_actions = self._determine_recovery_actions(issue)
            
            # Create incident record
            incident = IncidentRecord(
                incident_id=f"incident_{int(time.time())}",
                start_time=datetime.now(),
                end_time=None,
                severity=issue["severity"],
                root_cause=f"{issue['type']}: {issue['metric']}",
                recovery_actions=recovery_actions,
                success=False,
                impact_duration=0.0
            )
            
            logger.warning(f"Triggering recovery for issue: {issue}")
            
            # Execute recovery actions
            success = await self._execute_recovery_actions(recovery_actions, issue)
            
            # Update incident record
            incident.end_time = datetime.now()
            incident.success = success
            incident.impact_duration = (incident.end_time - incident.start_time).total_seconds()
            
            self.incident_history.append(incident)
            
            # Send alerts
            await self._send_alerts(incident)
            
        except Exception as e:
            logger.error(f"Recovery trigger failed: {e}")
    
    def _determine_recovery_actions(self, issue: Dict[str, Any]) -> List[RecoveryAction]:
        """Determine appropriate recovery actions for issue"""
        actions = []
        metric = issue["metric"]
        severity = issue["severity"]
        
        # High CPU usage
        if "cpu" in metric:
            if severity == HealthStatus.CRITICAL:
                actions.extend([
                    RecoveryAction.THROTTLE_REQUESTS,
                    RecoveryAction.SCALE_RESOURCES
                ])
            else:
                actions.append(RecoveryAction.THROTTLE_REQUESTS)
        
        # High memory usage
        elif "memory" in metric:
            if severity == HealthStatus.CRITICAL:
                actions.extend([
                    RecoveryAction.CLEAR_CACHE,
                    RecoveryAction.RESTART_SERVICE
                ])
            else:
                actions.append(RecoveryAction.CLEAR_CACHE)
        
        # High error rate
        elif "error" in metric:
            actions.extend([
                RecoveryAction.CIRCUIT_BREAKER,
                RecoveryAction.RESET_CONNECTIONS
            ])
        
        # High response time
        elif "response_time" in metric:
            actions.extend([
                RecoveryAction.SCALE_RESOURCES,
                RecoveryAction.CLEAR_CACHE
            ])
        
        # Default actions
        if not actions:
            actions.append(RecoveryAction.RESTART_SERVICE)
        
        return actions
    
    async def _execute_recovery_actions(self, actions: List[RecoveryAction], 
                                      issue: Dict[str, Any]) -> bool:
        """Execute recovery actions"""
        success = True
        
        for action in actions:
            try:
                action_success = await self._execute_single_action(action, issue)
                if not action_success:
                    success = False
                    logger.error(f"Recovery action failed: {action}")
                else:
                    logger.info(f"Recovery action succeeded: {action}")
                    
                # Wait between actions
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Recovery action {action} failed: {e}")
                success = False
        
        return success
    
    async def _execute_single_action(self, action: RecoveryAction, 
                                   issue: Dict[str, Any]) -> bool:
        """Execute a single recovery action"""
        try:
            if action == RecoveryAction.RESTART_SERVICE:
                return await self._restart_service(issue)
            
            elif action == RecoveryAction.SCALE_RESOURCES:
                return await self._scale_resources(issue)
            
            elif action == RecoveryAction.CLEAR_CACHE:
                return await self._clear_cache(issue)
            
            elif action == RecoveryAction.RESET_CONNECTIONS:
                return await self._reset_connections(issue)
            
            elif action == RecoveryAction.FAILOVER:
                return await self._perform_failover(issue)
            
            elif action == RecoveryAction.CIRCUIT_BREAKER:
                return await self._activate_circuit_breaker(issue)
            
            elif action == RecoveryAction.THROTTLE_REQUESTS:
                return await self._throttle_requests(issue)
            
            else:
                logger.warning(f"Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False
    
    async def _restart_service(self, issue: Dict[str, Any]) -> bool:
        """Restart affected service"""
        logger.info("Simulating service restart")
        await asyncio.sleep(5)  # Simulate restart time
        return True
    
    async def _scale_resources(self, issue: Dict[str, Any]) -> bool:
        """Scale system resources"""
        logger.info("Simulating resource scaling")
        await asyncio.sleep(10)  # Simulate scaling time
        return True
    
    async def _clear_cache(self, issue: Dict[str, Any]) -> bool:
        """Clear system caches"""
        logger.info("Simulating cache clear")
        await asyncio.sleep(2)
        return True
    
    async def _reset_connections(self, issue: Dict[str, Any]) -> bool:
        """Reset network connections"""
        logger.info("Simulating connection reset")
        await asyncio.sleep(3)
        return True
    
    async def _perform_failover(self, issue: Dict[str, Any]) -> bool:
        """Perform service failover"""
        logger.info("Simulating failover")
        await asyncio.sleep(15)
        return True
    
    async def _activate_circuit_breaker(self, issue: Dict[str, Any]) -> bool:
        """Activate circuit breaker for failing component"""
        component = issue.get("metric", "unknown")
        
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
        
        # Circuit breaker is now active for this component
        logger.info(f"Circuit breaker activated for {component}")
        return True
    
    async def _throttle_requests(self, issue: Dict[str, Any]) -> bool:
        """Throttle incoming requests"""
        logger.info("Simulating request throttling")
        await asyncio.sleep(1)
        return True
    
    async def _analyze_trends(self):
        """Analyze health metric trends"""
        try:
            trends = self._calculate_trends()
            degrading_metrics = [
                name for name, trend in trends.items() 
                if trend["slope"] > 0.1 and trend["direction"] == "increasing"
            ]
            
            if degrading_metrics:
                logger.warning(f"Degrading trend detected in: {degrading_metrics}")
                
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
    
    def _calculate_trends(self) -> Dict[str, Dict[str, Any]]:
        """Calculate trends for metrics"""
        trends = {}
        
        for name, values in self.baseline_metrics.items():
            if len(values) < 5:
                continue
            
            try:
                import numpy as np
                
                # Simple linear regression
                x = np.arange(len(values))
                y = np.array(values)
                slope = np.polyfit(x, y, 1)[0]
                
                trends[name] = {
                    "slope": slope,
                    "direction": "increasing" if slope > 0 else "decreasing",
                    "magnitude": abs(slope)
                }
                
            except ImportError:
                continue
            except Exception as e:
                logger.error(f"Trend calculation failed for {name}: {e}")
        
        return trends
    
    async def _predictive_maintenance(self):
        """Perform predictive maintenance analysis"""
        try:
            predictions = await self._predict_failures()
            
            for prediction in predictions:
                if prediction["probability"] > 0.7:
                    logger.warning(f"Predicted failure: {prediction}")
                    await self._preemptive_action(prediction)
                    
        except Exception as e:
            logger.error(f"Predictive maintenance failed: {e}")
    
    async def _predict_failures(self) -> List[Dict[str, Any]]:
        """Predict potential system failures"""
        predictions = []
        
        # Analyze patterns in incident history
        if len(self.incident_history) > 5:
            # Look for recurring patterns
            failure_patterns = self._analyze_failure_patterns()
            
            for pattern in failure_patterns:
                if pattern["likelihood"] > 0.6:
                    predictions.append({
                        "type": "pattern_based",
                        "component": pattern["component"],
                        "probability": pattern["likelihood"],
                        "expected_time": pattern["expected_time"],
                        "confidence": 0.7
                    })
        
        # Analyze metric trajectories
        trends = self._calculate_trends()
        for name, trend in trends.items():
            if trend["slope"] > 0.2 and trend["direction"] == "increasing":
                # Metric trending towards failure
                threshold = self.config["thresholds"].get(name, 100.0)
                current_value = list(self.baseline_metrics[name])[-1] if self.baseline_metrics[name] else 0
                
                if current_value > threshold * 0.8:
                    time_to_failure = self._estimate_time_to_failure(name, trend, threshold)
                    
                    predictions.append({
                        "type": "trajectory_based",
                        "component": name,
                        "probability": min((current_value / threshold) * 0.8, 1.0),
                        "expected_time": time_to_failure,
                        "confidence": 0.6
                    })
        
        return predictions
    
    def _analyze_failure_patterns(self) -> List[Dict[str, Any]]:
        """Analyze historical failure patterns"""
        patterns = []
        
        # Group incidents by root cause
        cause_groups = defaultdict(list)
        for incident in self.incident_history[-50:]:  # Last 50 incidents
            cause_groups[incident.root_cause].append(incident)
        
        for cause, incidents in cause_groups.items():
            if len(incidents) >= 3:  # Pattern threshold
                # Calculate time intervals
                intervals = []
                for i in range(1, len(incidents)):
                    interval = (incidents[i].start_time - incidents[i-1].start_time).total_seconds()
                    intervals.append(interval)
                
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    last_incident = incidents[-1].start_time
                    time_since_last = (datetime.now() - last_incident).total_seconds()
                    
                    likelihood = max(0, min(1, time_since_last / avg_interval))
                    
                    patterns.append({
                        "component": cause.split(":")[1] if ":" in cause else cause,
                        "likelihood": likelihood,
                        "expected_time": max(0, avg_interval - time_since_last),
                        "frequency": len(incidents),
                        "avg_interval": avg_interval
                    })
        
        return patterns
    
    def _estimate_time_to_failure(self, metric_name: str, trend: Dict[str, Any], 
                                 threshold: float) -> float:
        """Estimate time until metric reaches failure threshold"""
        try:
            if trend["slope"] <= 0:
                return float('inf')  # Not trending towards failure
            
            current_value = list(self.baseline_metrics[metric_name])[-1]
            remaining_headroom = threshold - current_value
            
            if remaining_headroom <= 0:
                return 0  # Already at threshold
            
            # Estimate time based on current rate of change
            time_to_failure = remaining_headroom / trend["slope"]
            return max(0, time_to_failure * self.config["monitoring_interval"])
            
        except Exception as e:
            logger.error(f"Time to failure estimation failed: {e}")
            return float('inf')
    
    async def _preemptive_action(self, prediction: Dict[str, Any]):
        """Take preemptive action based on failure prediction"""
        component = prediction["component"]
        probability = prediction["probability"]
        
        logger.info(f"Taking preemptive action for {component} (risk: {probability:.2%})")
        
        # Preemptive scaling for capacity issues
        if "cpu" in component or "memory" in component:
            await self._preemptive_scaling(component)
        
        # Preemptive connection refresh
        elif "connection" in component or "network" in component:
            await self._preemptive_connection_refresh(component)
        
        # Preemptive cache warming
        elif "cache" in component:
            await self._preemptive_cache_warming(component)
    
    async def _preemptive_scaling(self, component: str):
        """Preemptively scale resources"""
        logger.info(f"Preemptive scaling for {component}")
        await asyncio.sleep(5)  # Simulate scaling
    
    async def _preemptive_connection_refresh(self, component: str):
        """Preemptively refresh connections"""
        logger.info(f"Preemptive connection refresh for {component}")
        await asyncio.sleep(2)  # Simulate refresh
    
    async def _preemptive_cache_warming(self, component: str):
        """Preemptively warm caches"""
        logger.info(f"Preemptive cache warming for {component}")
        await asyncio.sleep(3)  # Simulate warming
    
    async def _send_alerts(self, incident: IncidentRecord):
        """Send alerts for incidents"""
        try:
            alert_data = {
                "incident_id": incident.incident_id,
                "severity": incident.severity.value,
                "root_cause": incident.root_cause,
                "recovery_success": incident.success,
                "impact_duration": incident.impact_duration,
                "timestamp": incident.start_time.isoformat()
            }
            
            for handler in self.alert_handlers:
                await handler(alert_data)
                
            logger.info(f"Alerts sent for incident {incident.incident_id}")
            
        except Exception as e:
            logger.error(f"Alert sending failed: {e}")
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        overall_status = HealthStatus.HEALTHY
        issues = []
        
        for name, metric in self.health_metrics.items():
            if metric.status in [HealthStatus.CRITICAL, HealthStatus.FAILING]:
                overall_status = HealthStatus.CRITICAL
                issues.append(f"{name}: {metric.value:.2f} (threshold: {metric.threshold})")
            elif metric.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
                issues.append(f"{name}: {metric.value:.2f} (threshold: {metric.threshold})")
        
        return {
            "status": overall_status.value,
            "issues": issues,
            "metrics": {name: asdict(metric) for name, metric in self.health_metrics.items()},
            "last_check": datetime.now().isoformat(),
            "incident_count_24h": len([
                inc for inc in self.incident_history 
                if inc.start_time > datetime.now() - timedelta(hours=24)
            ])
        }
    
    async def shutdown(self):
        """Gracefully shutdown the self-healing system"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Self-healing system shutdown complete")