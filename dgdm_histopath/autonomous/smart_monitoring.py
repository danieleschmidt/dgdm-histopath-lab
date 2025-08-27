"""
Smart Monitoring and Observability Framework

Features:
- Intelligent metric collection and analysis
- Adaptive alerting with ML-based noise reduction
- Distributed tracing and correlation analysis
- Performance anomaly detection
- Real-time dashboards with predictive insights
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import threading
import statistics

from ..utils.logging import get_logger
from ..utils.monitoring import MetricsCollector

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    condition: str  # e.g., "> 80", "< 0.1"
    threshold: float
    severity: AlertSeverity
    window_minutes: int = 5
    min_occurrences: int = 3
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Distributed tracing span"""
    span_id: str
    trace_id: str
    parent_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


class SmartMonitoring:
    """Intelligent monitoring and observability system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Metric storage
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        
        # Tracing system
        self.active_traces: Dict[str, List[TraceSpan]] = {}
        self.trace_handlers: List[Callable] = []
        
        # Anomaly detection
        self.baseline_models: Dict[str, Dict[str, Any]] = {}
        self.anomaly_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Background tasks
        self.monitoring_active = False
        self.processing_task = None
        self.cleanup_task = None
        
        # Initialize components
        self._initialize_alert_rules()
        self._initialize_anomaly_detection()
        
        # Start background processing
        self._start_background_tasks()
        
        logger.info("Smart monitoring system initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with intelligent defaults"""
        default_config = {
            "metric_retention_hours": 24,
            "alert_evaluation_interval": 30,
            "anomaly_detection_enabled": True,
            "trace_retention_hours": 6,
            "cleanup_interval_minutes": 30,
            "alert_noise_reduction": True,
            "adaptive_thresholds": True,
            "correlation_analysis": True,
            "predictive_alerting": True
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load monitoring config: {e}")
        
        return default_config
    
    def _initialize_alert_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric="cpu_usage",
                condition="> 80",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                window_minutes=5,
                min_occurrences=3
            ),
            AlertRule(
                name="critical_cpu_usage",
                metric="cpu_usage", 
                condition="> 95",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                window_minutes=2,
                min_occurrences=2
            ),
            AlertRule(
                name="high_memory_usage",
                metric="memory_usage",
                condition="> 85",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                window_minutes=5,
                min_occurrences=3
            ),
            AlertRule(
                name="high_error_rate",
                metric="error_rate",
                condition="> 5",
                threshold=5.0,
                severity=AlertSeverity.ERROR,
                window_minutes=3,
                min_occurrences=2
            ),
            AlertRule(
                name="slow_response_time",
                metric="avg_response_time",
                condition="> 2000",
                threshold=2000.0,
                severity=AlertSeverity.WARNING,
                window_minutes=5,
                min_occurrences=3
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
    
    def _initialize_anomaly_detection(self):
        """Initialize anomaly detection models"""
        if not self.config["anomaly_detection_enabled"]:
            return
        
        try:
            # Simple statistical models for anomaly detection
            self.anomaly_detectors = {
                "statistical": self._statistical_anomaly_detector,
                "seasonal": self._seasonal_anomaly_detector,
                "trend": self._trend_anomaly_detector
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detection: {e}")
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.processing_task = asyncio.create_task(self._processing_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started monitoring background tasks")
    
    async def _processing_loop(self):
        """Main processing loop for alerts and analysis"""
        while self.monitoring_active:
            try:
                await self._evaluate_alert_rules()
                await self._detect_anomalies()
                await self._analyze_correlations()
                await self._update_predictive_models()
                
                await asyncio.sleep(self.config["alert_evaluation_interval"])
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """Background cleanup of old data"""
        while self.monitoring_active:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(self.config["cleanup_interval_minutes"] * 60)
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric data point"""
        try:
            point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metric_type=metric_type
            )
            
            # Store in buffer
            self.metrics_buffer[name].append(point)
            
            # Update metadata
            if name not in self.metric_metadata:
                self.metric_metadata[name] = {
                    "first_seen": datetime.now(),
                    "metric_type": metric_type,
                    "sample_count": 0,
                    "min_value": value,
                    "max_value": value,
                    "avg_value": value
                }
            
            metadata = self.metric_metadata[name]
            metadata["sample_count"] += 1
            metadata["min_value"] = min(metadata["min_value"], value)
            metadata["max_value"] = max(metadata["max_value"], value)
            metadata["last_seen"] = datetime.now()
            
            # Update running average (simple exponential moving average)
            alpha = 0.1
            metadata["avg_value"] = alpha * value + (1 - alpha) * metadata["avg_value"]
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def start_trace(self, operation_name: str, parent_span_id: Optional[str] = None,
                   tags: Optional[Dict[str, Any]] = None) -> str:
        """Start a distributed trace span"""
        try:
            trace_id = str(uuid.uuid4()) if parent_span_id is None else self._get_trace_id(parent_span_id)
            span_id = str(uuid.uuid4())
            
            span = TraceSpan(
                span_id=span_id,
                trace_id=trace_id,
                parent_id=parent_span_id,
                operation_name=operation_name,
                start_time=datetime.now(),
                end_time=None,
                duration_ms=None,
                tags=tags or {}
            )
            
            if trace_id not in self.active_traces:
                self.active_traces[trace_id] = []
            
            self.active_traces[trace_id].append(span)
            
            return span_id
            
        except Exception as e:
            logger.error(f"Failed to start trace: {e}")
            return ""
    
    def finish_trace(self, span_id: str, tags: Optional[Dict[str, Any]] = None):
        """Finish a trace span"""
        try:
            for trace_id, spans in self.active_traces.items():
                for span in spans:
                    if span.span_id == span_id:
                        span.end_time = datetime.now()
                        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
                        
                        if tags:
                            span.tags.update(tags)
                        
                        # Record as metric
                        self.record_metric(
                            f"trace.{span.operation_name}.duration",
                            span.duration_ms,
                            MetricType.TIMER,
                            span.tags
                        )
                        
                        # Send to trace handlers
                        for handler in self.trace_handlers:
                            asyncio.create_task(handler(span))
                        
                        return
            
            logger.warning(f"Span {span_id} not found for finishing")
            
        except Exception as e:
            logger.error(f"Failed to finish trace: {e}")
    
    def add_trace_log(self, span_id: str, message: str, level: str = "info",
                     fields: Optional[Dict[str, Any]] = None):
        """Add log entry to trace span"""
        try:
            for spans in self.active_traces.values():
                for span in spans:
                    if span.span_id == span_id:
                        span.logs.append({
                            "timestamp": datetime.now(),
                            "level": level,
                            "message": message,
                            "fields": fields or {}
                        })
                        return
                        
        except Exception as e:
            logger.error(f"Failed to add trace log: {e}")
    
    def _get_trace_id(self, span_id: str) -> str:
        """Get trace ID for a given span ID"""
        for trace_id, spans in self.active_traces.items():
            for span in spans:
                if span.span_id == span_id:
                    return trace_id
        return str(uuid.uuid4())  # Fallback to new trace
    
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
                
            try:
                await self._evaluate_single_rule(rule)
            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule_name}: {e}")
    
    async def _evaluate_single_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        if rule.metric not in self.metrics_buffer:
            return
        
        # Get recent data points within window
        cutoff_time = datetime.now() - timedelta(minutes=rule.window_minutes)
        recent_points = [
            point for point in self.metrics_buffer[rule.metric]
            if point.timestamp >= cutoff_time
        ]
        
        if len(recent_points) < rule.min_occurrences:
            return
        
        # Evaluate condition
        violation_count = 0
        current_value = None
        
        for point in recent_points:
            if self._evaluate_condition(point.value, rule.condition, rule.threshold):
                violation_count += 1
                current_value = point.value
        
        # Check if alert should be triggered
        if violation_count >= rule.min_occurrences:
            await self._trigger_alert(rule, current_value or recent_points[-1].value)
        else:
            # Check if existing alert should be resolved
            await self._check_alert_resolution(rule)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        try:
            if condition.startswith(">"):
                return value > threshold
            elif condition.startswith("<"):
                return value < threshold
            elif condition.startswith("=="):
                return abs(value - threshold) < 0.001
            elif condition.startswith("!="):
                return abs(value - threshold) >= 0.001
            else:
                return False
        except Exception:
            return False
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert"""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        # Check if similar alert is already active (noise reduction)
        if self.config["alert_noise_reduction"]:
            for existing_alert in self.active_alerts.values():
                if (existing_alert.rule_name == rule.name and 
                    not existing_alert.resolved and
                    (datetime.now() - existing_alert.timestamp).total_seconds() < 300):  # 5 minutes
                    return  # Suppress duplicate alert
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=self._generate_alert_message(rule, current_value),
            metric_name=rule.metric,
            current_value=current_value,
            threshold=rule.threshold,
            timestamp=datetime.now(),
            tags=rule.tags
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send to alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"Alert triggered: {alert.message}")
    
    def _generate_alert_message(self, rule: AlertRule, current_value: float) -> str:
        """Generate human-readable alert message"""
        return (
            f"{rule.metric} is {current_value:.2f} {rule.condition} {rule.threshold} "
            f"(severity: {rule.severity.value})"
        )
    
    async def _check_alert_resolution(self, rule: AlertRule):
        """Check if any active alerts for this rule should be resolved"""
        for alert in list(self.active_alerts.values()):
            if alert.rule_name == rule.name and not alert.resolved:
                # Check if condition no longer met
                recent_points = [
                    point for point in self.metrics_buffer[rule.metric]
                    if point.timestamp >= datetime.now() - timedelta(minutes=rule.window_minutes)
                ]
                
                violation_count = sum(
                    1 for point in recent_points
                    if self._evaluate_condition(point.value, rule.condition, rule.threshold)
                )
                
                if violation_count < rule.min_occurrences:
                    await self._resolve_alert(alert)
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an active alert"""
        alert.resolved = True
        alert.resolved_at = datetime.now()
        
        if alert.id in self.active_alerts:
            del self.active_alerts[alert.id]
        
        logger.info(f"Alert resolved: {alert.message}")
        
        # Notify handlers of resolution
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert resolution handler failed: {e}")
    
    async def _detect_anomalies(self):
        """Detect anomalies in metrics using ML techniques"""
        if not self.config["anomaly_detection_enabled"]:
            return
        
        for metric_name, points in self.metrics_buffer.items():
            if len(points) < 20:  # Need sufficient data
                continue
            
            try:
                # Extract recent values
                recent_values = [point.value for point in list(points)[-50:]]
                
                # Run different anomaly detectors
                for detector_name, detector_func in self.anomaly_detectors.items():
                    anomaly_score = detector_func(recent_values)
                    
                    if anomaly_score > 0.8:  # High anomaly score
                        await self._handle_anomaly(metric_name, anomaly_score, detector_name)
                        
            except Exception as e:
                logger.error(f"Anomaly detection failed for {metric_name}: {e}")
    
    def _statistical_anomaly_detector(self, values: List[float]) -> float:
        """Statistical anomaly detection using z-score"""
        if len(values) < 10:
            return 0.0
        
        try:
            mean = statistics.mean(values[:-1])  # Exclude current value
            stdev = statistics.stdev(values[:-1])
            
            if stdev == 0:
                return 0.0
            
            current_value = values[-1]
            z_score = abs(current_value - mean) / stdev
            
            # Convert z-score to 0-1 anomaly score
            return min(z_score / 3.0, 1.0)  # 3-sigma rule
            
        except Exception:
            return 0.0
    
    def _seasonal_anomaly_detector(self, values: List[float]) -> float:
        """Simple seasonal anomaly detection"""
        if len(values) < 20:
            return 0.0
        
        try:
            # Look for patterns in similar time periods
            # Simplified: compare current value to same position in previous cycles
            cycle_length = 24  # Assume daily cycle
            
            if len(values) < cycle_length * 2:
                return 0.0
            
            current_value = values[-1]
            position = len(values) % cycle_length
            
            # Find similar positions in previous cycles
            similar_values = []
            for i in range(position, len(values) - 1, cycle_length):
                if i < len(values) - cycle_length:
                    similar_values.append(values[i])
            
            if len(similar_values) < 2:
                return 0.0
            
            mean_similar = statistics.mean(similar_values)
            stdev_similar = statistics.stdev(similar_values) if len(similar_values) > 1 else 1.0
            
            if stdev_similar == 0:
                return 0.0
            
            z_score = abs(current_value - mean_similar) / stdev_similar
            return min(z_score / 2.5, 1.0)
            
        except Exception:
            return 0.0
    
    def _trend_anomaly_detector(self, values: List[float]) -> float:
        """Trend-based anomaly detection"""
        if len(values) < 15:
            return 0.0
        
        try:
            # Simple linear regression to detect trend breaks
            import numpy as np
            
            x = np.arange(len(values) - 5)
            y = np.array(values[:-5])
            
            # Fit trend to historical data
            coeffs = np.polyfit(x, y, 1)
            trend_line = np.polyval(coeffs, range(len(values)))
            
            # Compare recent values to trend
            recent_values = values[-5:]
            recent_trend = trend_line[-5:]
            
            # Calculate deviation from trend
            deviations = [abs(actual - predicted) for actual, predicted in zip(recent_values, recent_trend)]
            avg_deviation = sum(deviations) / len(deviations)
            
            # Normalize by historical variance
            historical_variance = np.var(y)
            if historical_variance == 0:
                return 0.0
            
            anomaly_score = avg_deviation / (historical_variance ** 0.5)
            return min(anomaly_score / 2.0, 1.0)
            
        except ImportError:
            # Fallback without numpy
            return self._simple_trend_detector(values)
        except Exception:
            return 0.0
    
    def _simple_trend_detector(self, values: List[float]) -> float:
        """Simple trend detection without numpy"""
        if len(values) < 10:
            return 0.0
        
        try:
            # Compare recent slope to historical slope
            recent_slope = (values[-1] - values[-5]) / 4
            historical_slope = (values[-10] - values[-20]) / 10
            
            slope_diff = abs(recent_slope - historical_slope)
            max_value = max(values)
            min_value = min(values)
            value_range = max_value - min_value
            
            if value_range == 0:
                return 0.0
            
            normalized_diff = slope_diff / value_range
            return min(normalized_diff * 2, 1.0)
            
        except Exception:
            return 0.0
    
    async def _handle_anomaly(self, metric_name: str, anomaly_score: float, detector_type: str):
        """Handle detected anomaly"""
        logger.warning(
            f"Anomaly detected in {metric_name}: score={anomaly_score:.3f}, "
            f"detector={detector_type}"
        )
        
        # Create anomaly alert if severe enough
        if anomaly_score > 0.9:
            anomaly_rule = AlertRule(
                name=f"anomaly_{metric_name}",
                metric=metric_name,
                condition=f"anomaly_score > {anomaly_score}",
                threshold=anomaly_score,
                severity=AlertSeverity.WARNING,
                window_minutes=1,
                min_occurrences=1
            )
            
            current_value = list(self.metrics_buffer[metric_name])[-1].value
            await self._trigger_alert(anomaly_rule, current_value)
    
    async def _analyze_correlations(self):
        """Analyze correlations between metrics"""
        if not self.config["correlation_analysis"]:
            return
        
        try:
            # Find metrics with sufficient data
            metrics_with_data = [
                name for name, points in self.metrics_buffer.items()
                if len(points) >= 20
            ]
            
            if len(metrics_with_data) < 2:
                return
            
            # Calculate correlations between metrics
            correlations = {}
            for i, metric1 in enumerate(metrics_with_data):
                for metric2 in metrics_with_data[i+1:]:
                    correlation = self._calculate_correlation(metric1, metric2)
                    if abs(correlation) > 0.7:  # Strong correlation
                        correlations[f"{metric1}-{metric2}"] = correlation
            
            # Log interesting correlations
            if correlations:
                logger.info(f"Strong correlations detected: {correlations}")
                
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
    
    def _calculate_correlation(self, metric1: str, metric2: str) -> float:
        """Calculate correlation between two metrics"""
        try:
            # Get synchronized data points (same timestamps)
            points1 = {point.timestamp: point.value for point in self.metrics_buffer[metric1]}
            points2 = {point.timestamp: point.value for point in self.metrics_buffer[metric2]}
            
            # Find common timestamps
            common_times = set(points1.keys()) & set(points2.keys())
            
            if len(common_times) < 10:
                return 0.0
            
            values1 = [points1[t] for t in sorted(common_times)]
            values2 = [points2[t] for t in sorted(common_times)]
            
            # Calculate Pearson correlation
            return self._pearson_correlation(values1, values2)
            
        except Exception:
            return 0.0
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_x2 = sum(xi * xi for xi in x)
            sum_y2 = sum(yi * yi for yi in y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception:
            return 0.0
    
    async def _update_predictive_models(self):
        """Update predictive models for forecasting"""
        if not self.config["predictive_alerting"]:
            return
        
        # Placeholder for predictive model updates
        # In production, this would update ML models for forecasting
        pass
    
    async def _cleanup_old_data(self):
        """Clean up old metrics and traces"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config["metric_retention_hours"])
            
            # Clean old metrics
            for metric_name, points in self.metrics_buffer.items():
                original_length = len(points)
                # Filter out old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
                
                if len(points) < original_length:
                    logger.debug(f"Cleaned {original_length - len(points)} old points from {metric_name}")
            
            # Clean old traces
            trace_cutoff = datetime.now() - timedelta(hours=self.config["trace_retention_hours"])
            traces_to_remove = []
            
            for trace_id, spans in self.active_traces.items():
                # Check if all spans are old
                if all(span.start_time < trace_cutoff for span in spans):
                    traces_to_remove.append(trace_id)
            
            for trace_id in traces_to_remove:
                del self.active_traces[trace_id]
            
            # Clean old alert history
            alert_cutoff = datetime.now() - timedelta(days=7)
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp >= alert_cutoff
            ]
            
            logger.debug("Completed data cleanup")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def add_trace_handler(self, handler: Callable[[TraceSpan], None]):
        """Add trace handler function"""
        self.trace_handlers.append(handler)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        self.alert_rules[rule.name] = rule
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        
        for metric_name, points in self.metrics_buffer.items():
            if not points:
                continue
            
            recent_points = list(points)[-10:]  # Last 10 points
            values = [point.value for point in recent_points]
            
            summary[metric_name] = {
                "current_value": values[-1] if values else 0,
                "min_value": min(values) if values else 0,
                "max_value": max(values) if values else 0,
                "avg_value": sum(values) / len(values) if values else 0,
                "sample_count": len(points),
                "last_updated": recent_points[-1].timestamp.isoformat() if recent_points else None
            }
        
        return summary
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [asdict(alert) for alert in self.active_alerts.values()]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff
        ]
        return [asdict(alert) for alert in recent_alerts]
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of active traces"""
        return {
            "active_traces": len(self.active_traces),
            "total_spans": sum(len(spans) for spans in self.active_traces.values()),
            "trace_ids": list(self.active_traces.keys())
        }
    
    async def shutdown(self):
        """Gracefully shutdown monitoring system"""
        self.monitoring_active = False
        
        # Cancel background tasks
        if self.processing_task:
            self.processing_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Wait for tasks to complete
        for task in [self.processing_task, self.cleanup_task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Smart monitoring system shutdown complete")