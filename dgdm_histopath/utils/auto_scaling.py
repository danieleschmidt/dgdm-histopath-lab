"""
Auto-Scaling System for DGDM Histopath Lab.

Intelligent auto-scaling with load balancing and resource optimization
for high-throughput medical AI processing at scale.
"""

import time
import threading
import queue
import statistics
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ScalingDirection(Enum):
    """Scaling direction indicators."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    STABLE = "stable"


class LoadMetric(Enum):
    """Load metrics for scaling decisions."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    QUEUE_SIZE = "queue_size"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    queue_size: int
    active_workers: int
    avg_response_time_ms: float
    throughput_per_minute: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'queue_size': self.queue_size,
            'active_workers': self.active_workers,
            'avg_response_time_ms': self.avg_response_time_ms,
            'throughput_per_minute': self.throughput_per_minute
        }


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    # Scale up thresholds
    cpu_scale_up_threshold: float = 80.0
    memory_scale_up_threshold: float = 85.0
    queue_size_scale_up_threshold: int = 100
    response_time_scale_up_threshold_ms: float = 1000.0
    
    # Scale down thresholds
    cpu_scale_down_threshold: float = 30.0
    memory_scale_down_threshold: float = 40.0
    queue_size_scale_down_threshold: int = 10
    response_time_scale_down_threshold_ms: float = 200.0
    
    # Scaling parameters
    min_workers: int = 1
    max_workers: int = 32
    scale_up_increment: int = 2
    scale_down_increment: int = 1
    
    # Timing parameters
    monitoring_interval_seconds: float = 30.0
    cooldown_period_seconds: float = 300.0  # 5 minutes
    evaluation_window_minutes: int = 5


class AutoScaler:
    """
    Intelligent auto-scaling system with multiple metrics and policies.
    """
    
    def __init__(
        self,
        initial_workers: int = 4,
        scaling_policy: Optional[ScalingPolicy] = None,
        enable_predictive_scaling: bool = True
    ):
        self.current_workers = initial_workers
        self.scaling_policy = scaling_policy or ScalingPolicy()
        self.enable_predictive_scaling = enable_predictive_scaling
        
        # Metrics history
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Scaling state
        self.last_scaling_action = datetime.now()
        self.scaling_active = False
        self.monitoring_thread = None
        
        # Callbacks for scaling actions
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        # Internal state
        self.stop_event = threading.Event()
        
        self.logger = logging.getLogger(__name__)

    def set_scaling_callbacks(
        self,
        scale_up_callback: Callable[[int], bool],
        scale_down_callback: Callable[[int], bool]
    ) -> None:
        """Set callbacks for scaling actions."""
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback

    def start_monitoring(self) -> None:
        """Start auto-scaling monitoring."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Auto-scaling monitoring started")

    def stop_monitoring(self) -> None:
        """Stop auto-scaling monitoring."""
        self.scaling_active = False
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10.0)
        
        self.logger.info("Auto-scaling monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.scaling_active and not self.stop_event.is_set():
            try:
                # Collect current metrics
                current_metrics = self._collect_metrics()
                
                if current_metrics:
                    # Store metrics
                    self.metrics_history.append(current_metrics)
                    
                    # Trim history
                    cutoff_time = datetime.now() - timedelta(hours=1)
                    self.metrics_history = [
                        m for m in self.metrics_history 
                        if m.timestamp > cutoff_time
                    ]
                    
                    # Evaluate scaling decision
                    scaling_decision = self._evaluate_scaling_decision()
                    
                    if scaling_decision != ScalingDirection.STABLE:
                        self._execute_scaling_decision(scaling_decision, current_metrics)
                
                # Wait for next monitoring interval
                self.stop_event.wait(self.scaling_policy.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaling monitoring loop: {e}")
                self.stop_event.wait(min(self.scaling_policy.monitoring_interval_seconds, 60.0))

    def _collect_metrics(self) -> Optional[ScalingMetrics]:
        """Collect current system metrics."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            
            # Placeholder for application-specific metrics
            # In a real implementation, these would come from the actual system
            queue_size = 0  # Would get from task queue
            active_workers = self.current_workers
            avg_response_time = 100.0  # Would calculate from actual responses
            throughput = 60.0  # Would calculate from actual throughput
            
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                queue_size=queue_size,
                active_workers=active_workers,
                avg_response_time_ms=avg_response_time,
                throughput_per_minute=throughput
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return None

    def _evaluate_scaling_decision(self) -> ScalingDirection:
        """Evaluate whether to scale up, down, or stay stable."""
        if len(self.metrics_history) < 3:
            return ScalingDirection.STABLE
        
        # Check cooldown period
        time_since_last_scaling = datetime.now() - self.last_scaling_action
        if time_since_last_scaling.total_seconds() < self.scaling_policy.cooldown_period_seconds:
            return ScalingDirection.STABLE
        
        # Get recent metrics for evaluation
        evaluation_window = timedelta(minutes=self.scaling_policy.evaluation_window_minutes)
        cutoff_time = datetime.now() - evaluation_window
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if len(recent_metrics) < 2:
            return ScalingDirection.STABLE
        
        # Calculate average metrics over evaluation window
        avg_cpu = statistics.mean([m.cpu_usage_percent for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_usage_percent for m in recent_metrics])
        avg_queue_size = statistics.mean([m.queue_size for m in recent_metrics])
        avg_response_time = statistics.mean([m.avg_response_time_ms for m in recent_metrics])
        
        # Scale up conditions
        scale_up_triggers = [
            avg_cpu > self.scaling_policy.cpu_scale_up_threshold,
            avg_memory > self.scaling_policy.memory_scale_up_threshold,
            avg_queue_size > self.scaling_policy.queue_size_scale_up_threshold,
            avg_response_time > self.scaling_policy.response_time_scale_up_threshold_ms
        ]
        
        # Scale down conditions
        scale_down_triggers = [
            avg_cpu < self.scaling_policy.cpu_scale_down_threshold,
            avg_memory < self.scaling_policy.memory_scale_down_threshold,
            avg_queue_size < self.scaling_policy.queue_size_scale_down_threshold,
            avg_response_time < self.scaling_policy.response_time_scale_down_threshold_ms
        ]
        
        # Decision logic
        if any(scale_up_triggers) and self.current_workers < self.scaling_policy.max_workers:
            return ScalingDirection.SCALE_UP
        elif all(scale_down_triggers) and self.current_workers > self.scaling_policy.min_workers:
            return ScalingDirection.SCALE_DOWN
        else:
            return ScalingDirection.STABLE

    def _execute_scaling_decision(
        self,
        direction: ScalingDirection,
        current_metrics: ScalingMetrics
    ) -> None:
        """Execute the scaling decision."""
        if direction == ScalingDirection.SCALE_UP:
            new_worker_count = min(
                self.current_workers + self.scaling_policy.scale_up_increment,
                self.scaling_policy.max_workers
            )
            
            if self.scale_up_callback and new_worker_count > self.current_workers:
                success = self.scale_up_callback(new_worker_count)
                if success:
                    self._record_scaling_action("scale_up", self.current_workers, new_worker_count, current_metrics)
                    self.current_workers = new_worker_count
                    self.last_scaling_action = datetime.now()
        
        elif direction == ScalingDirection.SCALE_DOWN:
            new_worker_count = max(
                self.current_workers - self.scaling_policy.scale_down_increment,
                self.scaling_policy.min_workers
            )
            
            if self.scale_down_callback and new_worker_count < self.current_workers:
                success = self.scale_down_callback(new_worker_count)
                if success:
                    self._record_scaling_action("scale_down", self.current_workers, new_worker_count, current_metrics)
                    self.current_workers = new_worker_count
                    self.last_scaling_action = datetime.now()

    def _record_scaling_action(
        self,
        action: str,
        old_workers: int,
        new_workers: int,
        metrics: ScalingMetrics
    ) -> None:
        """Record a scaling action for analysis."""
        scaling_record = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'old_workers': old_workers,
            'new_workers': new_workers,
            'trigger_metrics': metrics.to_dict(),
            'reason': self._get_scaling_reason(metrics)
        }
        
        self.scaling_history.append(scaling_record)
        
        # Keep only recent history
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]
        
        self.logger.info(
            f"Scaling action: {action} from {old_workers} to {new_workers} workers"
        )

    def _get_scaling_reason(self, metrics: ScalingMetrics) -> str:
        """Get human-readable reason for scaling decision."""
        reasons = []
        
        if metrics.cpu_usage_percent > self.scaling_policy.cpu_scale_up_threshold:
            reasons.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        elif metrics.cpu_usage_percent < self.scaling_policy.cpu_scale_down_threshold:
            reasons.append(f"Low CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        if metrics.memory_usage_percent > self.scaling_policy.memory_scale_up_threshold:
            reasons.append(f"High memory usage: {metrics.memory_usage_percent:.1f}%")
        elif metrics.memory_usage_percent < self.scaling_policy.memory_scale_down_threshold:
            reasons.append(f"Low memory usage: {metrics.memory_usage_percent:.1f}%")
        
        if metrics.queue_size > self.scaling_policy.queue_size_scale_up_threshold:
            reasons.append(f"Large queue size: {metrics.queue_size}")
        elif metrics.queue_size < self.scaling_policy.queue_size_scale_down_threshold:
            reasons.append(f"Small queue size: {metrics.queue_size}")
        
        return "; ".join(reasons) if reasons else "Threshold-based decision"

    def get_current_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        recent_metrics = None
        if self.metrics_history:
            recent_metrics = self.metrics_history[-1].to_dict()
        
        return {
            'scaling_active': self.scaling_active,
            'current_workers': self.current_workers,
            'min_workers': self.scaling_policy.min_workers,
            'max_workers': self.scaling_policy.max_workers,
            'last_scaling_action': self.last_scaling_action.isoformat(),
            'recent_metrics': recent_metrics,
            'scaling_actions_count': len(self.scaling_history),
            'metrics_history_count': len(self.metrics_history)
        }

    def get_scaling_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get scaling action history."""
        history = self.scaling_history
        if limit:
            history = history[-limit:]
        return history

    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics history."""
        history = [m.to_dict() for m in self.metrics_history]
        if limit:
            history = history[-limit:]
        return history

    def force_scale_to(self, target_workers: int) -> bool:
        """Force scaling to a specific number of workers."""
        if target_workers < self.scaling_policy.min_workers or target_workers > self.scaling_policy.max_workers:
            self.logger.error(f"Target workers {target_workers} out of range")
            return False
        
        if target_workers == self.current_workers:
            return True
        
        if target_workers > self.current_workers:
            if self.scale_up_callback:
                success = self.scale_up_callback(target_workers)
                if success:
                    self.current_workers = target_workers
                    self.last_scaling_action = datetime.now()
                return success
        else:
            if self.scale_down_callback:
                success = self.scale_down_callback(target_workers)
                if success:
                    self.current_workers = target_workers
                    self.last_scaling_action = datetime.now()
                return success
        
        return False


# Global auto-scaler instance
global_auto_scaler = AutoScaler(
    initial_workers=4,
    enable_predictive_scaling=True
)

def configure_auto_scaling(
    min_workers: int = 1,
    max_workers: int = 16,
    scale_up_callback: Optional[Callable[[int], bool]] = None,
    scale_down_callback: Optional[Callable[[int], bool]] = None
) -> None:
    """Configure global auto-scaling parameters."""
    global_auto_scaler.scaling_policy.min_workers = min_workers
    global_auto_scaler.scaling_policy.max_workers = max_workers
    
    if scale_up_callback and scale_down_callback:
        global_auto_scaler.set_scaling_callbacks(scale_up_callback, scale_down_callback)

def start_auto_scaling() -> None:
    """Start global auto-scaling."""
    global_auto_scaler.start_monitoring()

def stop_auto_scaling() -> None:
    """Stop global auto-scaling."""
    global_auto_scaler.stop_monitoring()

def get_scaling_status() -> Dict[str, Any]:
    """Get current auto-scaling status."""
    return global_auto_scaler.get_current_status()

def force_scale_to(target_workers: int) -> bool:
    """Force scaling to specific number of workers."""
    return global_auto_scaler.force_scale_to(target_workers)


class AutoScalingManager:
    """High-level auto-scaling management system."""
    
    def __init__(self, initial_workers: int = 4):
        self.auto_scaler = AutoScaler(initial_workers=initial_workers)
        
    def get_current_load(self) -> Dict[str, float]:
        """Get current system load metrics."""
        try:
            return self.auto_scaler.get_current_status()
        except Exception:
            # Fallback metrics when monitoring unavailable
            return {
                "cpu_usage": 0.5,
                "memory_usage": 0.3,
                "active_workers": self.auto_scaler.current_workers
            }
        
    def scale_resources(self, target_workers: int) -> bool:
        """Scale resources to target number of workers."""
        try:
            return self.auto_scaler.force_scale_to(target_workers)
        except Exception:
            return False
            
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations based on current metrics."""
        metrics = self.get_current_load()
        return {
            "current_workers": self.auto_scaler.current_workers,
            "recommended_action": "stable",
            "metrics": metrics
        }
        
    def start_auto_scaling(self):
        """Start automatic scaling monitoring."""
        self.auto_scaler.start_monitoring()
        
    def stop_auto_scaling(self):
        """Stop automatic scaling monitoring."""
        self.auto_scaler.stop_monitoring()