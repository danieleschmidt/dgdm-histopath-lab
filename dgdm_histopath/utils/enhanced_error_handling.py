"""
Enhanced error handling and resilience for DGDM Histopath Lab.

This module provides comprehensive error handling, circuit breakers,
retry mechanisms, and graceful degradation strategies.
"""

import functools
import time
import logging
import sys
import traceback
import threading
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    traceback: str
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class ResilientLogger:
    """Enhanced logger with error tracking and analysis."""
    
    def __init__(self, name: str = "dgdm_resilient"):
        self.logger = logging.getLogger(name)
        self.error_history: List[ErrorInfo] = []
        self.error_patterns: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def log_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                  context: Optional[Dict[str, Any]] = None) -> str:
        """Log error with enhanced tracking."""
        error_info = ErrorInfo(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            traceback=traceback.format_exc(),
            context=context or {}
        )
        
        with self.lock:
            self.error_history.append(error_info)
            self.error_patterns[error_info.error_type] = self.error_patterns.get(error_info.error_type, 0) + 1
        
        # Log appropriately based on severity
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"{error_info.error_type}: {error_info.error_message}")
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"{error_info.error_type}: {error_info.error_message}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"{error_info.error_type}: {error_info.error_message}")
        else:
            self.logger.info(f"{error_info.error_type}: {error_info.error_message}")
        
        return error_info.error_type
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and patterns."""
        with self.lock:
            recent_errors = [e for e in self.error_history if (datetime.now() - e.timestamp).days < 1]
            
            return {
                "total_errors": len(self.error_history),
                "recent_errors": len(recent_errors),
                "error_patterns": self.error_patterns.copy(),
                "severity_distribution": {
                    severity.value: sum(1 for e in self.error_history if e.severity == severity)
                    for severity in ErrorSeverity
                },
                "most_common_errors": sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            }


class CircuitBreaker:
    """Circuit breaker for handling failures gracefully."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN - failing fast")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class RetryStrategy:
    """Configurable retry strategy with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    # Last attempt failed
                    break
                
                # Calculate delay
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        # All attempts failed
        raise last_exception


class GracefulDegradation:
    """Manage graceful degradation strategies."""
    
    def __init__(self):
        self.fallback_functions: Dict[str, Callable] = {}
        self.degradation_levels: Dict[str, int] = {}
    
    def register_fallback(self, operation_name: str, fallback_func: Callable, 
                         degradation_level: int = 1):
        """Register a fallback function for an operation."""
        self.fallback_functions[operation_name] = fallback_func
        self.degradation_levels[operation_name] = degradation_level
    
    def execute_with_fallback(self, operation_name: str, primary_func: Callable, 
                            *args, **kwargs) -> Any:
        """Execute function with fallback on failure."""
        try:
            return primary_func(*args, **kwargs)
        
        except Exception as e:
            logging.warning(f"Primary operation '{operation_name}' failed: {e}")
            
            if operation_name in self.fallback_functions:
                logging.info(f"Executing fallback for '{operation_name}'")
                try:
                    return self.fallback_functions[operation_name](*args, **kwargs)
                except Exception as fallback_error:
                    logging.error(f"Fallback for '{operation_name}' also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise e


# Global instances
resilient_logger = ResilientLogger()
global_circuit_breaker = CircuitBreaker()
default_retry = RetryStrategy()
degradation_manager = GracefulDegradation()


def resilient(max_retries: int = 3, circuit_breaker: bool = True, 
              fallback: Optional[Callable] = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for making functions resilient."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__name__}"
            
            # Register fallback if provided
            if fallback:
                degradation_manager.register_fallback(operation_name, fallback)
            
            def execute_func():
                if circuit_breaker:
                    return global_circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            try:
                if max_retries > 1:
                    retry_strategy = RetryStrategy(max_attempts=max_retries)
                    return retry_strategy.execute(execute_func)
                else:
                    return execute_func()
            
            except Exception as e:
                error_id = resilient_logger.log_error(e, severity, {
                    "function": operation_name,
                    "args": str(args)[:100],  # Truncate for privacy
                    "kwargs": str(kwargs)[:100]
                })
                
                if fallback:
                    return degradation_manager.execute_with_fallback(
                        operation_name, func, *args, **kwargs
                    )
                else:
                    raise e
        
        return wrapper
    return decorator


def safe_import(module_name: str, fallback_value: Any = None) -> Any:
    """Safely import module with fallback."""
    try:
        module = __import__(module_name)
        return module
    except ImportError as e:
        resilient_logger.log_error(e, ErrorSeverity.LOW, {"module": module_name})
        return fallback_value


def safe_execution(func: Callable, *args, default_return: Any = None, 
                  log_errors: bool = True, **kwargs) -> Any:
    """Execute function safely with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            resilient_logger.log_error(e, ErrorSeverity.LOW, {
                "function": func.__name__ if hasattr(func, "__name__") else str(func),
                "safe_execution": True
            })
        return default_return


class HealthMonitor:
    """Monitor system health and detect issues."""
    
    def __init__(self):
        self.health_metrics: Dict[str, Any] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.alert_thresholds: Dict[str, float] = {}
        self.last_check_time = None
    
    def register_health_check(self, name: str, check_func: Callable, 
                            alert_threshold: Optional[float] = None):
        """Register a health check function."""
        self.health_checks[name] = check_func
        if alert_threshold is not None:
            self.alert_thresholds[name] = alert_threshold
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = safe_execution(check_func, default_return={"status": "error"})
                execution_time = time.time() - start_time
                
                results[name] = {
                    "result": result,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Check alert thresholds
                if name in self.alert_thresholds:
                    threshold = self.alert_thresholds[name]
                    if execution_time > threshold:
                        resilient_logger.log_error(
                            Exception(f"Health check '{name}' exceeded threshold: {execution_time:.2f}s > {threshold}s"),
                            ErrorSeverity.MEDIUM,
                            {"health_check": name, "execution_time": execution_time}
                        )
                
            except Exception as e:
                results[name] = {
                    "result": {"status": "error", "error": str(e)},
                    "execution_time": 0,
                    "timestamp": datetime.now().isoformat()
                }
        
        self.health_metrics = results
        self.last_check_time = datetime.now()
        
        return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        if not self.health_metrics:
            return {"status": "no_data", "checks_run": 0}
        
        total_checks = len(self.health_metrics)
        healthy_checks = sum(1 for result in self.health_metrics.values() 
                           if result["result"].get("status") != "error")
        
        health_percentage = (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
        
        return {
            "status": "healthy" if health_percentage >= 80 else "degraded" if health_percentage >= 50 else "unhealthy",
            "health_percentage": health_percentage,
            "checks_run": total_checks,
            "healthy_checks": healthy_checks,
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "error_summary": resilient_logger.get_error_summary()
        }


# Global health monitor
health_monitor = HealthMonitor()


def health_check(name: str, alert_threshold: Optional[float] = None):
    """Decorator to register function as health check."""
    def decorator(func: Callable) -> Callable:
        health_monitor.register_health_check(name, func, alert_threshold)
        return func
    return decorator


# Example health checks
@health_check("memory_usage", alert_threshold=2.0)
def check_memory_usage():
    """Check memory usage."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "status": "healthy" if memory.percent < 80 else "warning",
            "memory_percent": memory.percent,
            "available_gb": memory.available / (1024**3)
        }
    except ImportError:
        return {"status": "unknown", "reason": "psutil not available"}


@health_check("disk_space", alert_threshold=2.0)
def check_disk_space():
    """Check disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        usage_percent = (used / total) * 100
        
        return {
            "status": "healthy" if usage_percent < 80 else "warning",
            "usage_percent": usage_percent,
            "free_gb": free / (1024**3)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def create_resilience_report() -> Dict[str, Any]:
    """Create comprehensive resilience report."""
    health_summary = health_monitor.get_health_summary()
    error_summary = resilient_logger.get_error_summary()
    
    # Run health checks
    health_check_results = health_monitor.run_health_checks()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system_health": health_summary,
        "error_analysis": error_summary,
        "health_checks": health_check_results,
        "circuit_breaker_state": global_circuit_breaker.state.value,
        "recommendations": _generate_resilience_recommendations(health_summary, error_summary)
    }


def _generate_resilience_recommendations(health_summary: Dict[str, Any], 
                                       error_summary: Dict[str, Any]) -> List[str]:
    """Generate recommendations for improving resilience."""
    recommendations = []
    
    if health_summary["health_percentage"] < 80:
        recommendations.append("System health is below optimal. Check failing health checks.")
    
    if error_summary["recent_errors"] > 10:
        recommendations.append("High error rate detected. Review error patterns and implement fixes.")
    
    if error_summary.get("most_common_errors"):
        most_common = error_summary["most_common_errors"][0]
        recommendations.append(f"Most common error: {most_common[0]} ({most_common[1]} occurrences). Consider targeted fixes.")
    
    critical_errors = error_summary["severity_distribution"].get("critical", 0)
    if critical_errors > 0:
        recommendations.append(f"{critical_errors} critical errors detected. Immediate attention required.")
    
    return recommendations


if __name__ == "__main__":
    # Generate and display resilience report
    report = create_resilience_report()
    
    print("=" * 60)
    print("DGDM RESILIENCE REPORT")
    print("=" * 60)
    print(f"System Health: {report['system_health']['status']}")
    print(f"Health Percentage: {report['system_health']['health_percentage']:.1f}%")
    print(f"Recent Errors: {report['error_analysis']['recent_errors']}")
    print(f"Circuit Breaker: {report['circuit_breaker_state']}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    print("=" * 60)