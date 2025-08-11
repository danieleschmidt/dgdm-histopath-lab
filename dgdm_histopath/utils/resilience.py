"""
Resilience patterns and circuit breaker implementations for DGDM Histopath Lab.

Provides fault tolerance, graceful degradation, and self-healing capabilities
for robust operation in clinical environments.
"""

import time
import logging
import threading
import asyncio
import random
import statistics
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import json

from dgdm_histopath.utils.exceptions import (
    DGDMException, PerformanceError, ResourceError, global_exception_handler
)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # For half-open state
    timeout: float = 30.0
    expected_exception: type = Exception


class CircuitBreaker:
    """Enhanced circuit breaker with adaptive thresholds and monitoring."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        # Adaptive thresholds
        self.adaptive_threshold = self.config.failure_threshold
        self.recent_response_times = []
        self.baseline_response_time = None
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': 0
        }
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.metrics['total_calls'] += 1
            
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.metrics['rejected_calls'] += 1
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
        
        # Execute the function with timeout and monitoring
        start_time = time.time()
        try:
            result = self._execute_with_timeout(func, *args, **kwargs)
            response_time = time.time() - start_time
            
            self._record_success(response_time)
            return result
            
        except self.config.expected_exception as e:
            response_time = time.time() - start_time
            self._record_failure(response_time)
            raise e
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs):
        """Execute function with timeout protection."""
        if asyncio.iscoroutinefunction(func):
            # Handle async functions
            return self._execute_async_with_timeout(func, *args, **kwargs)
        else:
            # Handle sync functions
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {self.config.timeout} seconds")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.timeout))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                return result
            except Exception:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                raise
    
    async def _execute_async_with_timeout(self, func: Callable, *args, **kwargs):
        """Execute async function with timeout."""
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Async function {func.__name__} timed out after {self.config.timeout} seconds")
    
    def _record_success(self, response_time: float):
        """Record successful execution."""
        with self._lock:
            self.failure_count = 0
            self.success_count += 1
            self.last_success_time = time.time()
            self.metrics['successful_calls'] += 1
            
            # Track response times for adaptive behavior
            self.recent_response_times.append(response_time)
            if len(self.recent_response_times) > 100:
                self.recent_response_times = self.recent_response_times[-100:]
            
            # Update baseline response time
            if self.baseline_response_time is None:
                self.baseline_response_time = response_time
            else:
                # Exponential moving average
                self.baseline_response_time = 0.9 * self.baseline_response_time + 0.1 * response_time
    
    def _record_failure(self, response_time: float = 0):
        """Record failed execution."""
        with self._lock:
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = time.time()
            self.metrics['failed_calls'] += 1
            
            # Adaptive threshold adjustment
            if len(self.recent_response_times) > 10:
                avg_response_time = statistics.mean(self.recent_response_times)
                if response_time > avg_response_time * 3:  # Very slow response
                    self.adaptive_threshold = max(2, self.adaptive_threshold - 1)
            
            if self.failure_count >= self.adaptive_threshold:
                self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _transition_to_open(self):
        """Transition to open state."""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.metrics['state_changes'] += 1
        
        self.logger.warning(
            f"Circuit breaker {self.name}: {old_state.value} -> {self.state.value} "
            f"after {self.failure_count} failures"
        )
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.metrics['state_changes'] += 1
        
        self.logger.info(f"Circuit breaker {self.name}: {old_state.value} -> {self.state.value}")
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        
        # Reset adaptive threshold gradually
        self.adaptive_threshold = min(
            self.config.failure_threshold,
            self.adaptive_threshold + 1
        )
        
        self.metrics['state_changes'] += 1
        
        self.logger.info(f"Circuit breaker {self.name}: {old_state.value} -> {self.state.value}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            success_rate = 0.0
            if self.metrics['total_calls'] > 0:
                success_rate = self.metrics['successful_calls'] / self.metrics['total_calls']
            
            avg_response_time = 0.0
            if self.recent_response_times:
                avg_response_time = statistics.mean(self.recent_response_times)
            
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'adaptive_threshold': self.adaptive_threshold,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'baseline_response_time': self.baseline_response_time,
                **self.metrics
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryPolicy:
    """Configurable retry policy with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for specific attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


class ResilientExecutor:
    """Execute operations with comprehensive resilience patterns."""
    
    def __init__(self, circuit_breaker: Optional[CircuitBreaker] = None,
                 retry_policy: Optional[RetryPolicy] = None,
                 timeout: Optional[float] = None):
        self.circuit_breaker = circuit_breaker
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def execute(self, func: Callable, *args, fallback: Optional[Callable] = None, **kwargs):
        """Execute function with all resilience patterns."""
        last_exception = None
        
        for attempt in range(self.retry_policy.max_attempts):
            try:
                if self.circuit_breaker:
                    return self.circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except (CircuitBreakerOpenError, TimeoutError) as e:
                # Don't retry these exceptions
                if fallback:
                    self.logger.warning(f"Using fallback due to: {e}")
                    return fallback(*args, **kwargs)
                raise e
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.retry_policy.max_attempts - 1:
                    delay = self.retry_policy.get_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.retry_policy.max_attempts} attempts failed")
        
        # All attempts failed
        if fallback:
            self.logger.warning("Using fallback after all retries failed")
            return fallback(*args, **kwargs)
        
        raise last_exception


class BulkheadExecutor:
    """Implement bulkhead pattern for resource isolation."""
    
    def __init__(self, max_concurrent: int = 10, queue_size: int = 100):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.queue_size = queue_size
        self.active_operations = 0
        self.queued_operations = 0
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def execute(self, func: Callable, *args, **kwargs):
        """Execute function with bulkhead protection."""
        with self._lock:
            if self.queued_operations >= self.queue_size:
                raise ResourceError(
                    f"Bulkhead queue full: {self.queued_operations} operations queued",
                    severity="WARNING"
                )
            self.queued_operations += 1
        
        try:
            # Wait for available slot
            acquired = self.semaphore.acquire(timeout=30.0)
            if not acquired:
                raise TimeoutError("Bulkhead timeout: Could not acquire execution slot")
            
            with self._lock:
                self.active_operations += 1
                self.queued_operations -= 1
            
            try:
                return func(*args, **kwargs)
            finally:
                with self._lock:
                    self.active_operations -= 1
                self.semaphore.release()
                
        except Exception as e:
            with self._lock:
                self.queued_operations -= 1
            raise e
    
    def get_status(self) -> Dict[str, Any]:
        """Get bulkhead status."""
        with self._lock:
            return {
                'active_operations': self.active_operations,
                'queued_operations': self.queued_operations,
                'available_slots': self.semaphore._value,
                'queue_utilization': self.queued_operations / self.queue_size
            }


class HealthMonitor:
    """Continuous health monitoring with automatic recovery."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.recovery_actions: Dict[str, Callable] = {}
        self.monitoring_active = False
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def register_health_check(self, name: str, check_func: Callable,
                             recovery_action: Optional[Callable] = None):
        """Register health check with optional recovery action."""
        with self._lock:
            self.health_checks[name] = check_func
            if recovery_action:
                self.recovery_actions[name] = recovery_action
            self.health_status[name] = {
                'status': 'unknown',
                'last_check': None,
                'failure_count': 0,
                'recovery_attempts': 0
            }
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        self.monitoring_active = False
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _run_health_checks(self):
        """Run all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = check_func()
                self._update_health_status(name, is_healthy, None)
                
                if not is_healthy and name in self.recovery_actions:
                    self._attempt_recovery(name)
                    
            except Exception as e:
                self._update_health_status(name, False, str(e))
                if name in self.recovery_actions:
                    self._attempt_recovery(name)
    
    def _update_health_status(self, name: str, is_healthy: bool, error: Optional[str]):
        """Update health status for component."""
        with self._lock:
            status = self.health_status[name]
            status['last_check'] = datetime.now().isoformat()
            
            if is_healthy:
                status['status'] = 'healthy'
                status['failure_count'] = 0
                status['error'] = None
            else:
                status['status'] = 'unhealthy'
                status['failure_count'] += 1
                status['error'] = error
                
                self.logger.warning(
                    f"Health check failed for {name}: {error} "
                    f"(failure count: {status['failure_count']})"
                )
    
    def _attempt_recovery(self, name: str):
        """Attempt automatic recovery for unhealthy component."""
        if name not in self.recovery_actions:
            return
        
        with self._lock:
            status = self.health_status[name]
            
            # Limit recovery attempts
            if status['recovery_attempts'] >= 3:
                self.logger.error(f"Max recovery attempts reached for {name}")
                return
            
            status['recovery_attempts'] += 1
        
        try:
            self.logger.info(f"Attempting recovery for {name}")
            recovery_action = self.recovery_actions[name]
            recovery_action()
            
            # Test health after recovery
            time.sleep(5)  # Wait a bit for recovery to take effect
            is_healthy = self.health_checks[name]()
            
            if is_healthy:
                self.logger.info(f"Recovery successful for {name}")
                with self._lock:
                    self.health_status[name]['recovery_attempts'] = 0
            else:
                self.logger.warning(f"Recovery failed for {name}")
                
        except Exception as e:
            self.logger.error(f"Recovery action failed for {name}: {e}")
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self._lock:
            healthy_count = sum(1 for status in self.health_status.values() 
                              if status['status'] == 'healthy')
            total_count = len(self.health_status)
            
            overall_status = 'healthy'
            if healthy_count == 0:
                overall_status = 'critical'
            elif healthy_count < total_count:
                overall_status = 'degraded'
            
            return {
                'overall_status': overall_status,
                'healthy_components': healthy_count,
                'total_components': total_count,
                'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 0,
                'component_status': dict(self.health_status),
                'timestamp': datetime.now().isoformat()
            }


# Decorators for easy integration
def with_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection."""
    def decorator(func):
        cb = CircuitBreaker(name, config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator to add retry functionality."""
    def decorator(func):
        retry_policy = RetryPolicy(max_attempts=max_attempts, base_delay=base_delay)
        executor = ResilientExecutor(retry_policy=retry_policy)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return executor.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def with_bulkhead(max_concurrent: int = 10):
    """Decorator to add bulkhead protection."""
    bulkhead = BulkheadExecutor(max_concurrent=max_concurrent)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return bulkhead.execute(func, *args, **kwargs)
        return wrapper
    return decorator


# Global instances
default_circuit_breaker = CircuitBreaker("default")
default_health_monitor = HealthMonitor()

# Start health monitoring by default
default_health_monitor.start_monitoring()