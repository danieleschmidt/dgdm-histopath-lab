"""
Intelligent auto-scaling and performance optimization for DGDM Histopath Lab.

This module provides dynamic resource management, intelligent caching,
and automated performance optimization based on workload analysis.
"""

import time
import threading
import json
import psutil
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import concurrent.futures
from functools import wraps, lru_cache
import weakref
import gc


@dataclass
class ResourceMetrics:
    """Real-time resource utilization metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    active_threads: int = 0
    queue_length: int = 0


@dataclass
class PerformanceProfile:
    """Performance profile for different workload types."""
    workload_type: str
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_execution_time: float
    optimal_batch_size: int
    recommended_workers: int
    cache_hit_rate: float
    throughput_per_hour: float


class IntelligentCache:
    """Adaptive caching system with automatic eviction and preloading."""
    
    def __init__(self, max_size_gb: float = 2.0, ttl_hours: float = 24.0):
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.ttl_seconds = ttl_hours * 3600
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.size_estimates = {}
        self.lock = threading.RLock()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with automatic statistics tracking."""
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.access_times[key] > self.ttl_seconds:
                    self._evict_key(key)
                    self.misses += 1
                    return None
                
                # Update access statistics
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, size_hint: Optional[int] = None) -> bool:
        """Put item in cache with intelligent eviction."""
        with self.lock:
            # Estimate size if not provided
            if size_hint is None:
                size_hint = self._estimate_size(value)
            
            # Check if we need to make space
            current_size = sum(self.size_estimates.values())
            if current_size + size_hint > self.max_size_bytes:
                if not self._make_space(size_hint):
                    return False  # Couldn't make enough space
            
            # Store item
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.size_estimates[key] = size_hint
            
            return True
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if hasattr(obj, '__sizeof__'):
            return obj.__sizeof__()
        elif isinstance(obj, (str, bytes)):
            return len(obj)
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
        else:
            return sys.getsizeof(obj)
    
    def _make_space(self, needed_bytes: int) -> bool:
        """Make space by evicting least recently used items."""
        # Sort by access time (LRU)
        sorted_keys = sorted(
            self.cache.keys(),
            key=lambda k: (self.access_times[k], self.access_counts[k])
        )
        
        freed_bytes = 0
        for key in sorted_keys:
            if freed_bytes >= needed_bytes:
                break
            
            freed_bytes += self.size_estimates[key]
            self._evict_key(key)
        
        return freed_bytes >= needed_bytes
    
    def _evict_key(self, key: str):
        """Evict a specific key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            del self.size_estimates[key]
            self.evictions += 1
    
    def _cleanup_worker(self):
        """Background worker for cache cleanup."""
        while True:
            time.sleep(300)  # Run every 5 minutes
            
            with self.lock:
                current_time = time.time()
                expired_keys = [
                    key for key, access_time in self.access_times.items()
                    if current_time - access_time > self.ttl_seconds
                ]
                
                for key in expired_keys:
                    self._evict_key(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) * 100 if total_requests > 0 else 0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "current_size_mb": sum(self.size_estimates.values()) / (1024**2),
                "max_size_mb": self.max_size_bytes / (1024**2),
                "items_count": len(self.cache)
            }


class AdaptiveThreadPool:
    """Thread pool that adapts size based on workload and system resources."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 16):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.current_workers)
        self.pending_tasks = 0
        self.completed_tasks = 0
        self.task_times = deque(maxlen=100)
        
        self.lock = threading.Lock()
        self.last_resize = time.time()
        self.resize_cooldown = 30  # seconds
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self.monitor_thread.start()
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task with automatic load tracking."""
        with self.lock:
            self.pending_tasks += 1
        
        def wrapper():
            start_time = time.time()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                with self.lock:
                    self.pending_tasks -= 1
                    self.completed_tasks += 1
                    self.task_times.append(execution_time)
        
        return self.executor.submit(wrapper)
    
    def _monitor_worker(self):
        """Monitor workload and adjust thread pool size."""
        while True:
            time.sleep(10)  # Check every 10 seconds
            
            if time.time() - self.last_resize < self.resize_cooldown:
                continue
            
            with self.lock:
                # Calculate metrics
                if len(self.task_times) > 5:
                    avg_task_time = sum(self.task_times) / len(self.task_times)
                    queue_pressure = self.pending_tasks / self.current_workers
                    
                    # Get system metrics
                    cpu_usage = psutil.cpu_percent(interval=1)
                    memory_usage = psutil.virtual_memory().percent
                    
                    # Decide if we need to resize
                    should_increase = (
                        queue_pressure > 2.0 and  # High queue pressure
                        cpu_usage < 80 and       # CPU not saturated
                        memory_usage < 85 and    # Memory not saturated
                        self.current_workers < self.max_workers
                    )
                    
                    should_decrease = (
                        queue_pressure < 0.5 and  # Low queue pressure
                        avg_task_time < 1.0 and   # Fast tasks
                        self.current_workers > self.min_workers
                    )
                    
                    if should_increase:
                        self._resize_pool(self.current_workers + 2)
                    elif should_decrease:
                        self._resize_pool(self.current_workers - 1)
    
    def _resize_pool(self, new_size: int):
        """Resize thread pool to new size."""
        new_size = max(self.min_workers, min(new_size, self.max_workers))
        
        if new_size != self.current_workers:
            old_executor = self.executor
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=new_size)
            self.current_workers = new_size
            self.last_resize = time.time()
            
            # Shutdown old executor gracefully
            threading.Thread(target=lambda: old_executor.shutdown(wait=True), daemon=True).start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        with self.lock:
            avg_task_time = sum(self.task_times) / len(self.task_times) if self.task_times else 0
            
            return {
                "current_workers": self.current_workers,
                "pending_tasks": self.pending_tasks,
                "completed_tasks": self.completed_tasks,
                "avg_task_time": avg_task_time,
                "queue_pressure": self.pending_tasks / self.current_workers if self.current_workers > 0 else 0
            }


class ResourceMonitor:
    """Continuous monitoring of system resources with trend analysis."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.lock = threading.Lock()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_worker(self):
        """Continuous resource monitoring."""
        while True:
            try:
                metrics = self._collect_metrics()
                with self.lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(5)  # Collect every 5 seconds
            except Exception:
                time.sleep(10)  # Back off on errors
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=(disk.used / disk.total) * 100,
            active_threads=threading.active_count()
        )
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent metrics."""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_trends(self, minutes: int = 30) -> Dict[str, Any]:
        """Analyze resource trends over time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {}
        
        # Calculate trends
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            "period_minutes": minutes,
            "samples": len(recent_metrics),
            "cpu_trend": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values),
                "current": cpu_values[-1]
            },
            "memory_trend": {
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": sum(memory_values) / len(memory_values),
                "current": memory_values[-1]
            },
            "alerts": self._generate_alerts(recent_metrics)
        }
    
    def _generate_alerts(self, metrics: List[ResourceMetrics]) -> List[str]:
        """Generate resource alerts."""
        alerts = []
        
        if metrics:
            latest = metrics[-1]
            
            if latest.cpu_percent > 90:
                alerts.append("HIGH CPU usage detected")
            
            if latest.memory_percent > 90:
                alerts.append("HIGH memory usage detected")
            
            if latest.disk_usage_percent > 90:
                alerts.append("HIGH disk usage detected")
            
            # Check for sustained high usage
            if len(metrics) >= 10:
                recent_cpu = [m.cpu_percent for m in metrics[-10:]]
                if all(cpu > 80 for cpu in recent_cpu):
                    alerts.append("SUSTAINED high CPU usage")
        
        return alerts


class IntelligentScaler:
    """Main orchestrator for intelligent scaling and optimization."""
    
    def __init__(self):
        self.cache = IntelligentCache()
        self.thread_pool = AdaptiveThreadPool()
        self.resource_monitor = ResourceMonitor()
        
        self.performance_profiles = {}
        self.optimization_history = []
        
        # Auto-optimization settings
        self.auto_optimize = True
        self.optimization_interval = 300  # 5 minutes
        
        # Start optimization thread
        self.optimizer_thread = threading.Thread(target=self._optimization_worker, daemon=True)
        self.optimizer_thread.start()
    
    def cached_operation(self, cache_key: str, ttl_hours: float = 24.0):
        """Decorator for caching expensive operations."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Try cache first
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.cache.put(cache_key, result)
                
                return result
            return wrapper
        return decorator
    
    def optimized_processing(self, workload_type: str = "default"):
        """Decorator for optimized processing with resource management."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Get current metrics
                metrics = self.resource_monitor.get_current_metrics()
                
                # Execute with resource monitoring
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Update performance profile
                    self._update_performance_profile(workload_type, metrics, execution_time, True)
                    
                    return result
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    self._update_performance_profile(workload_type, metrics, execution_time, False)
                    raise e
            
            return wrapper
        return decorator
    
    def _update_performance_profile(self, workload_type: str, metrics: Optional[ResourceMetrics], 
                                  execution_time: float, success: bool):
        """Update performance profile for workload type."""
        if workload_type not in self.performance_profiles:
            self.performance_profiles[workload_type] = {
                "samples": 0,
                "total_cpu": 0,
                "total_memory": 0,
                "total_time": 0,
                "success_count": 0
            }
        
        profile = self.performance_profiles[workload_type]
        profile["samples"] += 1
        profile["total_time"] += execution_time
        
        if metrics:
            profile["total_cpu"] += metrics.cpu_percent
            profile["total_memory"] += metrics.memory_percent
        
        if success:
            profile["success_count"] += 1
    
    def _optimization_worker(self):
        """Background worker for continuous optimization."""
        while True:
            time.sleep(self.optimization_interval)
            
            if self.auto_optimize:
                try:
                    self._run_optimizations()
                except Exception:
                    pass  # Don't let optimization failures crash the system
    
    def _run_optimizations(self):
        """Run optimization analysis and adjustments."""
        # Analyze cache performance
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < 50:  # Low hit rate
            # Could suggest cache size adjustment or TTL changes
            pass
        
        # Analyze thread pool performance
        pool_stats = self.thread_pool.get_stats()
        
        # Analyze resource trends
        trends = self.resource_monitor.get_trends()
        
        # Record optimization event
        optimization_event = {
            "timestamp": datetime.now().isoformat(),
            "cache_stats": cache_stats,
            "pool_stats": pool_stats,
            "resource_trends": trends,
            "performance_profiles": self._get_performance_summary()
        }
        
        self.optimization_history.append(optimization_event)
        
        # Keep only last 100 optimization events
        if len(self.optimization_history) > 100:
            self.optimization_history.pop(0)
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance profiles."""
        summary = {}
        
        for workload_type, profile in self.performance_profiles.items():
            if profile["samples"] > 0:
                summary[workload_type] = {
                    "avg_cpu": profile["total_cpu"] / profile["samples"],
                    "avg_memory": profile["total_memory"] / profile["samples"],
                    "avg_execution_time": profile["total_time"] / profile["samples"],
                    "success_rate": (profile["success_count"] / profile["samples"]) * 100,
                    "samples": profile["samples"]
                }
        
        return summary
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Generate comprehensive scaling and performance report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cache_performance": self.cache.get_stats(),
            "thread_pool_performance": self.thread_pool.get_stats(),
            "current_resources": asdict(self.resource_monitor.get_current_metrics()) if self.resource_monitor.get_current_metrics() else {},
            "resource_trends": self.resource_monitor.get_trends(),
            "performance_profiles": self._get_performance_summary(),
            "optimization_events": len(self.optimization_history),
            "scaling_recommendations": self._generate_scaling_recommendations()
        }
    
    def _generate_scaling_recommendations(self) -> List[str]:
        """Generate scaling recommendations based on analysis."""
        recommendations = []
        
        # Cache recommendations
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < 30:
            recommendations.append("Cache hit rate is low. Consider increasing cache size or adjusting TTL.")
        
        # Resource recommendations
        trends = self.resource_monitor.get_trends()
        if trends.get("cpu_trend", {}).get("avg", 0) > 80:
            recommendations.append("High average CPU usage. Consider scaling horizontally.")
        
        if trends.get("memory_trend", {}).get("avg", 0) > 80:
            recommendations.append("High memory usage. Consider optimizing memory usage or adding RAM.")
        
        # Thread pool recommendations
        pool_stats = self.thread_pool.get_stats()
        if pool_stats["queue_pressure"] > 3:
            recommendations.append("High queue pressure. Consider increasing max workers.")
        
        return recommendations
    
    def force_garbage_collection(self):
        """Force garbage collection and memory optimization."""
        # Clear weak references
        gc.collect()
        
        # Force cache cleanup if memory is high
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 85:
            with self.cache.lock:
                # Aggressively evict old cache entries
                current_time = time.time()
                old_keys = [
                    key for key, access_time in self.cache.access_times.items()
                    if current_time - access_time > 3600  # 1 hour
                ]
                
                for key in old_keys[:len(old_keys)//2]:  # Evict half of old entries
                    self.cache._evict_key(key)


# Global scaler instance
intelligent_scaler = IntelligentScaler()


# Convenience decorators
def cached(cache_key: str, ttl_hours: float = 24.0):
    """Convenience decorator for caching."""
    return intelligent_scaler.cached_operation(cache_key, ttl_hours)


def optimized(workload_type: str = "default"):
    """Convenience decorator for optimization."""
    return intelligent_scaler.optimized_processing(workload_type)


if __name__ == "__main__":
    # Generate scaling report
    report = intelligent_scaler.get_scaling_report()
    
    print("=" * 80)
    print("INTELLIGENT SCALING REPORT")
    print("=" * 80)
    
    print("Cache Performance:")
    cache_stats = report["cache_performance"]
    print(f"  Hit Rate: {cache_stats['hit_rate']:.1f}%")
    print(f"  Items: {cache_stats['items_count']}")
    print(f"  Size: {cache_stats['current_size_mb']:.1f}MB / {cache_stats['max_size_mb']:.1f}MB")
    
    print("\nThread Pool Performance:")
    pool_stats = report["thread_pool_performance"]
    print(f"  Workers: {pool_stats['current_workers']}")
    print(f"  Queue Pressure: {pool_stats['queue_pressure']:.2f}")
    print(f"  Avg Task Time: {pool_stats['avg_task_time']:.3f}s")
    
    print("\nCurrent Resources:")
    resources = report["current_resources"]
    if resources:
        print(f"  CPU: {resources['cpu_percent']:.1f}%")
        print(f"  Memory: {resources['memory_percent']:.1f}%")
        print(f"  Available Memory: {resources['memory_available_gb']:.1f}GB")
    
    print("\nRecommendations:")
    for rec in report["scaling_recommendations"]:
        print(f"  • {rec}")
    
    print("=" * 80)