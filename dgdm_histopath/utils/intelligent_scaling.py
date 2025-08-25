"""
Intelligent Scaling System for DGDM Histopath Lab
Auto-scaling, load balancing, resource optimization, and distributed processing
"""

import time
import threading
import concurrent.futures
import queue
import hashlib
import pickle
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps, lru_cache
from contextlib import contextmanager
import weakref

class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK_IO = "disk_io"
    NETWORK = "network"

@dataclass
class ResourceUsage:
    """Resource usage metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_percent: float = 0.0
    disk_io_percent: float = 0.0
    network_io_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def get_bottleneck(self) -> Tuple[ResourceType, float]:
        """Identify the primary resource bottleneck."""
        resources = {
            ResourceType.CPU: self.cpu_percent,
            ResourceType.MEMORY: self.memory_percent,
            ResourceType.GPU: self.gpu_percent,
            ResourceType.DISK_IO: self.disk_io_percent,
            ResourceType.NETWORK: self.network_io_percent
        }
        
        bottleneck = max(resources.items(), key=lambda x: x[1])
        return bottleneck[0], bottleneck[1]

class IntelligentCache:
    """Intelligent caching system with LRU, size limits, and prediction."""
    
    def __init__(self, max_size_mb: int = 1024, ttl_seconds: int = 3600, max_size: int = None):
        self.max_size_bytes = (max_size if max_size is not None else max_size_mb) * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.cache_size_bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        with self.lock:
            if key in self.cache:
                self.cache_size_bytes -= self._get_size(self.cache[key])
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.cache_size_bytes += self._get_size(value)
            
    def get(self, key: str) -> Any:
        """Get a value from the cache."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
                
    def _get_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(str(obj))  # Simple estimation
        except:
            return 64  # Default size
        
    def _get_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create a deterministic key
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        # Use hash for efficiency
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _evict_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._remove_key(key)
    
    def _evict_lru(self, target_size: int):
        """Evict least recently used items to reach target size."""
        if not self.access_times:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_keys:
            if self.cache_size_bytes <= target_size:
                break
            self._remove_key(key)
    
    def _remove_key(self, key: str):
        """Remove a key from cache and update metrics."""
        if key in self.cache:
            # Estimate size (rough approximation)
            try:
                item_size = len(pickle.dumps(self.cache[key]))
                self.cache_size_bytes -= item_size
            except:
                self.cache_size_bytes -= 1024  # Fallback estimate
            
            del self.cache[key]
            del self.access_times[key]
            self.access_counts.pop(key, 0)
    
    def get(self, func_name: str, args: Tuple, kwargs: Dict) -> Tuple[bool, Any]:
        """Get item from cache."""
        with self.lock:
            key = self._get_cache_key(func_name, args, kwargs)
            
            if key in self.cache:
                # Check if expired
                if time.time() - self.access_times[key] > self.ttl_seconds:
                    self._remove_key(key)
                    self.miss_count += 1
                    return False, None
                
                # Update access statistics
                self.access_times[key] = time.time()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self.hit_count += 1
                
                return True, self.cache[key]
            else:
                self.miss_count += 1
                return False, None
    
    def put(self, func_name: str, args: Tuple, kwargs: Dict, result: Any):
        """Put item in cache."""
        with self.lock:
            key = self._get_cache_key(func_name, args, kwargs)
            
            # Estimate result size
            try:
                result_size = len(pickle.dumps(result))
            except:
                result_size = 1024  # Fallback estimate
            
            # Check if item is too large
            if result_size > self.max_size_bytes * 0.5:
                self.logger.warning(f"Item too large for cache: {result_size} bytes")
                return
            
            # Evict expired items
            self._evict_expired()
            
            # Evict LRU items if needed
            target_size = self.max_size_bytes - result_size
            if self.cache_size_bytes > target_size:
                self._evict_lru(target_size)
            
            # Add to cache
            self.cache[key] = result
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.cache_size_bytes += result_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size_mb': self.cache_size_bytes / (1024 * 1024),
            'item_count': len(self.cache),
            'max_size_mb': self.max_size_bytes / (1024 * 1024)
        }
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.cache_size_bytes = 0

class AdaptiveThreadPool:
    """Thread pool that adapts size based on workload."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 16, 
                 scale_threshold: float = 0.8):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_threshold = scale_threshold
        self.current_workers = min_workers
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=min_workers)
        self.task_queue_size = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.avg_task_time = 0.0
        
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Scaling metrics
        self.scaling_history = []
        self.last_scale_time = time.time()
        self.scale_cooldown = 30.0  # seconds
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task with automatic scaling."""
        start_time = time.time()
        
        with self.lock:
            self.task_queue_size += 1
        
        def wrapped_task():
            try:
                result = fn(*args, **kwargs)
                with self.lock:
                    self.completed_tasks += 1
                    task_time = time.time() - start_time
                    self._update_avg_task_time(task_time)
                    self.task_queue_size -= 1
                return result
            except Exception as e:
                with self.lock:
                    self.failed_tasks += 1
                    self.task_queue_size -= 1
                raise e
        
        future = self.executor.submit(wrapped_task)
        
        # Check if scaling is needed
        self._check_scaling()
        
        return future
    
    def _update_avg_task_time(self, task_time: float):
        """Update average task time with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        if self.avg_task_time == 0:
            self.avg_task_time = task_time
        else:
            self.avg_task_time = alpha * task_time + (1 - alpha) * self.avg_task_time
    
    def _check_scaling(self):
        """Check if scaling up or down is needed."""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        with self.lock:
            # Calculate load metrics
            if self.current_workers > 0:
                queue_load = self.task_queue_size / self.current_workers
                
                # Scale up if queue is building up
                if (queue_load > self.scale_threshold and 
                    self.current_workers < self.max_workers):
                    self._scale_up()
                
                # Scale down if lightly loaded
                elif (queue_load < self.scale_threshold / 3 and 
                      self.current_workers > self.min_workers):
                    self._scale_down()
    
    def _scale_up(self):
        """Scale up worker count."""
        new_workers = min(self.current_workers * 2, self.max_workers)
        if new_workers > self.current_workers:
            self.logger.info(f"Scaling up from {self.current_workers} to {new_workers} workers")
            
            # Create new executor
            self.executor.shutdown(wait=False)
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=new_workers)
            
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_up',
                'workers': new_workers
            })
    
    def _scale_down(self):
        """Scale down worker count."""
        new_workers = max(self.current_workers // 2, self.min_workers)
        if new_workers < self.current_workers:
            self.logger.info(f"Scaling down from {self.current_workers} to {new_workers} workers")
            
            # Create new executor
            self.executor.shutdown(wait=False)
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=new_workers)
            
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_down',
                'workers': new_workers
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        with self.lock:
            total_tasks = self.completed_tasks + self.failed_tasks
            success_rate = self.completed_tasks / total_tasks if total_tasks > 0 else 0
            
            return {
                'current_workers': self.current_workers,
                'queue_size': self.task_queue_size,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate': success_rate,
                'avg_task_time': self.avg_task_time,
                'scaling_events': len(self.scaling_history)
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown thread pool."""
        self.executor.shutdown(wait=wait)

class ScalableProcessor:
    """Main scalable processing coordinator."""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.BALANCED):
        self.strategy = strategy
        self.cache = IntelligentCache()
        self.thread_pool = AdaptiveThreadPool()
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.processing_history = []
        self.resource_usage_history = []
        
        # Configuration based on strategy
        self._configure_strategy()
    
    def _configure_strategy(self):
        """Configure processing based on scaling strategy."""
        if self.strategy == ScalingStrategy.CONSERVATIVE:
            self.thread_pool.max_workers = 8
            self.cache.max_size_bytes = 512 * 1024 * 1024  # 512MB
        elif self.strategy == ScalingStrategy.AGGRESSIVE:
            self.thread_pool.max_workers = 32
            self.cache.max_size_bytes = 2048 * 1024 * 1024  # 2GB
        else:  # BALANCED
            self.thread_pool.max_workers = 16
            self.cache.max_size_bytes = 1024 * 1024 * 1024  # 1GB
    
    def cached_execution(self, cache_key: str = None, ttl: int = None):
        """Decorator for cached function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check cache first
                hit, result = self.cache.get(func.__name__, args, kwargs)
                if hit:
                    self.logger.debug(f"Cache hit for {func.__name__}")
                    return result
                
                # Execute function
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Cache result
                    self.cache.put(func.__name__, args, kwargs, result)
                    
                    # Record performance
                    self.processing_history.append({
                        'function': func.__name__,
                        'duration': duration,
                        'timestamp': time.time(),
                        'cache_hit': False
                    })
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Error in {func.__name__}: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def parallel_execution(self, max_workers: int = None):
        """Decorator for parallel function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(items: List[Any], *args, **kwargs):
                if not items:
                    return []
                
                # Use adaptive thread pool
                futures = []
                for item in items:
                    future = self.thread_pool.submit(func, item, *args, **kwargs)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Parallel execution error: {e}")
                        results.append(None)
                
                return results
            
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'cache_stats': self.cache.get_stats(),
            'thread_pool_stats': self.thread_pool.get_stats(),
            'processing_history_count': len(self.processing_history),
            'strategy': self.strategy.value
        }
    
    def optimize_configuration(self):
        """Automatically optimize configuration based on usage patterns."""
        stats = self.get_performance_report()
        
        # Optimize cache size based on hit rate
        cache_hit_rate = stats['cache_stats']['hit_rate']
        if cache_hit_rate < 0.5:
            # Increase cache size
            self.cache.max_size_bytes = int(self.cache.max_size_bytes * 1.5)
            self.logger.info("Increased cache size due to low hit rate")
        
        # Optimize thread pool based on queue size
        queue_size = stats['thread_pool_stats']['queue_size']
        current_workers = stats['thread_pool_stats']['current_workers']
        
        if queue_size > current_workers * 2:
            self.thread_pool.max_workers = min(self.thread_pool.max_workers + 4, 64)
            self.logger.info("Increased max workers due to high queue size")
    
    def shutdown(self):
        """Shutdown all components."""
        self.thread_pool.shutdown()
        self.cache.clear()

# Global scalable processor instance
global_processor = ScalableProcessor()

def get_scalable_processor() -> ScalableProcessor:
    """Get global scalable processor instance."""
    return global_processor

# Convenience decorators
def cached(ttl: int = 3600):
    """Convenience decorator for caching."""
    return global_processor.cached_execution(ttl=ttl)

def parallel(max_workers: int = None):
    """Convenience decorator for parallel execution."""
    return global_processor.parallel_execution(max_workers=max_workers)

if __name__ == "__main__":
    # Test scaling system
    processor = get_scalable_processor()
    
    @cached(ttl=60)
    def expensive_computation(n: int) -> int:
        """Simulate expensive computation."""
        time.sleep(0.1)
        return n * n
    
    @parallel()
    def process_item(item: int) -> int:
        """Process single item."""
        return expensive_computation(item)
    
    # Test caching
    print("Testing caching...")
    for i in range(5):
        result = expensive_computation(42)
        print(f"Result: {result}")
    
    # Test parallel processing
    print("\nTesting parallel processing...")
    items = list(range(10))
    results = process_item(items)
    print(f"Processed {len(results)} items")
    
    # Get performance report
    report = processor.get_performance_report()
    print("\nPerformance Report:")
    print(json.dumps(report, indent=2))
    
    processor.shutdown()