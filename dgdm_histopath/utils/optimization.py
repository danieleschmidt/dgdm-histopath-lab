"""
Advanced performance optimization and scaling system for DGDM Histopath Lab.

Implements comprehensive performance optimization including caching, parallelization,
resource pooling, and auto-scaling capabilities for production deployment.
"""

import os
import time
import threading
import multiprocessing
import asyncio
import weakref
import gc
import pickle
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, TypeVar, Generic
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import queue
import psutil
import numpy as np
from collections import defaultdict, OrderedDict
import json

try:
    import torch
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from dgdm_histopath.utils.exceptions import (
    PerformanceError, ResourceError, global_exception_handler
)
from dgdm_histopath.utils.monitoring import metrics_collector

T = TypeVar('T')


@dataclass
class PerformanceProfile:
    """Performance profiling data for optimization analysis."""
    operation_name: str
    duration: float
    memory_usage: int
    cpu_usage: float
    gpu_usage: Optional[float]
    cache_hits: int
    cache_misses: int
    throughput: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


class AdaptiveCache:
    """Adaptive caching system with intelligent eviction and performance tracking."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger("dgdm_histopath.optimization.cache")
        
        # Adaptive parameters
        self.access_frequency = defaultdict(int)
        self.access_recency = {}
        self.size_tracking = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with adaptive tracking."""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                item, stored_time = self.cache[key]
                
                # Check TTL
                if current_time - stored_time > self.ttl_seconds:
                    self._evict_item(key)
                    self.miss_count += 1
                    return None
                
                # Update access patterns
                self.access_frequency[key] += 1
                self.access_recency[key] = current_time
                self.access_times[key] = current_time
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                self.hit_count += 1
                self.logger.debug(f"Cache hit for key: {key[:50]}...")
                return item
            
            self.miss_count += 1
            self.logger.debug(f"Cache miss for key: {key[:50]}...")
            return None
    
    def put(self, key: str, value: Any, size_hint: Optional[int] = None) -> None:
        """Put item in cache with intelligent eviction."""
        with self.lock:
            current_time = time.time()
            
            # Calculate size if not provided
            if size_hint is None:
                try:
                    size_hint = len(pickle.dumps(value))
                except Exception:
                    size_hint = 1024  # Default size estimate
            
            # Remove existing item if updating
            if key in self.cache:
                del self.cache[key]
            
            # Evict items if necessary
            while len(self.cache) >= self.max_size:
                self._adaptive_evict()
            
            # Store item
            self.cache[key] = (value, current_time)
            self.access_times[key] = current_time
            self.access_frequency[key] += 1
            self.access_recency[key] = current_time
            self.size_tracking[key] = size_hint
            
            self.logger.debug(f"Cache put for key: {key[:50]}... (size: {size_hint} bytes)")
    
    def _adaptive_evict(self) -> None:
        """Adaptive eviction based on access patterns and size."""
        if not self.cache:
            return
        
        current_time = time.time()
        
        # Calculate eviction scores for all items
        eviction_scores = {}
        
        for key in self.cache:
            # Factors for eviction decision
            age_score = current_time - self.access_times.get(key, current_time)
            frequency_score = 1.0 / (self.access_frequency.get(key, 1) + 1)
            recency_score = current_time - self.access_recency.get(key, current_time)
            size_score = self.size_tracking.get(key, 1024) / 1024.0  # Normalize to KB
            
            # Combined score (higher = more likely to evict)
            eviction_scores[key] = (
                age_score * 0.3 +
                frequency_score * 0.4 +
                recency_score * 0.2 +
                size_score * 0.1
            )
        
        # Evict item with highest score
        key_to_evict = max(eviction_scores, key=eviction_scores.get)
        self._evict_item(key_to_evict)
    
    def _evict_item(self, key: str) -> None:
        """Evict specific item from cache."""
        if key in self.cache:
            del self.cache[key]
            self.access_times.pop(key, None)
            self.access_frequency.pop(key, None)
            self.access_recency.pop(key, None)
            self.size_tracking.pop(key, None)
            self.eviction_count += 1
            self.logger.debug(f"Evicted cache item: {key[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_frequency.clear()
            self.access_recency.clear()
            self.size_tracking.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'eviction_count': self.eviction_count,
                'total_memory_usage': sum(self.size_tracking.values()),
                'average_item_size': np.mean(list(self.size_tracking.values())) if self.size_tracking else 0
            }


class ResourcePool(Generic[T]):
    """Generic resource pool for expensive objects with lifecycle management."""
    
    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: int = 300,
        validator: Optional[Callable[[T], bool]] = None
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.validator = validator or (lambda x: True)
        
        self.pool = queue.Queue(maxsize=max_size)
        self.created_count = 0
        self.borrowed_count = 0
        self.returned_count = 0
        self.destroyed_count = 0
        
        self.resource_timestamps = {}
        self.active_resources = set()
        
        self.lock = threading.Lock()
        self.logger = logging.getLogger("dgdm_histopath.optimization.pool")
        
        # Initialize minimum resources
        self._initialize_pool()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _initialize_pool(self) -> None:
        """Initialize pool with minimum resources."""
        for _ in range(self.min_size):
            try:
                resource = self.factory()
                self.pool.put_nowait(resource)
                self.resource_timestamps[id(resource)] = time.time()
                self.created_count += 1
            except Exception as e:
                self.logger.error(f"Failed to create resource during initialization: {e}")
    
    def borrow(self, timeout: float = 30.0) -> T:
        """Borrow resource from pool."""
        try:
            # Try to get existing resource
            try:
                resource = self.pool.get_nowait()
                resource_id = id(resource)
                
                # Validate resource
                if not self.validator(resource):
                    self.logger.warning("Invalid resource found in pool, creating new one")
                    self._destroy_resource(resource)
                    resource = self.factory()
                    self.created_count += 1
                
                # Track as active
                with self.lock:
                    self.active_resources.add(resource_id)
                    self.resource_timestamps[resource_id] = time.time()
                
                self.borrowed_count += 1
                self.logger.debug(f"Borrowed resource from pool (active: {len(self.active_resources)})")
                return resource
                
            except queue.Empty:
                # Pool is empty, create new resource if under limit
                with self.lock:
                    total_resources = len(self.active_resources) + self.pool.qsize()
                    
                    if total_resources < self.max_size:
                        resource = self.factory()
                        resource_id = id(resource)
                        
                        self.active_resources.add(resource_id)
                        self.resource_timestamps[resource_id] = time.time()
                        self.created_count += 1
                        self.borrowed_count += 1
                        
                        self.logger.debug(f"Created new resource (total: {total_resources + 1})")
                        return resource
                    else:
                        # Wait for resource to become available
                        resource = self.pool.get(timeout=timeout)
                        resource_id = id(resource)
                        
                        if not self.validator(resource):
                            self._destroy_resource(resource)
                            resource = self.factory()
                            self.created_count += 1
                        
                        self.active_resources.add(resource_id)
                        self.resource_timestamps[resource_id] = time.time()
                        self.borrowed_count += 1
                        
                        return resource
                        
        except Exception as e:
            self.logger.error(f"Failed to borrow resource: {e}")
            raise ResourceError(f"Failed to borrow resource: {str(e)}")
    
    def return_resource(self, resource: T) -> None:
        """Return resource to pool."""
        try:
            resource_id = id(resource)
            
            with self.lock:
                if resource_id not in self.active_resources:
                    self.logger.warning("Attempting to return unknown resource")
                    return
                
                self.active_resources.remove(resource_id)
            
            # Validate before returning
            if self.validator(resource):
                try:
                    self.pool.put_nowait(resource)
                    self.resource_timestamps[resource_id] = time.time()
                    self.returned_count += 1
                    self.logger.debug(f"Returned resource to pool (pool size: {self.pool.qsize()})")
                except queue.Full:
                    # Pool is full, destroy resource
                    self._destroy_resource(resource)
            else:
                self.logger.warning("Invalid resource returned, destroying")
                self._destroy_resource(resource)
                
        except Exception as e:
            self.logger.error(f"Failed to return resource: {e}")
    
    def _destroy_resource(self, resource: T) -> None:
        """Safely destroy resource."""
        try:
            resource_id = id(resource)
            
            # Call cleanup method if available
            if hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, 'cleanup'):
                resource.cleanup()
            
            # Remove from tracking
            with self.lock:
                self.resource_timestamps.pop(resource_id, None)
                self.active_resources.discard(resource_id)
            
            self.destroyed_count += 1
            self.logger.debug("Destroyed resource")
            
        except Exception as e:
            self.logger.error(f"Error destroying resource: {e}")
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of idle resources."""
        while True:
            try:
                time.sleep(60)  # Run cleanup every minute
                current_time = time.time()
                
                # Check for idle resources in pool
                idle_resources = []
                temp_queue = queue.Queue()
                
                # Empty pool and check each resource
                while True:
                    try:
                        resource = self.pool.get_nowait()
                        resource_id = id(resource)
                        
                        if (current_time - self.resource_timestamps.get(resource_id, current_time)) > self.max_idle_time:
                            idle_resources.append(resource)
                        else:
                            temp_queue.put(resource)
                            
                    except queue.Empty:
                        break
                
                # Put back non-idle resources
                while not temp_queue.empty():
                    try:
                        self.pool.put_nowait(temp_queue.get_nowait())
                    except queue.Full:
                        break
                
                # Destroy idle resources (keeping minimum)
                current_pool_size = self.pool.qsize()
                for resource in idle_resources:
                    if current_pool_size > self.min_size:
                        self._destroy_resource(resource)
                        current_pool_size -= 1
                    else:
                        # Keep resource, put back in pool
                        try:
                            self.pool.put_nowait(resource)
                        except queue.Full:
                            self._destroy_resource(resource)
                
                if idle_resources:
                    self.logger.info(f"Cleaned up {len(idle_resources)} idle resources")
                    
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    @contextmanager
    def resource(self):
        """Context manager for borrowing/returning resources."""
        resource = self.borrow()
        try:
            yield resource
        finally:
            self.return_resource(resource)
    
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                'pool_size': self.pool.qsize(),
                'active_count': len(self.active_resources),
                'max_size': self.max_size,
                'min_size': self.min_size,
                'created_count': self.created_count,
                'borrowed_count': self.borrowed_count,
                'returned_count': self.returned_count,
                'destroyed_count': self.destroyed_count,
                'utilization': len(self.active_resources) / self.max_size
            }


class ParallelProcessor:
    """Advanced parallel processing with dynamic worker allocation."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        worker_type: str = 'thread',  # 'thread', 'process', 'auto'
        adaptive_scaling: bool = True
    ):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.worker_type = worker_type
        self.adaptive_scaling = adaptive_scaling
        
        self.executor = None
        self.current_workers = 0
        self.task_queue = queue.Queue()
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        self.performance_history = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger("dgdm_histopath.optimization.parallel")
        
        # Adaptive scaling parameters
        self.cpu_threshold = 0.8
        self.memory_threshold = 0.85
        self.scale_up_threshold = 0.9
        self.scale_down_threshold = 0.3
        
        self._initialize_executor()
    
    def _initialize_executor(self) -> None:
        """Initialize executor based on configuration."""
        if self.worker_type == 'auto':
            # Auto-select based on system resources
            cpu_count = os.cpu_count() or 1
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if memory_gb > 16 and cpu_count >= 8:
                self.worker_type = 'process'
            else:
                self.worker_type = 'thread'
        
        initial_workers = min(4, self.max_workers)
        
        if self.worker_type == 'process':
            self.executor = ProcessPoolExecutor(max_workers=initial_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=initial_workers)
        
        self.current_workers = initial_workers
        self.logger.info(f"Initialized {self.worker_type} executor with {initial_workers} workers")
    
    def submit_batch(
        self,
        func: Callable,
        items: List[Any],
        chunk_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """Submit batch of tasks with automatic chunking and progress tracking."""
        
        if not items:
            return []
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.current_workers * 2))
        
        # Create chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Submit tasks
        start_time = time.time()
        future_to_chunk = {}
        
        try:
            for i, chunk in enumerate(chunks):
                if len(chunk) == 1:
                    # Single item
                    future = self.executor.submit(func, chunk[0])
                else:
                    # Batch processing
                    future = self.executor.submit(self._process_chunk, func, chunk)
                
                future_to_chunk[future] = i
            
            # Collect results
            results = [None] * len(chunks)
            completed_chunks = 0
            
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per chunk
                    results[chunk_index] = result
                    self.completed_tasks += 1
                    
                except Exception as e:
                    self.logger.error(f"Chunk {chunk_index} failed: {e}")
                    results[chunk_index] = None
                    self.failed_tasks += 1
                
                completed_chunks += 1
                
                if progress_callback:
                    progress_callback(completed_chunks, len(chunks))
                
                # Adaptive scaling check
                if self.adaptive_scaling and completed_chunks % 5 == 0:
                    self._check_scaling()
            
            # Flatten results
            flattened_results = []
            for result in results:
                if isinstance(result, list):
                    flattened_results.extend(result)
                elif result is not None:
                    flattened_results.append(result)
            
            duration = time.time() - start_time
            throughput = len(items) / duration if duration > 0 else 0
            
            # Record performance
            self._record_performance(len(items), duration, throughput)
            
            self.logger.info(
                f"Processed {len(items)} items in {duration:.2f}s "
                f"({throughput:.1f} items/s, {self.failed_tasks} failures)"
            )
            
            return flattened_results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise PerformanceError(f"Batch processing failed: {str(e)}")
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        results = []
        for item in chunk:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Item processing failed: {e}")
                results.append(None)
        return results
    
    def _check_scaling(self) -> None:
        """Check if executor should be scaled up or down."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Check queue depth
            queue_depth = self.task_queue.qsize()
            
            # Scaling decisions
            should_scale_up = (
                cpu_percent < self.cpu_threshold * 100 and
                memory_percent < self.memory_threshold * 100 and
                queue_depth > self.current_workers * 2 and
                self.current_workers < self.max_workers
            )
            
            should_scale_down = (
                (cpu_percent > self.scale_up_threshold * 100 or 
                 memory_percent > self.memory_threshold * 100 or
                 queue_depth < self.current_workers * self.scale_down_threshold) and
                self.current_workers > 2
            )
            
            if should_scale_up:
                self._scale_up()
            elif should_scale_down:
                self._scale_down()
                
        except Exception as e:
            self.logger.error(f"Scaling check failed: {e}")
    
    def _scale_up(self) -> None:
        """Scale up worker count."""
        new_workers = min(self.current_workers + 2, self.max_workers)
        if new_workers > self.current_workers:
            # Recreate executor with more workers
            old_executor = self.executor
            
            if self.worker_type == 'process':
                self.executor = ProcessPoolExecutor(max_workers=new_workers)
            else:
                self.executor = ThreadPoolExecutor(max_workers=new_workers)
            
            # Shutdown old executor
            old_executor.shutdown(wait=False)
            
            self.current_workers = new_workers
            self.logger.info(f"Scaled up to {new_workers} workers")
    
    def _scale_down(self) -> None:
        """Scale down worker count."""
        new_workers = max(self.current_workers - 1, 2)
        if new_workers < self.current_workers:
            # Recreate executor with fewer workers
            old_executor = self.executor
            
            if self.worker_type == 'process':
                self.executor = ProcessPoolExecutor(max_workers=new_workers)
            else:
                self.executor = ThreadPoolExecutor(max_workers=new_workers)
            
            # Shutdown old executor
            old_executor.shutdown(wait=False)
            
            self.current_workers = new_workers
            self.logger.info(f"Scaled down to {new_workers} workers")
    
    def _record_performance(self, item_count: int, duration: float, throughput: float) -> None:
        """Record performance metrics."""
        with self.lock:
            perf_data = {
                'timestamp': datetime.now(),
                'item_count': item_count,
                'duration': duration,
                'throughput': throughput,
                'worker_count': self.current_workers,
                'worker_type': self.worker_type
            }
            
            self.performance_history.append(perf_data)
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Report to metrics collector
            metrics_collector.record_custom_metric(
                'parallel_processing_throughput',
                throughput,
                tags={'worker_type': self.worker_type, 'worker_count': str(self.current_workers)}
            )
    
    def shutdown(self) -> None:
        """Shutdown executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.info("Parallel processor shutdown complete")
    
    def stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self.lock:
            recent_perf = self.performance_history[-10:] if self.performance_history else []
            avg_throughput = np.mean([p['throughput'] for p in recent_perf]) if recent_perf else 0
            
            return {
                'current_workers': self.current_workers,
                'max_workers': self.max_workers,
                'worker_type': self.worker_type,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate': self.completed_tasks / (self.completed_tasks + self.failed_tasks) if (self.completed_tasks + self.failed_tasks) > 0 else 0,
                'average_throughput': avg_throughput,
                'adaptive_scaling': self.adaptive_scaling
            }


class MemoryOptimizer:
    """Memory optimization and garbage collection management."""
    
    def __init__(self):
        self.logger = logging.getLogger("dgdm_histopath.optimization.memory")
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.cleanup_callbacks = []
        
        # Start memory monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add callback for memory cleanup."""
        self.cleanup_callbacks.append(callback)
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization."""
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Run garbage collection
        collected_objects = []
        for generation in range(3):
            collected = gc.collect(generation)
            collected_objects.append(collected)
            
        # Clear PyTorch cache if available
        if TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                self.logger.debug("Cleared PyTorch CUDA cache")
            except Exception:
                pass
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Cleanup callback failed: {e}")
        
        # Force garbage collection
        gc.collect()
        
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_freed = start_memory - end_memory
        
        optimization_stats = {
            'memory_before_mb': start_memory,
            'memory_after_mb': end_memory,
            'memory_freed_mb': memory_freed,
            'gc_collected': sum(collected_objects),
            'gc_by_generation': collected_objects,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Memory optimization: freed {memory_freed:.1f}MB, collected {sum(collected_objects)} objects")
        return optimization_stats
    
    def _monitor_loop(self) -> None:
        """Monitor memory usage and trigger optimization."""
        while self.monitoring:
            try:
                memory_percent = psutil.virtual_memory().percent / 100
                
                if memory_percent > self.memory_threshold:
                    self.logger.warning(f"High memory usage: {memory_percent:.1%}, running optimization")
                    self.optimize_memory()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(60)
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False


# Performance decorators
def cached(cache: AdaptiveCache, key_func: Optional[Callable] = None, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result)
            
            return result
        return wrapper
    return decorator


def profile_performance(track_memory: bool = True, track_gpu: bool = True):
    """Decorator for performance profiling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Record start state
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss if track_memory else 0
            start_gpu_memory = 0
            
            if track_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    start_gpu_memory = torch.cuda.memory_allocated()
                except Exception:
                    pass
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                
                # Record end state
                end_time = time.time()
                duration = end_time - start_time
                
                end_memory = psutil.Process().memory_info().rss if track_memory else 0
                memory_delta = end_memory - start_memory
                
                gpu_memory_delta = 0
                if track_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        gpu_memory_delta = torch.cuda.memory_allocated() - start_gpu_memory
                    except Exception:
                        pass
                
                # Create performance profile
                profile = PerformanceProfile(
                    operation_name=func.__name__,
                    duration=duration,
                    memory_usage=memory_delta,
                    cpu_usage=psutil.cpu_percent(),
                    gpu_usage=gpu_memory_delta / 1024 / 1024 if gpu_memory_delta > 0 else None,
                    cache_hits=0,  # Would need cache reference
                    cache_misses=0,
                    throughput=1 / duration if duration > 0 else 0,
                    timestamp=datetime.now()
                )
                
                # Log performance
                logging.getLogger("dgdm_histopath.optimization.profiler").info(
                    f"Performance profile: {func.__name__} took {duration:.3f}s, "
                    f"memory: {memory_delta/1024/1024:.1f}MB"
                )
                
                # Record metrics
                metrics_collector.record_custom_metric(
                    f'function_duration_{func.__name__}',
                    duration,
                    tags={'function': func.__name__}
                )
                
                return result
                
            except Exception as e:
                # Record failed operation
                duration = time.time() - start_time
                logging.getLogger("dgdm_histopath.optimization.profiler").error(
                    f"Function {func.__name__} failed after {duration:.3f}s: {e}"
                )
                raise
                
        return wrapper
    return decorator


@contextmanager
def memory_limit(max_memory_mb: int):
    """Context manager to enforce memory limits."""
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = current_memory - start_memory
        
        if memory_used > max_memory_mb:
            raise ResourceError(
                f"Memory limit exceeded: used {memory_used:.1f}MB, limit {max_memory_mb}MB"
            )


# Global optimization instances
global_cache = AdaptiveCache(max_size=10000, ttl_seconds=3600)
global_parallel_processor = ParallelProcessor(adaptive_scaling=True)
global_memory_optimizer = MemoryOptimizer()


# Utility functions for common optimization patterns
def optimize_numpy_operations():
    """Configure NumPy for optimal performance."""
    try:
        import numpy as np
        
        # Set optimal thread count for NumPy operations
        cpu_count = os.cpu_count() or 1
        optimal_threads = min(cpu_count, 8)  # Don't use more than 8 threads for NumPy
        
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
        
        logging.getLogger("dgdm_histopath.optimization").info(
            f"Configured NumPy to use {optimal_threads} threads"
        )
        
    except ImportError:
        pass


def optimize_torch_settings():
    """Configure PyTorch for optimal performance."""
    if not TORCH_AVAILABLE:
        return
    
    try:
        # Set optimal thread count
        cpu_count = os.cpu_count() or 1
        optimal_threads = min(cpu_count, 8)
        
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(optimal_threads)
        
        # Enable optimized kernels
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable memory format optimization
        torch.backends.cudnn.allow_tf32 = True
        
        logging.getLogger("dgdm_histopath.optimization").info(
            f"Configured PyTorch with {optimal_threads} threads and optimized settings"
        )
        
    except Exception as e:
        logging.getLogger("dgdm_histopath.optimization").warning(
            f"Failed to optimize PyTorch settings: {e}"
        )


# Initialize optimizations
optimize_numpy_operations()
optimize_torch_settings()