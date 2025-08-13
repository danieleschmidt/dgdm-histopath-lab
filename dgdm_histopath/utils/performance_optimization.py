"""
Advanced Performance Optimization Framework

High-performance computing optimization for medical AI systems with
GPU acceleration, memory optimization, and intelligent caching.
"""

import time
import threading
import multiprocessing
import queue
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import warnings
from pathlib import Path
import pickle
import hashlib
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from dgdm_histopath.utils.exceptions import DGDMException
from dgdm_histopath.utils.monitoring import record_metric, MetricType, MonitoringScope


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"
    CUSTOM = "custom"


class CacheStrategy(Enum):
    """Caching strategies for different scenarios."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    execution_time: float
    memory_usage: int
    gpu_utilization: float
    cache_hit_rate: float
    throughput: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE
    enable_gpu_acceleration: bool = True
    enable_mixed_precision: bool = True
    enable_memory_pooling: bool = True
    enable_intelligent_caching: bool = True
    cache_size_mb: int = 1024
    batch_size_optimization: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None


class IntelligentCache:
    """
    High-performance intelligent caching system with multiple strategies,
    automatic eviction, and memory management.
    """
    
    def __init__(
        self,
        max_size_mb: int = 1024,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        ttl_seconds: Optional[int] = None,
        enable_compression: bool = True
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression
        
        # Cache storage
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.creation_times = {}
        self.sizes = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL expiration
            if self.ttl_seconds and key in self.creation_times:
                age = time.time() - self.creation_times[key]
                if age > self.ttl_seconds:
                    self._remove_item(key)
                    self.misses += 1
                    return None
            
            # Update access statistics
            self.access_times[key] = time.time()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.hits += 1
            
            value = self.cache[key]
            
            # Decompress if needed
            if self.enable_compression and isinstance(value, bytes):
                try:
                    value = pickle.loads(value)
                except:
                    pass  # Not compressed
            
            return value
    
    def put(self, key: str, value: Any) -> bool:
        """Store item in cache."""
        with self._lock:
            # Serialize and optionally compress
            if self.enable_compression:
                try:
                    serialized_value = pickle.dumps(value)
                    size = len(serialized_value)
                except:
                    serialized_value = value
                    size = self._estimate_size(value)
            else:
                serialized_value = value
                size = self._estimate_size(value)
            
            # Check if item is too large
            if size > self.max_size_bytes:
                self.logger.warning(f"Cache item too large: {key} ({size} bytes)")
                return False
            
            # Evict items if necessary
            while self._get_total_size() + size > self.max_size_bytes:
                if not self._evict_one_item():
                    break  # No more items to evict
            
            # Store item
            self.cache[key] = serialized_value
            self.sizes[key] = size
            self.creation_times[key] = time.time()
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """Remove specific item from cache."""
        with self._lock:
            if key in self.cache:
                self._remove_item(key)
                return True
            return False
    
    def clear(self):
        """Clear entire cache."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.creation_times.clear()
            self.sizes.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "total_size_mb": self._get_total_size() / (1024 * 1024),
                "item_count": len(self.cache),
                "max_size_mb": self.max_size_bytes / (1024 * 1024)
            }
    
    def _evict_one_item(self) -> bool:
        """Evict one item based on strategy."""
        if not self.cache:
            return False
        
        if self.strategy == CacheStrategy.LRU:
            # Least recently used
            oldest_key = min(self.access_times, key=self.access_times.get)
        elif self.strategy == CacheStrategy.LFU:
            # Least frequently used
            oldest_key = min(self.access_counts, key=self.access_counts.get)
        elif self.strategy == CacheStrategy.TTL:
            # Oldest creation time
            oldest_key = min(self.creation_times, key=self.creation_times.get)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy considering both frequency and recency
            current_time = time.time()
            scores = {}
            for key in self.cache:
                recency_score = current_time - self.access_times[key]
                frequency_score = 1.0 / (self.access_counts[key] + 1)
                scores[key] = recency_score * frequency_score
            oldest_key = max(scores, key=scores.get)
        else:
            # Default to LRU
            oldest_key = min(self.access_times, key=self.access_times.get)
        
        self._remove_item(oldest_key)
        self.evictions += 1
        return True
    
    def _remove_item(self, key: str):
        """Remove item and all its metadata."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
        self.creation_times.pop(key, None)
        self.sizes.pop(key, None)
    
    def _get_total_size(self) -> int:
        """Get total size of cached items."""
        return sum(self.sizes.values())
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate size of object in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            else:
                return 1024  # Default estimate


class GPUOptimizer:
    """
    GPU optimization and acceleration manager with memory pooling,
    mixed precision, and intelligent batch sizing.
    """
    
    def __init__(
        self,
        enable_mixed_precision: bool = True,
        enable_memory_pooling: bool = True,
        memory_fraction: float = 0.9
    ):
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_memory_pooling = enable_memory_pooling
        self.memory_fraction = memory_fraction
        
        self.device = None
        self.memory_pool = None
        self.scaler = None
        
        if TORCH_AVAILABLE:
            self._initialize_gpu()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_gpu(self):
        """Initialize GPU optimization settings."""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, GPU optimization disabled")
            return
        
        # Select best device
        self.device = torch.device("cuda")
        
        # Set memory fraction
        if self.memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
        
        # Enable memory pooling
        if self.enable_memory_pooling:
            torch.cuda.empty_cache()
        
        # Initialize mixed precision scaler
        if self.enable_mixed_precision:
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
            except ImportError:
                self.logger.warning("Mixed precision not available")
                self.enable_mixed_precision = False
        
        self.logger.info(f"GPU optimization initialized on {torch.cuda.get_device_name()}")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimize model for GPU acceleration."""
        if not TORCH_AVAILABLE or self.device is None:
            return model
        
        # Move to GPU
        model = model.to(self.device)
        
        # Enable mixed precision if available
        if self.enable_mixed_precision:
            model = model.half()
        
        # Optimize for inference
        model.eval()
        
        # Compile model for optimization (PyTorch 2.0+)
        try:
            model = torch.compile(model)
            self.logger.info("Model compiled for optimization")
        except:
            pass  # Compilation not available
        
        return model
    
    def optimize_batch_size(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        max_batch_size: int = 64,
        memory_threshold: float = 0.8
    ) -> int:
        """Find optimal batch size for GPU memory."""
        if not TORCH_AVAILABLE or self.device is None:
            return 1
        
        model.eval()
        best_batch_size = 1
        
        for batch_size in [2**i for i in range(int(np.log2(max_batch_size)) + 1)]:
            try:
                # Create batch
                batch_input = sample_input.repeat(batch_size, 1, 1, 1)
                batch_input = batch_input.to(self.device)
                
                # Test forward pass
                with torch.no_grad():
                    _ = model(batch_input)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                
                if memory_used < memory_threshold:
                    best_batch_size = batch_size
                else:
                    break
                
                # Clean up
                del batch_input
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                break
            except Exception as e:
                self.logger.warning(f"Error testing batch size {batch_size}: {e}")
                break
        
        self.logger.info(f"Optimal batch size: {best_batch_size}")
        return best_batch_size
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU utilization statistics."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}
        
        stats = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "memory_cached_mb": torch.cuda.memory_reserved() / (1024**2),
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        }
        
        # GPU utilization (requires nvidia-ml-py)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats["gpu_utilization"] = util.gpu
            stats["memory_utilization"] = util.memory
        except:
            pass
        
        return stats


class MemoryOptimizer:
    """
    Advanced memory optimization with automatic garbage collection,
    memory pooling, and efficient data structures.
    """
    
    def __init__(
        self,
        enable_aggressive_gc: bool = True,
        memory_threshold: float = 0.8,
        monitoring_interval: float = 60.0
    ):
        self.enable_aggressive_gc = enable_aggressive_gc
        self.memory_threshold = memory_threshold
        self.monitoring_interval = monitoring_interval
        
        # Memory monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Memory pool
        self.memory_pool = {}
        self.pool_lock = threading.Lock()
        
        if enable_aggressive_gc:
            gc.set_threshold(700, 10, 10)  # More aggressive GC
        
        self.logger = logging.getLogger(__name__)
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start memory monitoring thread."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
    
    def _monitoring_worker(self):
        """Background memory monitoring worker."""
        while self._monitoring_active:
            try:
                memory_percent = psutil.virtual_memory().percent / 100.0
                
                # Record metrics
                record_metric(
                    "memory_usage_percent",
                    memory_percent * 100,
                    MetricType.GAUGE,
                    MonitoringScope.SYSTEM
                )
                
                # Trigger cleanup if needed
                if memory_percent > self.memory_threshold:
                    self._emergency_cleanup()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                time.sleep(60)
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup."""
        self.logger.warning("Memory threshold exceeded, performing emergency cleanup")
        
        # Clear memory pool
        with self.pool_lock:
            self.memory_pool.clear()
        
        # Force garbage collection
        if self.enable_aggressive_gc:
            collected = gc.collect()
            self.logger.info(f"Garbage collection freed {collected} objects")
        
        # Clear GPU cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_pool(self, pool_name: str, item_size: int, pool_size: int) -> List[Any]:
        """Get or create memory pool for reusable objects."""
        with self.pool_lock:
            if pool_name not in self.memory_pool:
                self.memory_pool[pool_name] = {
                    "items": [],
                    "item_size": item_size,
                    "max_size": pool_size,
                    "created": datetime.now()
                }
            
            return self.memory_pool[pool_name]["items"]
    
    def return_to_pool(self, pool_name: str, item: Any):
        """Return item to memory pool for reuse."""
        with self.pool_lock:
            if pool_name in self.memory_pool:
                pool = self.memory_pool[pool_name]
                if len(pool["items"]) < pool["max_size"]:
                    pool["items"].append(item)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        
        stats = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent,
            "gc_collections": gc.get_count(),
        }
        
        # Pool statistics
        with self.pool_lock:
            pool_stats = {}
            for name, pool in self.memory_pool.items():
                pool_stats[name] = {
                    "items": len(pool["items"]),
                    "max_size": pool["max_size"],
                    "item_size": pool["item_size"]
                }
            stats["memory_pools"] = pool_stats
        
        return stats


class ParallelProcessor:
    """
    High-performance parallel processing with intelligent work distribution,
    load balancing, and resource management.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        enable_load_balancing: bool = True
    ):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.enable_load_balancing = enable_load_balancing
        
        # Work queue
        self.work_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Worker management
        self.workers = []
        self.worker_stats = {}
        self.active = False
        
        self.logger = logging.getLogger(__name__)
    
    def start_workers(self):
        """Start worker threads/processes."""
        if self.active:
            return
        
        self.active = True
        
        if self.use_processes:
            worker_class = multiprocessing.Process
        else:
            worker_class = threading.Thread
        
        for i in range(self.max_workers):
            worker = worker_class(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            self.worker_stats[i] = {
                "tasks_completed": 0,
                "total_time": 0.0,
                "last_activity": time.time()
            }
        
        self.logger.info(f"Started {self.max_workers} workers")
    
    def stop_workers(self):
        """Stop all workers."""
        if not self.active:
            return
        
        self.active = False
        
        # Send stop signals
        for _ in self.workers:
            self.work_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            if hasattr(worker, 'join'):
                worker.join(timeout=5.0)
        
        self.workers.clear()
        self.logger.info("Stopped all workers")
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        while self.active:
            try:
                task = self.work_queue.get(timeout=1.0)
                if task is None:  # Stop signal
                    break
                
                start_time = time.time()
                
                # Execute task
                func, args, kwargs, task_id = task
                try:
                    result = func(*args, **kwargs)
                    self.result_queue.put((task_id, result, None))
                except Exception as e:
                    self.result_queue.put((task_id, None, e))
                
                # Update statistics
                execution_time = time.time() - start_time
                self.worker_stats[worker_id]["tasks_completed"] += 1
                self.worker_stats[worker_id]["total_time"] += execution_time
                self.worker_stats[worker_id]["last_activity"] = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in worker {worker_id}: {e}")
    
    def submit_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit task for parallel execution."""
        if not self.active:
            self.start_workers()
        
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}"
        
        task = (func, args, kwargs, task_id)
        self.work_queue.put(task)
        
        return task_id
    
    def get_result(self, timeout: Optional[float] = None) -> Tuple[str, Any, Optional[Exception]]:
        """Get result from completed task."""
        return self.result_queue.get(timeout=timeout)
    
    def process_batch(
        self,
        func: Callable,
        items: List[Any],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """Process batch of items in parallel."""
        if not items:
            return []
        
        if batch_size is None:
            batch_size = max(1, len(items) // self.max_workers)
        
        # Split into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Submit batch tasks
        task_ids = []
        for batch in batches:
            task_id = self.submit_task(self._process_batch_worker, func, batch, **kwargs)
            task_ids.append(task_id)
        
        # Collect results
        results = {}
        for _ in task_ids:
            task_id, result, error = self.get_result()
            if error:
                raise error
            results[task_id] = result
        
        # Combine results in order
        combined_results = []
        for task_id in task_ids:
            combined_results.extend(results[task_id])
        
        return combined_results
    
    def _process_batch_worker(self, func: Callable, batch: List[Any], **kwargs) -> List[Any]:
        """Worker function for batch processing."""
        return [func(item, **kwargs) for item in batch]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parallel processing statistics."""
        total_tasks = sum(stats["tasks_completed"] for stats in self.worker_stats.values())
        total_time = sum(stats["total_time"] for stats in self.worker_stats.values())
        
        return {
            "active_workers": len(self.workers),
            "max_workers": self.max_workers,
            "total_tasks_completed": total_tasks,
            "total_execution_time": total_time,
            "queue_size": self.work_queue.qsize(),
            "pending_results": self.result_queue.qsize(),
            "worker_stats": self.worker_stats
        }


class PerformanceOptimizer:
    """
    Main performance optimization framework that coordinates all optimization
    components for maximum system performance.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        # Initialize optimization components
        self.cache = IntelligentCache(
            max_size_mb=config.cache_size_mb,
            strategy=CacheStrategy.ADAPTIVE,
            enable_compression=True
        ) if config.enable_intelligent_caching else None
        
        self.gpu_optimizer = GPUOptimizer(
            enable_mixed_precision=config.enable_mixed_precision,
            enable_memory_pooling=config.enable_memory_pooling
        ) if config.enable_gpu_acceleration else None
        
        self.memory_optimizer = MemoryOptimizer(
            enable_aggressive_gc=True,
            memory_threshold=0.8
        )
        
        self.parallel_processor = ParallelProcessor(
            max_workers=config.max_workers,
            use_processes=False,
            enable_load_balancing=True
        ) if config.parallel_processing else None
        
        # Performance tracking
        self.performance_history = []
        self.optimization_active = True
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Performance optimizer initialized: {config.optimization_level.value}")
    
    def optimize_function(
        self,
        func: Callable,
        cache_key: Optional[str] = None,
        enable_parallel: bool = False,
        enable_gpu: bool = False
    ) -> Callable:
        """Decorator to optimize function performance."""
        def optimized_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Check cache first
            if self.cache and cache_key:
                full_cache_key = f"{cache_key}_{hashlib.md5(str(args).encode()).hexdigest()}"
                cached_result = self.cache.get(full_cache_key)
                if cached_result is not None:
                    execution_time = time.time() - start_time
                    self._record_performance("cache_hit", execution_time)
                    return cached_result
            
            # Execute function
            if enable_parallel and self.parallel_processor:
                # Submit to parallel processor
                task_id = self.parallel_processor.submit_task(func, *args, **kwargs)
                _, result, error = self.parallel_processor.get_result()
                if error:
                    raise error
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            if self.cache and cache_key:
                self.cache.put(full_cache_key, result)
            
            execution_time = time.time() - start_time
            self._record_performance(func.__name__, execution_time)
            
            return result
        
        return optimized_wrapper
    
    def optimize_model_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> Tuple[nn.Module, int]:
        """Optimize model for inference performance."""
        if not self.gpu_optimizer:
            return model, batch_size or 1
        
        # Optimize model
        optimized_model = self.gpu_optimizer.optimize_model(model)
        
        # Find optimal batch size
        if batch_size is None:
            batch_size = self.gpu_optimizer.optimize_batch_size(
                optimized_model, sample_input
            )
        
        return optimized_model, batch_size
    
    def process_large_dataset(
        self,
        dataset: List[Any],
        processing_func: Callable,
        batch_size: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> List[Any]:
        """Optimized processing of large datasets."""
        if not dataset:
            return []
        
        # Determine optimal batch size
        if batch_size is None:
            batch_size = max(1, len(dataset) // (self.config.max_workers or 4))
        
        # Process with caching and parallelization
        results = []
        
        if self.parallel_processor:
            # Parallel processing
            results = self.parallel_processor.process_batch(
                processing_func, dataset, batch_size, **kwargs
            )
        else:
            # Sequential processing with optimization
            for item in dataset:
                if use_cache and self.cache:
                    cache_key = f"dataset_{hashlib.md5(str(item).encode()).hexdigest()}"
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        results.append(cached_result)
                        continue
                
                result = processing_func(item, **kwargs)
                results.append(result)
                
                if use_cache and self.cache:
                    self.cache.put(cache_key, result)
        
        return results
    
    def _record_performance(self, operation: str, execution_time: float):
        """Record performance metrics."""
        # Get system metrics
        memory_stats = self.memory_optimizer.get_memory_stats()
        gpu_stats = self.gpu_optimizer.get_gpu_stats() if self.gpu_optimizer else {}
        cache_stats = self.cache.get_stats() if self.cache else {}
        
        metrics = PerformanceMetrics(
            operation_name=operation,
            execution_time=execution_time,
            memory_usage=int(memory_stats.get("used_gb", 0) * 1024),
            gpu_utilization=gpu_stats.get("gpu_utilization", 0.0),
            cache_hit_rate=cache_stats.get("hit_rate", 0.0),
            throughput=1.0 / execution_time if execution_time > 0 else 0.0,
            metadata={
                "memory_stats": memory_stats,
                "gpu_stats": gpu_stats,
                "cache_stats": cache_stats
            }
        )
        
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # Record to monitoring system
        record_metric(f"execution_time_{operation}", execution_time, MetricType.TIMER, MonitoringScope.PERFORMANCE)
        record_metric("memory_usage_mb", metrics.memory_usage, MetricType.GAUGE, MonitoringScope.SYSTEM)
        record_metric("cache_hit_rate", metrics.cache_hit_rate, MetricType.GAUGE, MonitoringScope.PERFORMANCE)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {}
        
        # Calculate aggregated metrics
        execution_times = [m.execution_time for m in self.performance_history]
        memory_usage = [m.memory_usage for m in self.performance_history]
        throughput = [m.throughput for m in self.performance_history]
        
        summary = {
            "total_operations": len(self.performance_history),
            "avg_execution_time": np.mean(execution_times) if NUMPY_AVAILABLE else sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "avg_memory_usage_mb": np.mean(memory_usage) if NUMPY_AVAILABLE else sum(memory_usage) / len(memory_usage),
            "avg_throughput": np.mean(throughput) if NUMPY_AVAILABLE else sum(throughput) / len(throughput),
            "cache_stats": self.cache.get_stats() if self.cache else {},
            "memory_stats": self.memory_optimizer.get_memory_stats(),
            "gpu_stats": self.gpu_optimizer.get_gpu_stats() if self.gpu_optimizer else {},
            "parallel_stats": self.parallel_processor.get_stats() if self.parallel_processor else {}
        }
        
        return summary
    
    def shutdown(self):
        """Shutdown optimization framework."""
        self.optimization_active = False
        
        if self.parallel_processor:
            self.parallel_processor.stop_workers()
        
        if self.memory_optimizer:
            self.memory_optimizer.stop_monitoring()
        
        self.logger.info("Performance optimizer shutdown complete")


# Global optimizer instance
_global_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        config = OptimizationConfig(optimization_level=OptimizationLevel.AGGRESSIVE)
        _global_optimizer = PerformanceOptimizer(config)
    return _global_optimizer


def optimize_function(cache_key: Optional[str] = None, **kwargs):
    """Decorator for function optimization."""
    def decorator(func):
        optimizer = get_performance_optimizer()
        return optimizer.optimize_function(func, cache_key=cache_key, **kwargs)
    return decorator


# Example usage and testing
if __name__ == "__main__":
    print("Performance Optimization Framework Loaded")
    print("Optimization capabilities:")
    print("- Intelligent caching with adaptive eviction strategies")
    print("- GPU acceleration with memory optimization")
    print("- Advanced memory management with monitoring")
    print("- High-performance parallel processing")
    print("- Comprehensive performance tracking and analytics")