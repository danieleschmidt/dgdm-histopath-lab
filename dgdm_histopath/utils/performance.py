"""Performance optimization utilities and caching systems."""

import time
import threading
import multiprocessing as mp
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import logging
import pickle
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps, lru_cache
import weakref
import gc
import psutil
import torch
import numpy as np
from collections import OrderedDict, defaultdict


class PerformanceOptimizer:
    """System performance optimization and resource management."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cpu_count = mp.cpu_count()
        self.memory_info = psutil.virtual_memory()
        
        # Performance settings
        self.torch_optimizations = {
            'num_threads': min(self.cpu_count, 8),
            'benchmark': True,
            'deterministic': False,
            'use_flash_attention': True
        }
        
        self.apply_torch_optimizations()
    
    def apply_torch_optimizations(self):
        """Apply PyTorch performance optimizations."""
        try:
            # Set optimal thread count
            torch.set_num_threads(self.torch_optimizations['num_threads'])
            
            # Enable benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = self.torch_optimizations['benchmark']
            
            # Disable deterministic for better performance (unless needed for reproducibility)
            torch.backends.cudnn.deterministic = self.torch_optimizations['deterministic']
            
            # Enable tensor core usage
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
            self.logger.info("Applied PyTorch performance optimizations")
            
        except Exception as e:
            self.logger.warning(f"Failed to apply some PyTorch optimizations: {e}")
    
    def optimize_dataloader_settings(self, dataset_size: int, batch_size: int) -> Dict[str, Any]:
        """Calculate optimal dataloader settings."""
        
        # Calculate optimal number of workers
        if dataset_size < 1000:
            num_workers = 2
        elif dataset_size < 10000:
            num_workers = min(4, self.cpu_count // 2)
        else:
            num_workers = min(8, self.cpu_count)
        
        # Adjust for memory constraints
        available_memory_gb = self.memory_info.available / (1024**3)
        if available_memory_gb < 8:
            num_workers = max(1, num_workers // 2)
        
        # Pin memory settings
        pin_memory = torch.cuda.is_available() and available_memory_gb > 4
        
        # Prefetch factor
        prefetch_factor = 2 if num_workers > 0 else None
        
        settings = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': num_workers > 0,
            'batch_size': self._optimize_batch_size(batch_size, available_memory_gb)
        }
        
        self.logger.info(f"Optimized dataloader settings: {settings}")
        return settings
    
    def _optimize_batch_size(self, requested_batch_size: int, available_memory_gb: float) -> int:
        """Optimize batch size based on available memory."""
        
        if not torch.cuda.is_available():
            return requested_batch_size
            
        try:
            # Get GPU memory info
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Estimate memory usage per sample (rough heuristic)
            estimated_memory_per_sample_mb = 50  # Conservative estimate
            
            # Calculate maximum batch size that fits in GPU memory
            max_batch_size = int((gpu_memory_gb * 1024 * 0.7) / estimated_memory_per_sample_mb)  # 70% utilization
            
            optimized_batch_size = min(requested_batch_size, max_batch_size)
            
            if optimized_batch_size != requested_batch_size:
                self.logger.warning(f"Reduced batch size from {requested_batch_size} to {optimized_batch_size} due to memory constraints")
                
            return max(1, optimized_batch_size)
            
        except Exception as e:
            self.logger.warning(f"Failed to optimize batch size: {e}")
            return requested_batch_size


class AdvancedCache:
    """Advanced caching system with LRU, TTL, and intelligent eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, max_memory_mb: int = 1000):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self.cache = OrderedDict()
        self.access_times = {}
        self.memory_usage = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
                
            value, timestamp = self.cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                self._remove_item(key)
                self.stats['misses'] += 1
                return None
            
            # Update access time and move to end (most recently used)
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            self.stats['hits'] += 1
            
            return value
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache."""
        with self.lock:
            try:
                # Estimate memory usage
                value_size = self._estimate_size(value)
                
                # Check if single item exceeds max memory
                if value_size > self.max_memory_bytes:
                    self.logger.warning(f"Item too large for cache: {value_size / 1024 / 1024:.1f}MB")
                    return False
                
                # Remove existing item if updating
                if key in self.cache:
                    self._remove_item(key)
                
                # Ensure we have space
                while (len(self.cache) >= self.max_size or 
                       self.memory_usage + value_size > self.max_memory_bytes):
                    if not self._evict_lru():
                        break
                
                # Add new item
                self.cache[key] = (value, time.time())
                self.access_times[key] = time.time()
                self.memory_usage += value_size
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache item: {e}")
                return False
    
    def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            value, _ = self.cache[key]
            self.memory_usage -= self._estimate_size(value)
            del self.cache[key]
            del self.access_times[key]
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item."""
        if not self.cache:
            return False
            
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_item(lru_key)
        self.stats['evictions'] += 1
        
        return True
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            if isinstance(obj, (str, int, float, bool)):
                return len(str(obj))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            elif hasattr(obj, '__sizeof__'):
                return obj.__sizeof__()
            else:
                # Fallback to pickle size
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1000  # Conservative fallback
    
    def clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.memory_usage = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'size': len(self.cache),
                'memory_usage_mb': self.memory_usage / 1024 / 1024,
                'hit_rate': hit_rate
            }


class ProcessPool:
    """Intelligent process pool for CPU-intensive tasks."""
    
    def __init__(self, max_workers: Optional[int] = None, task_timeout: int = 300):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.task_timeout = task_timeout
        self.executor = None
        self.logger = logging.getLogger(__name__)
        
        # Task queue and monitoring
        self.active_tasks = {}
        self.task_history = []
        self.lock = threading.Lock()
        
    def __enter__(self):
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to process pool."""
        if not self.executor:
            raise RuntimeError("ProcessPool not initialized. Use as context manager.")
            
        task_id = hashlib.md5(f"{func.__name__}_{time.time()}".encode()).hexdigest()[:8]
        
        with self.lock:
            future = self.executor.submit(func, *args, **kwargs)
            self.active_tasks[task_id] = {
                'future': future,
                'function': func.__name__,
                'submitted_at': time.time(),
                'args_hash': hashlib.md5(str(args).encode()).hexdigest()[:8]
            }
            
        self.logger.info(f"Submitted task {task_id}: {func.__name__}")
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[int] = None) -> Any:
        """Get result from completed task."""
        timeout = timeout or self.task_timeout
        
        with self.lock:
            if task_id not in self.active_tasks:
                raise ValueError(f"Task {task_id} not found")
                
            task_info = self.active_tasks[task_id]
            future = task_info['future']
        
        try:
            result = future.result(timeout=timeout)
            
            with self.lock:
                # Move to history
                task_info['completed_at'] = time.time()
                task_info['duration'] = task_info['completed_at'] - task_info['submitted_at']
                task_info['success'] = True
                
                self.task_history.append(task_info)
                del self.active_tasks[task_id]
                
                # Keep history size manageable
                if len(self.task_history) > 1000:
                    self.task_history = self.task_history[-500:]
                    
            self.logger.info(f"Task {task_id} completed in {task_info['duration']:.2f}s")
            return result
            
        except Exception as e:
            with self.lock:
                task_info['completed_at'] = time.time()
                task_info['duration'] = task_info['completed_at'] - task_info['submitted_at']
                task_info['success'] = False
                task_info['error'] = str(e)
                
                self.task_history.append(task_info)
                del self.active_tasks[task_id]
                
            self.logger.error(f"Task {task_id} failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get process pool status."""
        with self.lock:
            active_count = len(self.active_tasks)
            completed_count = len([t for t in self.task_history if t.get('success', False)])
            failed_count = len([t for t in self.task_history if not t.get('success', True)])
            
            avg_duration = 0
            if self.task_history:
                durations = [t.get('duration', 0) for t in self.task_history if 'duration' in t]
                avg_duration = sum(durations) / len(durations) if durations else 0
                
            return {
                'max_workers': self.max_workers,
                'active_tasks': active_count,
                'completed_tasks': completed_count,
                'failed_tasks': failed_count,
                'average_duration': avg_duration,
                'utilization': active_count / self.max_workers
            }


class MemoryPool:
    """Memory pool for efficient tensor allocation and reuse."""
    
    def __init__(self, device: str = 'cpu', max_pool_size_mb: int = 1000):
        self.device = device
        self.max_pool_size_bytes = max_pool_size_mb * 1024 * 1024
        
        # Pool organized by shape and dtype
        self.pools = defaultdict(list)  # {(shape, dtype): [tensors]}
        self.pool_size = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'evictions': 0
        }
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from pool or allocate new one."""
        key = (tuple(shape), dtype)
        
        with self.lock:
            if key in self.pools and self.pools[key]:
                # Reuse existing tensor
                tensor = self.pools[key].pop()
                self.stats['reuses'] += 1
                return tensor
            else:
                # Allocate new tensor
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                self.stats['allocations'] += 1
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse."""
        if not isinstance(tensor, torch.Tensor):
            return
            
        key = (tuple(tensor.shape), tensor.dtype)
        tensor_size = tensor.numel() * tensor.element_size()
        
        with self.lock:
            # Check if we have space
            if self.pool_size + tensor_size > self.max_pool_size_bytes:
                self._evict_tensors(tensor_size)
            
            # Clear tensor data for security
            tensor.zero_()
            
            # Add to pool
            self.pools[key].append(tensor)
            self.pool_size += tensor_size
    
    def _evict_tensors(self, needed_size: int):
        """Evict tensors to make space."""
        evicted_size = 0
        
        # Evict largest tensors first
        for key in sorted(self.pools.keys(), key=lambda k: k[0], reverse=True):
            if evicted_size >= needed_size:
                break
                
            pool = self.pools[key]
            while pool and evicted_size < needed_size:
                tensor = pool.pop()
                evicted_size += tensor.numel() * tensor.element_size()
                self.stats['evictions'] += 1
                
            if not pool:
                del self.pools[key]
        
        self.pool_size -= evicted_size
    
    def clear(self):
        """Clear all tensors from pool."""
        with self.lock:
            self.pools.clear()
            self.pool_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            total_requests = self.stats['allocations'] + self.stats['reuses']
            reuse_rate = self.stats['reuses'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'pool_size_mb': self.pool_size / 1024 / 1024,
                'num_shapes': len(self.pools),
                'reuse_rate': reuse_rate
            }


def cached_computation(cache_size: int = 128, ttl_seconds: int = 3600):
    """Decorator for caching expensive computations."""
    
    def decorator(func: Callable) -> Callable:
        cache = AdvancedCache(max_size=cache_size, ttl_seconds=ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        wrapper.cache = cache
        return wrapper
        
    return decorator


def async_batch_processor(batch_size: int = 10, max_wait_time: float = 1.0):
    """Decorator for batching async operations."""
    
    def decorator(func: Callable) -> Callable:
        pending_items = []
        pending_futures = []
        last_process_time = time.time()
        lock = threading.Lock()
        
        def process_batch():
            nonlocal pending_items, pending_futures, last_process_time
            
            with lock:
                if not pending_items:
                    return
                    
                items = pending_items[:]
                futures = pending_futures[:]
                pending_items.clear()
                pending_futures.clear()
                last_process_time = time.time()
            
            try:
                # Process batch
                results = func(items)
                
                # Distribute results to futures
                for future, result in zip(futures, results):
                    future.set_result(result)
                    
            except Exception as e:
                # Set exception for all futures
                for future in futures:
                    future.set_exception(e)
        
        @wraps(func)
        def wrapper(item):
            from concurrent.futures import Future
            
            future = Future()
            
            with lock:
                pending_items.append(item)
                pending_futures.append(future)
                
                should_process = (
                    len(pending_items) >= batch_size or
                    time.time() - last_process_time >= max_wait_time
                )
            
            if should_process:
                # Process in background thread
                threading.Thread(target=process_batch, daemon=True).start()
            
            return future
        
        return wrapper
        
    return decorator


class ModelOptimizer:
    """Advanced model optimization with JIT, quantization, and pruning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimized_models = {}
        self.quantization_enabled = torch.version.cuda is not None
    
    def optimize_model(self, model: torch.nn.Module, optimization_level: str = 'basic') -> torch.nn.Module:
        """Apply comprehensive model optimizations."""
        model_key = f"{model.__class__.__name__}_{id(model)}"
        
        if model_key in self.optimized_models:
            return self.optimized_models[model_key]
        
        optimized_model = model
        
        try:
            if optimization_level in ['basic', 'aggressive']:
                # JIT compilation
                optimized_model = self._apply_jit_optimization(optimized_model)
            
            if optimization_level == 'aggressive':
                # Quantization for inference
                optimized_model = self._apply_quantization(optimized_model)
                
                # Graph optimization
                optimized_model = self._apply_graph_optimization(optimized_model)
            
            self.optimized_models[model_key] = optimized_model
            self.logger.info(f"Model {model.__class__.__name__} optimized with {optimization_level} level")
            
        except Exception as e:
            self.logger.warning(f"Model optimization failed: {e}, using original model")
            optimized_model = model
        
        return optimized_model
    
    def _apply_jit_optimization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply TorchScript JIT compilation."""
        try:
            model.eval()
            
            # Create sample input based on expected input shape
            sample_input = self._create_sample_input(model)
            
            if sample_input is not None:
                # Trace the model
                traced_model = torch.jit.trace(model, sample_input)
                
                # Optimize the traced model
                traced_model = torch.jit.optimize_for_inference(traced_model)
                
                return traced_model
            else:
                self.logger.warning("Could not create sample input for JIT tracing")
                return model
                
        except Exception as e:
            self.logger.warning(f"JIT optimization failed: {e}")
            return model
    
    def _apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply dynamic quantization for inference speedup."""
        if not self.quantization_enabled:
            return model
        
        try:
            # Apply dynamic quantization to linear layers
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            return quantized_model
            
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
            return model
    
    def _apply_graph_optimization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply graph-level optimizations."""
        try:
            if hasattr(model, 'graph'):
                # Fuse common patterns
                torch.jit.fuse_subgraphs(model.graph)
                
                # Remove dead code
                torch.jit.eliminate_dead_code(model.graph)
                
                # Constant folding
                torch.jit.constant_fold(model.graph)
            
            return model
            
        except Exception as e:
            self.logger.warning(f"Graph optimization failed: {e}")
            return model
    
    def _create_sample_input(self, model: torch.nn.Module):
        """Create sample input for JIT tracing."""
        try:
            # Try to infer input shape from first layer
            for module in model.modules():
                if hasattr(module, 'in_features'):
                    return torch.randn(1, module.in_features)
                elif hasattr(module, 'in_channels'):
                    return torch.randn(1, module.in_channels, 224, 224)
            
            # Default fallback
            return torch.randn(1, 128)
            
        except Exception:
            return None


class MultiGPUManager:
    """Multi-GPU processing and memory management."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available_gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.gpu_utilization = {gpu: 0.0 for gpu in self.available_gpus}
        self.gpu_memory_usage = {gpu: 0.0 for gpu in self.available_gpus}
        self._lock = threading.Lock()
    
    def get_optimal_gpu(self) -> int:
        """Get the GPU with lowest utilization."""
        if not self.available_gpus:
            return -1  # CPU only
        
        with self._lock:
            # Update GPU utilization
            for gpu_id in self.available_gpus:
                try:
                    torch.cuda.set_device(gpu_id)
                    memory_info = torch.cuda.memory_stats(gpu_id)
                    
                    allocated = memory_info.get('allocated_bytes.all.current', 0)
                    reserved = memory_info.get('reserved_bytes.all.current', 0)
                    
                    self.gpu_memory_usage[gpu_id] = allocated / (1024**3)  # GB
                    
                    # Simple utilization estimate based on memory usage
                    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                    self.gpu_utilization[gpu_id] = reserved / total_memory
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get stats for GPU {gpu_id}: {e}")
                    self.gpu_utilization[gpu_id] = 1.0  # Mark as fully utilized
            
            # Return GPU with lowest utilization
            return min(self.gpu_utilization.keys(), key=lambda x: self.gpu_utilization[x])
    
    def parallelize_model(self, model: torch.nn.Module, strategy: str = 'data_parallel') -> torch.nn.Module:
        """Apply multi-GPU parallelization."""
        if len(self.available_gpus) <= 1:
            return model
        
        try:
            if strategy == 'data_parallel':
                return torch.nn.DataParallel(model, device_ids=self.available_gpus)
            
            elif strategy == 'distributed_data_parallel':
                # Requires proper DDP setup
                return torch.nn.parallel.DistributedDataParallel(model)
            
            else:
                self.logger.warning(f"Unknown parallelization strategy: {strategy}")
                return model
                
        except Exception as e:
            self.logger.warning(f"Model parallelization failed: {e}")
            return model
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get comprehensive GPU statistics."""
        stats = {}
        
        for gpu_id in self.available_gpus:
            try:
                torch.cuda.set_device(gpu_id)
                
                memory_info = torch.cuda.memory_stats(gpu_id)
                device_props = torch.cuda.get_device_properties(gpu_id)
                
                stats[f'gpu_{gpu_id}'] = {
                    'name': device_props.name,
                    'total_memory_gb': device_props.total_memory / (1024**3),
                    'allocated_gb': memory_info.get('allocated_bytes.all.current', 0) / (1024**3),
                    'reserved_gb': memory_info.get('reserved_bytes.all.current', 0) / (1024**3),
                    'utilization': self.gpu_utilization[gpu_id],
                    'temperature': torch.cuda.temperature(gpu_id) if hasattr(torch.cuda, 'temperature') else None
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to get stats for GPU {gpu_id}: {e}")
                stats[f'gpu_{gpu_id}'] = {'error': str(e)}
        
        return stats


class PipelineOptimizer:
    """Optimize data processing pipelines with prefetching and caching."""
    
    def __init__(self, prefetch_factor: int = 2, num_workers: int = 4):
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.pipeline_cache = {}
        self.logger = logging.getLogger(__name__)
    
    def optimize_dataloader(self, dataloader_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize DataLoader configuration for maximum throughput."""
        optimized_config = dataloader_config.copy()
        
        # Optimize based on system capabilities
        cpu_count = mp.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Adjust num_workers based on CPU and memory
        optimal_workers = min(
            self.num_workers,
            cpu_count,
            int(available_memory_gb / 2)  # 2GB per worker heuristic
        )
        optimized_config['num_workers'] = optimal_workers
        
        # Enable optimizations
        optimized_config['pin_memory'] = torch.cuda.is_available()
        optimized_config['persistent_workers'] = optimal_workers > 0
        optimized_config['prefetch_factor'] = self.prefetch_factor if optimal_workers > 0 else None
        
        # Enable automatic batching optimizations
        if 'batch_size' in optimized_config:
            optimized_config['drop_last'] = True  # For consistent batch sizes
        
        self.logger.info(f"Optimized DataLoader: {optimal_workers} workers, prefetch={self.prefetch_factor}")
        
        return optimized_config
    
    def create_prefetch_pipeline(self, data_source: Callable, batch_size: int = 32) -> 'PrefetchPipeline':
        """Create optimized data prefetching pipeline."""
        return PrefetchPipeline(data_source, batch_size, self.prefetch_factor, self.num_workers)


class PrefetchPipeline:
    """High-performance data prefetching pipeline."""
    
    def __init__(self, data_source: Callable, batch_size: int, prefetch_factor: int, num_workers: int):
        self.data_source = data_source
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.prefetch_queue = None
        self.executor = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_factor * self.batch_size)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        # Start prefetching
        for _ in range(self.prefetch_factor):
            future = self.executor.submit(self._fetch_batch)
            self.prefetch_queue.put(future)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.prefetch_queue and not self.prefetch_queue.empty():
            future = self.prefetch_queue.get()
            
            try:
                batch = future.result(timeout=30.0)
                
                # Queue next batch
                next_future = self.executor.submit(self._fetch_batch)
                self.prefetch_queue.put(next_future)
                
                return batch
                
            except Exception as e:
                self.logger.error(f"Prefetch error: {e}")
                raise StopIteration
        else:
            raise StopIteration
    
    def _fetch_batch(self):
        """Fetch a batch of data."""
        try:
            batch = []
            for _ in range(self.batch_size):
                item = self.data_source()
                batch.append(item)
            return batch
        except Exception as e:
            self.logger.error(f"Batch fetch failed: {e}")
            return []


# Global instances
performance_optimizer = PerformanceOptimizer()
model_optimizer = ModelOptimizer()
multi_gpu_manager = MultiGPUManager()
pipeline_optimizer = PipelineOptimizer()
global_cache = AdvancedCache(max_size=10000, ttl_seconds=7200, max_memory_mb=2000)
memory_pool = MemoryPool(device='cpu', max_pool_size_mb=500)

# GPU memory pool if available
gpu_memory_pool = None
if torch.cuda.is_available():
    gpu_memory_pool = MemoryPool(device='cuda', max_pool_size_mb=1000)