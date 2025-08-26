"""
Production Optimization System for DGDM Histopath Lab
Comprehensive optimization framework with minimal dependencies
"""

import time
import threading
import json
import logging
import os
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
import weakref


class OptimizationLevel(Enum):
    """Optimization levels for different environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    HIGH_PERFORMANCE = "high_performance"


class ResourceType(Enum):
    """Types of resources to optimize."""
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"
    CACHE = "cache"


@dataclass
class OptimizationMetrics:
    """Optimization performance metrics."""
    timestamp: float
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    io_operations_per_second: float = 0.0
    network_latency_ms: float = 0.0
    gc_collections: int = 0
    active_threads: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'cache_hit_rate': self.cache_hit_rate,
            'io_operations_per_second': self.io_operations_per_second,
            'network_latency_ms': self.network_latency_ms,
            'gc_collections': self.gc_collections,
            'active_threads': self.active_threads
        }


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tracked_objects = weakref.WeakSet()
        self.gc_threshold = 1000
        self.last_gc_time = time.time()
        
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        start_time = time.time()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Update tracking
        self.last_gc_time = time.time()
        
        # Get memory statistics
        memory_stats = self._get_memory_stats()
        
        optimization_time = time.time() - start_time
        
        result = {
            'collected_objects': collected,
            'optimization_time_ms': optimization_time * 1000,
            'memory_stats': memory_stats
        }
        
        self.logger.debug(f"Memory optimization completed: {collected} objects collected in {optimization_time:.3f}s")
        return result
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics with fallback."""
        try:
            # Try to get system memory info
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent()
            }
        except ImportError:
            # Fallback to basic Python stats
            return {
                'gc_counts': gc.get_count(),
                'gc_stats': gc.get_stats() if hasattr(gc, 'get_stats') else []
            }
    
    def register_object(self, obj: Any):
        """Register object for memory tracking."""
        self.tracked_objects.add(obj)
    
    def get_tracked_count(self) -> int:
        """Get number of tracked objects."""
        return len(self.tracked_objects)


class IOOptimizer:
    """I/O optimization utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.io_cache = {}
        self.max_cache_size = 100
        self.io_stats = {
            'reads': 0,
            'writes': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def optimized_read(self, file_path: str, use_cache: bool = True) -> Optional[str]:
        """Optimized file reading with caching."""
        if use_cache and file_path in self.io_cache:
            self.io_stats['cache_hits'] += 1
            return self.io_cache[file_path]
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if use_cache and len(self.io_cache) < self.max_cache_size:
                self.io_cache[file_path] = content
            
            self.io_stats['reads'] += 1
            self.io_stats['cache_misses'] += 1 if use_cache else 0
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    def optimized_write(self, file_path: str, content: str, sync: bool = False) -> bool:
        """Optimized file writing."""
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
                if sync:
                    f.flush()
                    os.fsync(f.fileno())
            
            # Update cache
            if file_path in self.io_cache:
                self.io_cache[file_path] = content
            
            self.io_stats['writes'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write file {file_path}: {e}")
            return False
    
    def clear_cache(self):
        """Clear I/O cache."""
        self.io_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get I/O statistics."""
        total_operations = self.io_stats['reads'] + self.io_stats['writes']
        cache_operations = self.io_stats['cache_hits'] + self.io_stats['cache_misses']
        
        return {
            **self.io_stats,
            'total_operations': total_operations,
            'cache_hit_rate': self.io_stats['cache_hits'] / max(cache_operations, 1),
            'cached_files': len(self.io_cache)
        }


class CPUOptimizer:
    """CPU optimization utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cpu_bound_tasks = []
        self.optimization_history = []
    
    def optimize_cpu_usage(self) -> Dict[str, Any]:
        """Perform CPU optimization."""
        start_time = time.time()
        
        # Optimize thread usage
        thread_count = threading.active_count()
        
        # Set thread priorities if possible
        try:
            import os
            current_process = os.getpid()
            # Try to set nice value for better scheduling
            try:
                os.nice(0)  # Reset to default priority
            except:
                pass
        except ImportError:
            pass
        
        # CPU affinity optimization (if available)
        cpu_info = self._get_cpu_info()
        
        optimization_time = time.time() - start_time
        
        result = {
            'active_threads': thread_count,
            'optimization_time_ms': optimization_time * 1000,
            'cpu_info': cpu_info
        }
        
        self.optimization_history.append(result)
        
        # Keep only recent history
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        return result
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information with fallback."""
        try:
            import psutil
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except ImportError:
            try:
                import os
                return {
                    'cpu_count': os.cpu_count() or 1,
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                }
            except:
                return {'cpu_count': 1}
    
    def cpu_intensive_task(self, task_name: str):
        """Decorator for CPU-intensive tasks."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    self.cpu_bound_tasks.append({
                        'task_name': task_name,
                        'execution_time': execution_time,
                        'timestamp': time.time()
                    })
                    
                    # Keep only recent tasks
                    if len(self.cpu_bound_tasks) > 50:
                        self.cpu_bound_tasks = self.cpu_bound_tasks[-50:]
                    
                    return result
                except Exception as e:
                    self.logger.error(f"CPU task {task_name} failed: {e}")
                    raise
            return wrapper
        return decorator


class ProductionOptimizer:
    """Main production optimization system."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.PRODUCTION):
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimizers
        self.memory_optimizer = MemoryOptimizer()
        self.io_optimizer = IOOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        
        # Optimization state
        self.optimization_active = False
        self.optimization_thread = None
        self.stop_event = threading.Event()
        
        # Metrics
        self.metrics_history: List[OptimizationMetrics] = []
        self.optimization_interval = self._get_optimization_interval()
        
        # Configuration based on level
        self._configure_optimization_level()
    
    def _get_optimization_interval(self) -> float:
        """Get optimization interval based on level."""
        intervals = {
            OptimizationLevel.DEVELOPMENT: 60.0,  # 1 minute
            OptimizationLevel.TESTING: 30.0,     # 30 seconds
            OptimizationLevel.PRODUCTION: 120.0,  # 2 minutes
            OptimizationLevel.HIGH_PERFORMANCE: 60.0  # 1 minute
        }
        return intervals.get(self.optimization_level, 120.0)
    
    def _configure_optimization_level(self):
        """Configure optimization based on level."""
        if self.optimization_level == OptimizationLevel.HIGH_PERFORMANCE:
            self.memory_optimizer.gc_threshold = 500
            self.io_optimizer.max_cache_size = 200
        elif self.optimization_level == OptimizationLevel.DEVELOPMENT:
            self.memory_optimizer.gc_threshold = 2000
            self.io_optimizer.max_cache_size = 50
    
    def start_optimization(self):
        """Start continuous optimization."""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        self.logger.info(f"Production optimization started ({self.optimization_level.value} level)")
    
    def stop_optimization(self):
        """Stop continuous optimization."""
        self.optimization_active = False
        self.stop_event.set()
        
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=10.0)
        
        self.logger.info("Production optimization stopped")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.optimization_active and not self.stop_event.is_set():
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Perform optimizations
                self._perform_optimizations(metrics)
                
                # Wait for next optimization cycle
                self.stop_event.wait(self.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                self.stop_event.wait(min(self.optimization_interval, 60.0))
    
    def _collect_metrics(self) -> OptimizationMetrics:
        """Collect current optimization metrics."""
        try:
            memory_stats = self.memory_optimizer._get_memory_stats()
            cpu_info = self.cpu_optimizer._get_cpu_info()
            io_stats = self.io_optimizer.get_stats()
            
            return OptimizationMetrics(
                timestamp=time.time(),
                memory_usage_mb=memory_stats.get('rss_mb', 0.0),
                cpu_usage_percent=cpu_info.get('cpu_percent', 0.0),
                cache_hit_rate=io_stats.get('cache_hit_rate', 0.0),
                io_operations_per_second=io_stats.get('total_operations', 0) / max(self.optimization_interval, 1),
                gc_collections=len(memory_stats.get('gc_counts', [])),
                active_threads=threading.active_count()
            )
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return OptimizationMetrics(timestamp=time.time())
    
    def _perform_optimizations(self, metrics: OptimizationMetrics):
        """Perform optimization based on metrics."""
        # Memory optimization
        if metrics.memory_usage_mb > 500 or time.time() - self.memory_optimizer.last_gc_time > 300:
            self.memory_optimizer.optimize_memory()
        
        # I/O cache management
        if len(self.io_optimizer.io_cache) > self.io_optimizer.max_cache_size * 0.8:
            cache_items = list(self.io_optimizer.io_cache.items())
            # Keep recent half
            recent_items = cache_items[len(cache_items)//2:]
            self.io_optimizer.io_cache = dict(recent_items)
        
        # CPU optimization
        if metrics.active_threads > 50:
            self.cpu_optimizer.optimize_cpu_usage()
    
    def optimize_now(self) -> Dict[str, Any]:
        """Perform immediate optimization."""
        start_time = time.time()
        
        # Collect current metrics
        current_metrics = self._collect_metrics()
        
        # Perform all optimizations
        memory_result = self.memory_optimizer.optimize_memory()
        cpu_result = self.cpu_optimizer.optimize_cpu_usage()
        io_stats = self.io_optimizer.get_stats()
        
        total_time = time.time() - start_time
        
        return {
            'timestamp': time.time(),
            'optimization_time_ms': total_time * 1000,
            'current_metrics': current_metrics.to_dict(),
            'memory_optimization': memory_result,
            'cpu_optimization': cpu_result,
            'io_stats': io_stats
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        # Calculate averages
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        
        return {
            'optimization_level': self.optimization_level.value,
            'optimization_active': self.optimization_active,
            'metrics_count': len(self.metrics_history),
            'average_metrics': {
                'memory_usage_mb': avg_memory,
                'cpu_usage_percent': avg_cpu,
                'cache_hit_rate': avg_cache_hit_rate
            },
            'latest_metrics': self.metrics_history[-1].to_dict() if self.metrics_history else None,
            'io_stats': self.io_optimizer.get_stats(),
            'tracked_objects': self.memory_optimizer.get_tracked_count()
        }
    
    @contextmanager
    def optimization_context(self):
        """Context manager for temporary optimization."""
        self.start_optimization()
        try:
            yield self
        finally:
            self.stop_optimization()
    
    def memory_optimized(self, func):
        """Decorator for memory-optimized functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Register objects for tracking
            self.memory_optimizer.register_object(func)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Force garbage collection after memory-intensive operations
                if time.time() - self.memory_optimizer.last_gc_time > 60:
                    self.memory_optimizer.optimize_memory()
        return wrapper


# Global production optimizer
global_optimizer = ProductionOptimizer()


def get_production_optimizer() -> ProductionOptimizer:
    """Get global production optimizer instance."""
    return global_optimizer


def optimize_now() -> Dict[str, Any]:
    """Perform immediate optimization."""
    return global_optimizer.optimize_now()


def start_production_optimization(level: OptimizationLevel = OptimizationLevel.PRODUCTION):
    """Start production optimization."""
    global_optimizer.optimization_level = level
    global_optimizer._configure_optimization_level()
    global_optimizer.start_optimization()


def stop_production_optimization():
    """Stop production optimization."""
    global_optimizer.stop_optimization()


# Convenience decorators
def memory_optimized(func):
    """Decorator for memory optimization."""
    return global_optimizer.memory_optimized(func)


def cpu_intensive(task_name: str):
    """Decorator for CPU-intensive tasks."""
    return global_optimizer.cpu_optimizer.cpu_intensive_task(task_name)


if __name__ == "__main__":
    # Test production optimization
    optimizer = get_production_optimizer()
    
    # Test immediate optimization
    result = optimize_now()
    print("Optimization Result:")
    print(json.dumps(result, indent=2))
    
    # Test continuous optimization
    start_production_optimization(OptimizationLevel.HIGH_PERFORMANCE)
    
    # Simulate some work
    time.sleep(2)
    
    # Get report
    report = optimizer.get_optimization_report()
    print("\nOptimization Report:")
    print(json.dumps(report, indent=2))
    
    stop_production_optimization()