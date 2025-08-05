"""Auto-scaling and load balancing utilities for distributed processing."""

import time
import threading
import queue
import logging
import psutil
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import weakref
import math


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: Optional[float] = None
    active_workers: int = 0
    queue_size: int = 0
    throughput: float = 0.0  # tasks per second


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    # Scale-up thresholds
    cpu_scale_up_threshold: float = 70.0
    memory_scale_up_threshold: float = 70.0
    queue_scale_up_threshold: int = 10
    
    # Scale-down thresholds
    cpu_scale_down_threshold: float = 30.0
    memory_scale_down_threshold: float = 30.0
    queue_scale_down_threshold: int = 2
    
    # Scaling parameters
    min_workers: int = 1
    max_workers: int = 8
    scale_up_cooldown: int = 60  # seconds
    scale_down_cooldown: int = 120  # seconds
    evaluation_window: int = 300  # seconds
    
    # Batch processing
    adaptive_batch_size: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 32
    target_batch_latency: float = 5.0  # seconds


class AdaptiveLoadBalancer:
    """Intelligent load balancer with performance-based routing."""
    
    def __init__(self, workers: List[Any] = None):
        self.workers = workers or []
        self.worker_stats = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.response_times = {worker: deque(maxlen=100) for worker in self.workers}
        self.error_counts = {worker: 0 for worker in self.workers}
        self.success_counts = {worker: 0 for worker in self.workers}
        
        # Load balancing algorithms
        self.algorithms = {
            'round_robin': self._round_robin,
            'least_connections': self._least_connections,
            'weighted_response_time': self._weighted_response_time,
            'adaptive': self._adaptive_routing
        }
        
        self.current_algorithm = 'adaptive'
        self.round_robin_counter = 0
    
    def add_worker(self, worker: Any):
        """Add worker to the pool."""
        with self.lock:
            if worker not in self.workers:
                self.workers.append(worker)
                self.response_times[worker] = deque(maxlen=100)
                self.error_counts[worker] = 0
                self.success_counts[worker] = 0
                self.logger.info(f"Added worker: {worker}")
    
    def remove_worker(self, worker: Any):
        """Remove worker from the pool."""
        with self.lock:
            if worker in self.workers:
                self.workers.remove(worker)
                self.response_times.pop(worker, None)
                self.error_counts.pop(worker, None)
                self.success_counts.pop(worker, None)
                self.logger.info(f"Removed worker: {worker}")
    
    def select_worker(self, request_info: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Select optimal worker based on current algorithm."""
        with self.lock:
            if not self.workers:
                return None
                
            available_workers = [w for w in self.workers if self._is_worker_healthy(w)]
            
            if not available_workers:
                # Fallback to any worker if none are "healthy"
                available_workers = self.workers
                
            return self.algorithms[self.current_algorithm](available_workers, request_info)
    
    def record_response(self, worker: Any, response_time: float, success: bool):
        """Record worker response for performance tracking."""
        with self.lock:
            if worker in self.response_times:
                self.response_times[worker].append(response_time)
                
                if success:
                    self.success_counts[worker] += 1
                else:
                    self.error_counts[worker] += 1
    
    def _is_worker_healthy(self, worker: Any) -> bool:
        """Check if worker is healthy based on recent performance."""
        total_requests = self.success_counts.get(worker, 0) + self.error_counts.get(worker, 0)
        
        if total_requests < 10:
            return True  # Not enough data
            
        error_rate = self.error_counts.get(worker, 0) / total_requests
        return error_rate < 0.1  # Less than 10% error rate
    
    def _round_robin(self, workers: List[Any], request_info: Optional[Dict[str, Any]]) -> Any:
        """Round-robin load balancing."""
        if not workers:
            return None
            
        selected = workers[self.round_robin_counter % len(workers)]
        self.round_robin_counter = (self.round_robin_counter + 1) % len(workers)
        return selected
    
    def _least_connections(self, workers: List[Any], request_info: Optional[Dict[str, Any]]) -> Any:
        """Select worker with least active connections."""
        # Simplified - would need actual connection tracking
        return min(workers, key=lambda w: len(self.response_times.get(w, [])))
    
    def _weighted_response_time(self, workers: List[Any], request_info: Optional[Dict[str, Any]]) -> Any:
        """Select worker based on weighted response time."""
        worker_weights = {}
        
        for worker in workers:
            response_times = self.response_times.get(worker, [])
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                # Lower response time = higher weight (inverse relationship)
                worker_weights[worker] = 1.0 / (avg_response_time + 0.001)
            else:
                worker_weights[worker] = 1.0
        
        # Weighted random selection
        total_weight = sum(worker_weights.values())
        if total_weight == 0:
            return workers[0]
            
        import random
        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for worker, weight in worker_weights.items():
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return worker
                
        return workers[-1]
    
    def _adaptive_routing(self, workers: List[Any], request_info: Optional[Dict[str, Any]]) -> Any:
        """Adaptive routing based on multiple factors."""
        if len(workers) == 1:
            return workers[0]
            
        scores = {}
        
        for worker in workers:
            score = 0.0
            
            # Response time factor (lower is better)
            response_times = self.response_times.get(worker, [])
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                score += 1.0 / (avg_response_time + 0.001) * 0.4
            else:
                score += 0.4  # Neutral score for new workers
            
            # Error rate factor (lower is better)
            total_requests = self.success_counts.get(worker, 0) + self.error_counts.get(worker, 0)
            if total_requests > 0:
                error_rate = self.error_counts.get(worker, 0) / total_requests
                score += (1.0 - error_rate) * 0.3
            else:
                score += 0.3
            
            # Load factor (lower current load is better)
            current_load = len(self.response_times.get(worker, []))
            max_load = max(len(self.response_times.get(w, [])) for w in workers)
            if max_load > 0:
                load_factor = 1.0 - (current_load / max_load)
                score += load_factor * 0.3
            else:
                score += 0.3
            
            scores[worker] = score
        
        # Select worker with highest score
        return max(workers, key=lambda w: scores.get(w, 0))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            stats = {
                'num_workers': len(self.workers),
                'algorithm': self.current_algorithm,
                'worker_stats': {}
            }
            
            for worker in self.workers:
                response_times = list(self.response_times.get(worker, []))
                total_requests = self.success_counts.get(worker, 0) + self.error_counts.get(worker, 0)
                
                worker_stats = {
                    'total_requests': total_requests,
                    'success_count': self.success_counts.get(worker, 0),
                    'error_count': self.error_counts.get(worker, 0),
                    'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                    'is_healthy': self._is_worker_healthy(worker)
                }
                
                stats['worker_stats'][str(worker)] = worker_stats
            
            return stats


class AutoScaler:
    """Automatic scaling based on resource utilization and queue depth."""
    
    def __init__(self, scaling_policy: ScalingPolicy = None):
        self.policy = scaling_policy or ScalingPolicy()
        self.metrics_history = deque(maxlen=1000)
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.current_workers = self.policy.min_workers
        
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Callbacks for scaling actions
        self.scale_up_callbacks = []
        self.scale_down_callbacks = []
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def add_scale_up_callback(self, callback: Callable[[int], None]):
        """Add callback for scale-up events."""
        self.scale_up_callbacks.append(callback)
    
    def add_scale_down_callback(self, callback: Callable[[int], None]):
        """Add callback for scale-down events."""
        self.scale_down_callbacks.append(callback)
    
    def record_metrics(self, metrics: ResourceMetrics):
        """Record current resource metrics."""
        with self.lock:
            self.metrics_history.append(metrics)
    
    def _monitor_loop(self):
        """Background monitoring and scaling loop."""
        while self.monitoring:
            try:
                self._evaluate_scaling()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in scaling monitor: {e}")
                time.sleep(60)
    
    def _evaluate_scaling(self):
        """Evaluate if scaling action is needed."""
        with self.lock:
            if len(self.metrics_history) < 3:
                return  # Not enough data
            
            # Get recent metrics for evaluation
            evaluation_window_start = time.time() - self.policy.evaluation_window
            recent_metrics = [m for m in self.metrics_history if m.timestamp > evaluation_window_start]
            
            if not recent_metrics:
                return
            
            # Calculate average metrics
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_queue_size = sum(m.queue_size for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
            
            current_time = time.time()
            
            # Check for scale-up conditions
            should_scale_up = (
                (avg_cpu > self.policy.cpu_scale_up_threshold or
                 avg_memory > self.policy.memory_scale_up_threshold or
                 avg_queue_size > self.policy.queue_scale_up_threshold) and
                self.current_workers < self.policy.max_workers and
                current_time - self.last_scale_up > self.policy.scale_up_cooldown
            )
            
            # Check for scale-down conditions
            should_scale_down = (
                avg_cpu < self.policy.cpu_scale_down_threshold and
                avg_memory < self.policy.memory_scale_down_threshold and
                avg_queue_size < self.policy.queue_scale_down_threshold and
                self.current_workers > self.policy.min_workers and
                current_time - self.last_scale_down > self.policy.scale_down_cooldown
            )
            
            if should_scale_up:
                self._scale_up(avg_cpu, avg_memory, avg_queue_size)
            elif should_scale_down:
                self._scale_down(avg_cpu, avg_memory, avg_queue_size)
    
    def _scale_up(self, cpu_percent: float, memory_percent: float, queue_size: float):
        """Execute scale-up action."""
        old_workers = self.current_workers
        
        # Determine scale-up amount based on pressure
        if cpu_percent > 90 or memory_percent > 90 or queue_size > 50:
            # High pressure - scale up more aggressively
            scale_factor = 2
        else:
            scale_factor = 1
            
        new_workers = min(self.current_workers + scale_factor, self.policy.max_workers)
        
        if new_workers > old_workers:
            self.current_workers = new_workers
            self.last_scale_up = time.time()
            
            self.logger.info(f"Scaling up from {old_workers} to {new_workers} workers")
            self.logger.info(f"Triggers - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Queue: {queue_size:.1f}")
            
            # Execute callbacks
            for callback in self.scale_up_callbacks:
                try:
                    callback(new_workers - old_workers)
                except Exception as e:
                    self.logger.error(f"Scale-up callback error: {e}")
    
    def _scale_down(self, cpu_percent: float, memory_percent: float, queue_size: float):
        """Execute scale-down action."""
        old_workers = self.current_workers
        new_workers = max(self.current_workers - 1, self.policy.min_workers)
        
        if new_workers < old_workers:
            self.current_workers = new_workers
            self.last_scale_down = time.time()
            
            self.logger.info(f"Scaling down from {old_workers} to {new_workers} workers")
            self.logger.info(f"Metrics - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Queue: {queue_size:.1f}")
            
            # Execute callbacks
            for callback in self.scale_down_callbacks:
                try:
                    callback(old_workers - new_workers)
                except Exception as e:
                    self.logger.error(f"Scale-down callback error: {e}")
    
    def get_optimal_batch_size(self, current_latency: float, current_batch_size: int) -> int:
        """Calculate optimal batch size based on current performance."""
        if not self.policy.adaptive_batch_size:
            return current_batch_size
            
        # Simple adaptive logic
        if current_latency > self.policy.target_batch_latency * 1.2:
            # Latency too high - reduce batch size
            new_batch_size = max(self.policy.min_batch_size, int(current_batch_size * 0.8))
        elif current_latency < self.policy.target_batch_latency * 0.8:
            # Latency low - can increase batch size
            new_batch_size = min(self.policy.max_batch_size, int(current_batch_size * 1.2))
        else:
            new_batch_size = current_batch_size
            
        return new_batch_size
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        with self.lock:
            recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
            
            return {
                'current_workers': self.current_workers,
                'policy': {
                    'min_workers': self.policy.min_workers,
                    'max_workers': self.policy.max_workers,
                    'cpu_thresholds': [self.policy.cpu_scale_down_threshold, self.policy.cpu_scale_up_threshold],
                    'memory_thresholds': [self.policy.memory_scale_down_threshold, self.policy.memory_scale_up_threshold]
                },
                'recent_metrics': [
                    {
                        'timestamp': m.timestamp,
                        'cpu_percent': m.cpu_percent,
                        'memory_percent': m.memory_percent,
                        'queue_size': m.queue_size,
                        'throughput': m.throughput
                    } for m in recent_metrics
                ],
                'last_scale_up_ago': time.time() - self.last_scale_up if self.last_scale_up > 0 else None,
                'last_scale_down_ago': time.time() - self.last_scale_down if self.last_scale_down > 0 else None
            }


class DistributedTaskManager:
    """Manage distributed task execution with auto-scaling and load balancing."""
    
    def __init__(self, initial_workers: int = 2, scaling_policy: ScalingPolicy = None):
        self.scaling_policy = scaling_policy or ScalingPolicy()
        self.load_balancer = AdaptiveLoadBalancer()
        self.auto_scaler = AutoScaler(self.scaling_policy)
        
        # Task management
        self.task_queue = queue.Queue()
        self.workers = []
        self.worker_threads = []
        
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        
        # Setup auto-scaling callbacks
        self.auto_scaler.add_scale_up_callback(self._add_workers)
        self.auto_scaler.add_scale_down_callback(self._remove_workers)
        
        # Initialize workers
        self._add_workers(initial_workers)
        
        # Start metrics collection
        self.metrics_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self.metrics_thread.start()
    
    def submit_task(self, task_func: Callable, *args, **kwargs) -> Future:
        """Submit task for distributed execution."""
        future = Future()
        task_item = {
            'function': task_func,
            'args': args,
            'kwargs': kwargs,
            'future': future,
            'submitted_at': time.time()
        }
        
        self.task_queue.put(task_item)
        return future
    
    def _add_workers(self, count: int):
        """Add worker threads."""
        with self.lock:
            for _ in range(count):
                worker_id = len(self.workers)
                worker_thread = threading.Thread(
                    target=self._worker_loop, 
                    args=(worker_id,), 
                    daemon=True
                )
                
                self.workers.append(worker_id)
                self.worker_threads.append(worker_thread)
                self.load_balancer.add_worker(worker_id)
                
                worker_thread.start()
                
            self.logger.info(f"Added {count} workers, total: {len(self.workers)}")
    
    def _remove_workers(self, count: int):
        """Remove worker threads (graceful shutdown)."""
        with self.lock:
            # Mark workers for removal (they'll stop when they check)
            workers_to_remove = self.workers[-count:] if count < len(self.workers) else self.workers[1:]  # Keep at least 1
            
            for worker_id in workers_to_remove:
                self.load_balancer.remove_worker(worker_id)
                
            # Remove from tracking
            self.workers = [w for w in self.workers if w not in workers_to_remove]
            
            self.logger.info(f"Removed {len(workers_to_remove)} workers, remaining: {len(self.workers)}")
    
    def _worker_loop(self, worker_id: int):
        """Worker thread main loop."""
        self.logger.info(f"Worker {worker_id} started")
        
        while worker_id in self.workers:
            try:
                # Get task with timeout
                task_item = self.task_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                try:
                    # Execute task
                    result = task_item['function'](*task_item['args'], **task_item['kwargs'])
                    task_item['future'].set_result(result)
                    
                    # Record success
                    response_time = time.time() - start_time
                    self.load_balancer.record_response(worker_id, response_time, True)
                    self.completed_tasks += 1
                    
                except Exception as e:
                    task_item['future'].set_exception(e)
                    
                    # Record failure
                    response_time = time.time() - start_time
                    self.load_balancer.record_response(worker_id, response_time, False)
                    self.failed_tasks += 1
                    
                finally:
                    self.task_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                
        self.logger.info(f"Worker {worker_id} stopped")
    
    def _collect_metrics_loop(self):
        """Collect and report metrics for auto-scaling."""
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # GPU metrics if available
                gpu_memory_percent = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory_info = torch.cuda.mem_get_info()
                        gpu_memory_percent = ((memory_info[1] - memory_info[0]) / memory_info[1]) * 100
                except ImportError:
                    pass
                
                # Task queue metrics
                queue_size = self.task_queue.qsize()
                
                # Throughput calculation
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                throughput = self.completed_tasks / elapsed_time if elapsed_time > 0 else 0
                
                # Create metrics object
                metrics = ResourceMetrics(
                    timestamp=current_time,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    gpu_memory_percent=gpu_memory_percent,
                    active_workers=len(self.workers),
                    queue_size=queue_size,
                    throughput=throughput
                )
                
                # Report to auto-scaler
                self.auto_scaler.record_metrics(metrics)
                
                time.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                time.sleep(30)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        elapsed_time = time.time() - self.start_time
        
        return {
            'task_stats': {
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'queue_size': self.task_queue.qsize(),
                'throughput': self.completed_tasks / elapsed_time if elapsed_time > 0 else 0,
                'uptime_seconds': elapsed_time
            },
            'load_balancer': self.load_balancer.get_stats(),
            'auto_scaler': self.auto_scaler.get_stats()
        }
    
    def shutdown(self):
        """Graceful shutdown of the task manager."""
        self.logger.info("Shutting down distributed task manager...")
        
        # Stop auto-scaler
        self.auto_scaler.stop_monitoring()
        
        # Wait for pending tasks
        self.task_queue.join()
        
        # Clear workers (they'll stop naturally)
        with self.lock:
            self.workers.clear()
        
        self.logger.info("Shutdown complete")


# Example usage and global instances
default_scaling_policy = ScalingPolicy(
    min_workers=2,
    max_workers=min(8, psutil.cpu_count()),
    cpu_scale_up_threshold=75.0,
    cpu_scale_down_threshold=25.0,
    evaluation_window=180
)

# Global task manager instance
global_task_manager = None

def get_task_manager() -> DistributedTaskManager:
    """Get or create global task manager."""
    global global_task_manager
    if global_task_manager is None:
        global_task_manager = DistributedTaskManager(
            initial_workers=2,
            scaling_policy=default_scaling_policy
        )
    return global_task_manager