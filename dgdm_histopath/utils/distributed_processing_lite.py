"""
Lightweight Distributed Processing for DGDM Histopath Lab
Minimal-dependency distributed processing with graceful fallback
"""

import threading
import queue
import time
import json
import logging
import os
import socket
import subprocess
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from functools import wraps
import weakref


class ProcessingMode(Enum):
    """Processing execution modes."""
    SINGLE_THREAD = "single_thread"
    MULTI_THREAD = "multi_thread"
    MULTI_PROCESS = "multi_process"
    DISTRIBUTED = "distributed"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTask:
    """Distributed processing task."""
    task_id: str
    function_name: str
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_time: float = field(default_factory=time.time)
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'function_name': self.function_name,
            'args': self.args,
            'kwargs': self.kwargs,
            'priority': self.priority,
            'max_retries': self.max_retries,
            'retry_count': self.retry_count,
            'status': self.status.value,
            'created_time': self.created_time,
            'started_time': self.started_time,
            'completed_time': self.completed_time,
            'result': self.result,
            'error': self.error,
            'worker_id': self.worker_id
        }


class TaskQueue:
    """Thread-safe task queue with priority support."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue = queue.PriorityQueue(maxsize=max_size)
        self._task_registry: Dict[str, ProcessingTask] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def put(self, task: ProcessingTask) -> bool:
        """Add task to queue."""
        try:
            with self._lock:
                if len(self._task_registry) >= self.max_size:
                    self.logger.warning("Task queue is full")
                    return False
                
                # Priority queue uses negative priority for max-heap behavior
                priority_item = (-task.priority, task.created_time, task)
                self._queue.put(priority_item, block=False)
                self._task_registry[task.task_id] = task
                
                self.logger.debug(f"Task {task.task_id} added to queue")
                return True
                
        except queue.Full:
            self.logger.warning("Failed to add task: queue is full")
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[ProcessingTask]:
        """Get next task from queue."""
        try:
            priority_item = self._queue.get(timeout=timeout)
            task = priority_item[2]  # Extract task from priority tuple
            return task
        except queue.Empty:
            return None
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status by ID."""
        with self._lock:
            task = self._task_registry.get(task_id)
            return task.status if task else None
    
    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get task by ID."""
        with self._lock:
            return self._task_registry.get(task_id)
    
    def update_task(self, task: ProcessingTask):
        """Update task in registry."""
        with self._lock:
            self._task_registry[task.task_id] = task
    
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            status_counts = {}
            for task in self._task_registry.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'queue_size': self._queue.qsize(),
                'total_tasks': len(self._task_registry),
                'status_counts': status_counts
            }


class LiteWorker:
    """Lightweight processing worker."""
    
    def __init__(self, worker_id: str, task_queue: TaskQueue):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        self.function_registry: Dict[str, Callable] = {}
        
        # Worker stats
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        self.started_time = time.time()
    
    def register_function(self, name: str, func: Callable):
        """Register a function that can be executed by this worker."""
        self.function_registry[name] = func
    
    def start(self):
        """Start the worker."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        self.logger.info(f"Worker {self.worker_id} started")
    
    def stop(self):
        """Stop the worker."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    def _worker_loop(self):
        """Main worker processing loop."""
        while self.running:
            try:
                # Get next task
                task = self.task_queue.get(timeout=1.0)
                if not task:
                    continue
                
                # Process task
                self._process_task(task)
                
            except Exception as e:
                self.logger.error(f"Worker {self.worker_id} error: {e}")
    
    def _process_task(self, task: ProcessingTask):
        """Process a single task."""
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_time = time.time()
            task.worker_id = self.worker_id
            self.task_queue.update_task(task)
            
            # Execute function
            func = self.function_registry.get(task.function_name)
            if not func:
                raise ValueError(f"Function '{task.function_name}' not registered")
            
            start_time = time.time()
            result = func(*task.args, **task.kwargs)
            execution_time = time.time() - start_time
            
            # Update task with result
            task.status = TaskStatus.COMPLETED
            task.completed_time = time.time()
            task.result = result
            self.task_queue.update_task(task)
            
            # Update worker stats
            self.tasks_completed += 1
            self.total_execution_time += execution_time
            
            self.logger.debug(f"Worker {self.worker_id} completed task {task.task_id} in {execution_time:.3f}s")
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.completed_time = time.time()
            task.error = str(e)
            task.retry_count += 1
            self.task_queue.update_task(task)
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING
                task.started_time = None
                task.worker_id = None
                self.task_queue.put(task)  # Re-queue for retry
            
            self.tasks_failed += 1
            self.logger.error(f"Worker {self.worker_id} failed task {task.task_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = time.time() - self.started_time
        avg_execution_time = self.total_execution_time / max(self.tasks_completed, 1)
        
        return {
            'worker_id': self.worker_id,
            'running': self.running,
            'uptime_seconds': uptime,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_execution_time,
            'throughput_per_minute': (self.tasks_completed / uptime) * 60 if uptime > 0 else 0
        }


class DistributedManager:
    """Lightweight distributed processing manager."""
    
    def __init__(self, num_workers: int = 4, processing_mode: ProcessingMode = ProcessingMode.MULTI_THREAD):
        self.num_workers = num_workers
        self.processing_mode = processing_mode
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.task_queue = TaskQueue()
        self.workers: List[LiteWorker] = []
        self.function_registry: Dict[str, Callable] = {}
        
        # Distributed state
        self.running = False
        self.start_time = time.time()
        
        # Initialize workers
        self._initialize_workers()
        
        # Fallback executors
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
    
    def _initialize_workers(self):
        """Initialize worker instances."""
        for i in range(self.num_workers):
            worker = LiteWorker(f"worker_{i}", self.task_queue)
            self.workers.append(worker)
    
    def register_function(self, name: str, func: Callable):
        """Register a function for distributed execution."""
        self.function_registry[name] = func
        
        # Register with all workers
        for worker in self.workers:
            worker.register_function(name, func)
    
    def start(self):
        """Start distributed processing."""
        if self.running:
            return
        
        self.running = True
        
        # Start workers based on mode
        if self.processing_mode in [ProcessingMode.MULTI_THREAD, ProcessingMode.DISTRIBUTED]:
            for worker in self.workers:
                worker.start()
            self.logger.info(f"Started {len(self.workers)} workers in {self.processing_mode.value} mode")
        
        # Initialize fallback executors
        if self.processing_mode == ProcessingMode.MULTI_PROCESS:
            self.process_executor = ProcessPoolExecutor(max_workers=self.num_workers)
        else:
            self.thread_executor = ThreadPoolExecutor(max_workers=self.num_workers)
    
    def stop(self):
        """Stop distributed processing."""
        self.running = False
        
        # Stop workers
        for worker in self.workers:
            worker.stop()
        
        # Shutdown executors
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
        
        self.logger.info("Distributed processing stopped")
    
    def submit_task(self, function_name: str, *args, **kwargs) -> str:
        """Submit a task for distributed processing."""
        task_id = f"task_{int(time.time() * 1000000)}"
        
        task = ProcessingTask(
            task_id=task_id,
            function_name=function_name,
            args=args,
            kwargs=kwargs
        )
        
        if self.processing_mode == ProcessingMode.SINGLE_THREAD:
            # Execute immediately in single thread mode
            self._execute_immediately(task)
        else:
            # Queue for worker processing
            success = self.task_queue.put(task)
            if not success:
                raise RuntimeError("Failed to queue task: queue is full")
        
        return task_id
    
    def _execute_immediately(self, task: ProcessingTask):
        """Execute task immediately in single-thread mode."""
        try:
            func = self.function_registry.get(task.function_name)
            if not func:
                raise ValueError(f"Function '{task.function_name}' not registered")
            
            task.status = TaskStatus.RUNNING
            task.started_time = time.time()
            
            result = func(*task.args, **task.kwargs)
            
            task.status = TaskStatus.COMPLETED
            task.completed_time = time.time()
            task.result = result
            
            self.task_queue.update_task(task)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_time = time.time()
            self.task_queue.update_task(task)
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of a completed task."""
        start_time = time.time()
        
        while True:
            task = self.task_queue.get_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise RuntimeError(f"Task failed: {task.error}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            time.sleep(0.1)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        return self.task_queue.get_task_status(task_id)
    
    def distributed_function(self, func_name: str = None):
        """Decorator for creating distributed functions."""
        def decorator(func: Callable):
            name = func_name or func.__name__
            self.register_function(name, func)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.running:
                    # Fallback to direct execution if not running
                    return func(*args, **kwargs)
                
                task_id = self.submit_task(name, *args, **kwargs)
                return self.get_task_result(task_id)
            
            return wrapper
        return decorator
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        worker_stats = [worker.get_stats() for worker in self.workers]
        queue_stats = self.task_queue.get_stats()
        
        total_completed = sum(stats['tasks_completed'] for stats in worker_stats)
        total_failed = sum(stats['tasks_failed'] for stats in worker_stats)
        uptime = time.time() - self.start_time
        
        return {
            'running': self.running,
            'processing_mode': self.processing_mode.value,
            'uptime_seconds': uptime,
            'num_workers': len(self.workers),
            'registered_functions': list(self.function_registry.keys()),
            'queue_stats': queue_stats,
            'worker_stats': worker_stats,
            'total_tasks_completed': total_completed,
            'total_tasks_failed': total_failed,
            'overall_throughput_per_minute': (total_completed / uptime) * 60 if uptime > 0 else 0
        }
    
    def scale_workers(self, new_count: int):
        """Scale the number of workers."""
        current_count = len(self.workers)
        
        if new_count > current_count:
            # Add workers
            for i in range(current_count, new_count):
                worker = LiteWorker(f"worker_{i}", self.task_queue)
                
                # Register functions
                for name, func in self.function_registry.items():
                    worker.register_function(name, func)
                
                self.workers.append(worker)
                
                if self.running:
                    worker.start()
            
            self.logger.info(f"Scaled up from {current_count} to {new_count} workers")
            
        elif new_count < current_count:
            # Remove workers
            workers_to_remove = self.workers[new_count:]
            self.workers = self.workers[:new_count]
            
            for worker in workers_to_remove:
                worker.stop()
            
            self.logger.info(f"Scaled down from {current_count} to {new_count} workers")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        healthy_workers = sum(1 for worker in self.workers if worker.running)
        
        return {
            'status': 'healthy' if healthy_workers > 0 else 'unhealthy',
            'healthy_workers': healthy_workers,
            'total_workers': len(self.workers),
            'queue_size': self.task_queue.size(),
            'function_count': len(self.function_registry)
        }


# Global distributed manager
global_manager = DistributedManager()


def get_distributed_manager() -> DistributedManager:
    """Get global distributed manager."""
    return global_manager


def start_distributed_processing(num_workers: int = 4, mode: ProcessingMode = ProcessingMode.MULTI_THREAD):
    """Start distributed processing."""
    global_manager.num_workers = num_workers
    global_manager.processing_mode = mode
    global_manager._initialize_workers()
    global_manager.start()


def stop_distributed_processing():
    """Stop distributed processing."""
    global_manager.stop()


def distributed_function(func_name: str = None):
    """Decorator for distributed functions."""
    return global_manager.distributed_function(func_name)


if __name__ == "__main__":
    # Test distributed processing
    manager = get_distributed_manager()
    
    # Register test function
    @manager.distributed_function("test_computation")
    def compute_square(n: int) -> int:
        """Test computation function."""
        time.sleep(0.1)  # Simulate work
        return n * n
    
    # Start processing
    manager.start()
    
    # Submit tasks
    print("Submitting tasks...")
    task_ids = []
    for i in range(10):
        task_id = manager.submit_task("test_computation", i)
        task_ids.append(task_id)
    
    # Get results
    print("Getting results...")
    results = []
    for task_id in task_ids:
        try:
            result = manager.get_task_result(task_id, timeout=5.0)
            results.append(result)
        except Exception as e:
            print(f"Task failed: {e}")
    
    print(f"Results: {results}")
    
    # Get stats
    stats = manager.get_system_stats()
    print(f"\nSystem Stats:")
    print(json.dumps(stats, indent=2))
    
    # Stop processing
    manager.stop()