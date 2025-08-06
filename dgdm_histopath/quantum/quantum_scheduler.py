"""
Quantum-Enhanced Scheduler for DGDM Training and Inference Workflows.

Implements quantum-inspired scheduling algorithms for optimal resource allocation
and task orchestration in medical AI pipelines.
"""

import asyncio
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import Future
import psutil
import GPUtil

from dgdm_histopath.quantum.quantum_planner import QuantumTaskPlanner, Task, TaskPriority, ResourceType
from dgdm_histopath.utils.logging import get_logger
from dgdm_histopath.utils.monitoring import monitor_operation


class SchedulingStrategy(Enum):
    """Quantum-inspired scheduling strategies."""
    QUANTUM_ANNEALING = "quantum_annealing"
    SUPERPOSITION = "superposition"
    ENTANGLEMENT = "entanglement"
    HYBRID_CLASSICAL = "hybrid_classical"


class WorkflowType(Enum):
    """Types of medical AI workflows."""
    TRAINING = "training"
    INFERENCE = "inference"
    PREPROCESSING = "preprocessing"
    EVALUATION = "evaluation"
    FEDERATED_LEARNING = "federated_learning"


@dataclass
class ScheduledJob:
    """A scheduled job in the quantum scheduler."""
    id: str
    workflow_type: WorkflowType
    function: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    estimated_duration: float
    resource_requirements: Dict[ResourceType, float]
    scheduled_time: float
    dependencies: List[str]
    max_retries: int = 3
    retry_count: int = 0
    result: Optional[Any] = None
    error: Optional[Exception] = None


class QuantumScheduler:
    """
    Quantum-enhanced scheduler for medical AI workflows.
    
    Combines quantum task planning with real-time system monitoring
    to optimize resource utilization and minimize execution time.
    """
    
    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.QUANTUM_ANNEALING,
        max_concurrent_jobs: int = 4,
        resource_monitoring_interval: float = 1.0,
        enable_gpu_scheduling: bool = True,
        adaptive_scheduling: bool = True
    ):
        self.logger = get_logger(__name__)
        
        # Configuration
        self.strategy = strategy
        self.max_concurrent_jobs = max_concurrent_jobs
        self.resource_monitoring_interval = resource_monitoring_interval
        self.enable_gpu_scheduling = enable_gpu_scheduling
        self.adaptive_scheduling = adaptive_scheduling
        
        # Core components
        self.quantum_planner = QuantumTaskPlanner(
            max_concurrent_tasks=max_concurrent_jobs,
            resource_monitoring_interval=resource_monitoring_interval
        )
        
        # Job management
        self.scheduled_jobs: Dict[str, ScheduledJob] = {}
        self.running_jobs: Dict[str, Future] = {}
        self.completed_jobs: List[str] = []
        self.failed_jobs: List[str] = []
        
        # Resource monitoring
        self.cpu_usage_history: List[float] = []
        self.memory_usage_history: List[float] = []
        self.gpu_usage_history: List[float] = []
        
        # Performance metrics
        self.total_jobs_processed = 0
        self.average_job_duration = 0.0
        self.scheduler_efficiency = 0.0
        
        # Threading
        self.monitoring_thread = None
        self.scheduling_thread = None
        self.shutdown_event = threading.Event()
        
        # Start background monitoring
        self._start_monitoring()
        
        self.logger.info(f"QuantumScheduler initialized with strategy: {strategy.value}")
    
    def _start_monitoring(self):
        """Start background resource monitoring thread."""
        def monitor_resources():
            while not self.shutdown_event.is_set():
                try:
                    # Monitor CPU and memory
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory_percent = psutil.virtual_memory().percent
                    
                    self.cpu_usage_history.append(cpu_percent / 100.0)
                    self.memory_usage_history.append(memory_percent / 100.0)
                    
                    # Monitor GPU if available
                    if self.enable_gpu_scheduling:
                        try:
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                avg_gpu_load = np.mean([gpu.load for gpu in gpus])
                                self.gpu_usage_history.append(avg_gpu_load)
                            else:
                                self.gpu_usage_history.append(0.0)
                        except:
                            self.gpu_usage_history.append(0.0)
                    
                    # Keep history limited
                    max_history = 100
                    for history in [self.cpu_usage_history, self.memory_usage_history, self.gpu_usage_history]:
                        if len(history) > max_history:
                            history.pop(0)
                    
                    # Update quantum planner resource state
                    self._update_planner_resources()
                    
                    time.sleep(self.resource_monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    time.sleep(self.resource_monitoring_interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitoring_thread.start()
        
        # Start scheduling thread
        self.scheduling_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduling_thread.start()
    
    def _update_planner_resources(self):
        """Update quantum planner with current resource state."""
        if self.cpu_usage_history:
            self.quantum_planner.resource_state.cpu_usage = self.cpu_usage_history[-1]
        if self.memory_usage_history:
            self.quantum_planner.resource_state.memory_usage = self.memory_usage_history[-1]
        if self.gpu_usage_history and self.enable_gpu_scheduling:
            self.quantum_planner.resource_state.gpu_usage = self.gpu_usage_history[-1]
    
    @monitor_operation("schedule_job")
    def schedule_job(
        self,
        job_id: str,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        workflow_type: WorkflowType = WorkflowType.INFERENCE,
        priority: TaskPriority = TaskPriority.NORMAL,
        estimated_duration: float = 60.0,
        resource_requirements: Dict[ResourceType, float] = None,
        dependencies: List[str] = None,
        scheduled_time: float = None,
        max_retries: int = 3
    ) -> ScheduledJob:
        """Schedule a job for quantum-optimized execution."""
        if job_id in self.scheduled_jobs:
            raise ValueError(f"Job {job_id} already scheduled")
        
        # Default resource requirements based on workflow type
        if resource_requirements is None:
            resource_requirements = self._get_default_resources(workflow_type)
        
        # Default scheduling time
        if scheduled_time is None:
            scheduled_time = time.time()
        
        # Create scheduled job
        job = ScheduledJob(
            id=job_id,
            workflow_type=workflow_type,
            function=function,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements,
            scheduled_time=scheduled_time,
            dependencies=dependencies or [],
            max_retries=max_retries
        )
        
        # Add to quantum planner as task
        self.quantum_planner.add_task(
            task_id=job_id,
            name=f"{workflow_type.value}_{job_id}",
            priority=priority,
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements,
            dependencies=dependencies
        )
        
        self.scheduled_jobs[job_id] = job
        self.logger.info(f"Scheduled job {job_id} with priority {priority.value}")
        
        return job
    
    def _get_default_resources(self, workflow_type: WorkflowType) -> Dict[ResourceType, float]:
        """Get default resource requirements for workflow type."""
        defaults = {
            WorkflowType.TRAINING: {
                ResourceType.CPU: 0.8,
                ResourceType.GPU: 0.9,
                ResourceType.MEMORY: 0.7,
                ResourceType.STORAGE: 0.3,
                ResourceType.NETWORK: 0.2,
            },
            WorkflowType.INFERENCE: {
                ResourceType.CPU: 0.4,
                ResourceType.GPU: 0.6,
                ResourceType.MEMORY: 0.3,
                ResourceType.STORAGE: 0.1,
                ResourceType.NETWORK: 0.1,
            },
            WorkflowType.PREPROCESSING: {
                ResourceType.CPU: 0.6,
                ResourceType.GPU: 0.3,
                ResourceType.MEMORY: 0.5,
                ResourceType.STORAGE: 0.8,
                ResourceType.NETWORK: 0.1,
            },
            WorkflowType.EVALUATION: {
                ResourceType.CPU: 0.3,
                ResourceType.GPU: 0.4,
                ResourceType.MEMORY: 0.2,
                ResourceType.STORAGE: 0.2,
                ResourceType.NETWORK: 0.1,
            },
            WorkflowType.FEDERATED_LEARNING: {
                ResourceType.CPU: 0.5,
                ResourceType.GPU: 0.7,
                ResourceType.MEMORY: 0.4,
                ResourceType.STORAGE: 0.2,
                ResourceType.NETWORK: 0.9,
            },
        }
        
        return defaults.get(workflow_type, {
            ResourceType.CPU: 0.3,
            ResourceType.GPU: 0.3,
            ResourceType.MEMORY: 0.2,
            ResourceType.STORAGE: 0.1,
            ResourceType.NETWORK: 0.1,
        })
    
    def _run_scheduler(self):
        """Main scheduler loop running in background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Run quantum optimization
                    future = asyncio.ensure_future(self._quantum_schedule_cycle())
                    loop.run_until_complete(future)
                    
                    # Brief pause between scheduling cycles
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Scheduler cycle error: {e}")
                    time.sleep(1.0)
        finally:
            loop.close()
    
    async def _quantum_schedule_cycle(self):
        """Single cycle of quantum scheduling optimization."""
        # Check if we have jobs ready to schedule
        ready_jobs = self._get_ready_jobs()
        if not ready_jobs:
            return
        
        # Get optimal execution order from quantum planner
        try:
            optimal_schedule = await self.quantum_planner.quantum_optimize_schedule()
            
            # Execute jobs according to quantum-optimized schedule
            for task_id in optimal_schedule[:self.max_concurrent_jobs - len(self.running_jobs)]:
                if task_id in ready_jobs and task_id not in self.running_jobs:
                    await self._execute_job(task_id)
                    
        except Exception as e:
            self.logger.error(f"Quantum scheduling error: {e}")
    
    def _get_ready_jobs(self) -> List[str]:
        """Get list of jobs ready for execution."""
        ready = []
        current_time = time.time()
        
        for job_id, job in self.scheduled_jobs.items():
            if (job_id not in self.running_jobs and 
                job_id not in self.completed_jobs and
                job_id not in self.failed_jobs and
                job.scheduled_time <= current_time):
                
                # Check dependencies
                deps_satisfied = all(dep in self.completed_jobs for dep in job.dependencies)
                if deps_satisfied:
                    ready.append(job_id)
        
        return ready
    
    async def _execute_job(self, job_id: str):
        """Execute a scheduled job."""
        job = self.scheduled_jobs[job_id]
        
        self.logger.info(f"Starting execution of job {job_id}")
        
        # Create async task for job execution
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            None, 
            self._run_job_function, 
            job
        )
        
        self.running_jobs[job_id] = future
        
        # Handle job completion asynchronously
        future.add_done_callback(lambda f: self._handle_job_completion(job_id, f))
    
    def _run_job_function(self, job: ScheduledJob) -> Any:
        """Run the actual job function."""
        start_time = time.time()
        
        try:
            # Execute the function
            result = job.function(*job.args, **job.kwargs)
            
            # Record successful completion
            execution_time = time.time() - start_time
            job.result = result
            
            self.logger.info(f"Job {job.id} completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Job {job.id} failed: {str(e)}")
            job.error = e
            raise e
    
    def _handle_job_completion(self, job_id: str, future: Future):
        """Handle job completion or failure."""
        job = self.scheduled_jobs[job_id]
        
        try:
            # Get result (may raise exception)
            result = future.result()
            
            # Successful completion
            self.completed_jobs.append(job_id)
            self.total_jobs_processed += 1
            
            # Update performance metrics
            if job.result is not None:
                self._update_performance_metrics(job)
            
        except Exception as e:
            # Job failed
            job.retry_count += 1
            
            if job.retry_count <= job.max_retries:
                # Reschedule for retry
                job.scheduled_time = time.time() + (2 ** job.retry_count)  # Exponential backoff
                self.logger.warning(f"Retrying job {job_id} (attempt {job.retry_count}/{job.max_retries})")
            else:
                # Max retries exceeded
                self.failed_jobs.append(job_id)
                self.logger.error(f"Job {job_id} failed after {job.max_retries} retries")
        
        finally:
            # Clean up running jobs
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    def _update_performance_metrics(self, job: ScheduledJob):
        """Update scheduler performance metrics."""
        # Update average job duration
        if hasattr(job, 'execution_time'):
            execution_time = job.execution_time
        else:
            execution_time = job.estimated_duration  # Fallback
        
        if self.total_jobs_processed == 1:
            self.average_job_duration = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_job_duration = (alpha * execution_time + 
                                       (1 - alpha) * self.average_job_duration)
        
        # Calculate scheduler efficiency
        if self.cpu_usage_history and self.memory_usage_history:
            avg_cpu = np.mean(self.cpu_usage_history[-10:])  # Last 10 measurements
            avg_memory = np.mean(self.memory_usage_history[-10:])
            
            # Efficiency is inverse of resource waste
            resource_utilization = (avg_cpu + avg_memory) / 2
            self.scheduler_efficiency = min(1.0, resource_utilization)
    
    @monitor_operation("wait_for_job")
    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a specific job to complete and return its result."""
        if job_id not in self.scheduled_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        start_time = time.time()
        
        while True:
            if job_id in self.completed_jobs:
                return self.scheduled_jobs[job_id].result
            
            if job_id in self.failed_jobs:
                error = self.scheduled_jobs[job_id].error
                raise RuntimeError(f"Job {job_id} failed: {error}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
            
            time.sleep(0.1)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive status of a specific job."""
        if job_id not in self.scheduled_jobs:
            return {"status": "not_found"}
        
        job = self.scheduled_jobs[job_id]
        
        if job_id in self.completed_jobs:
            status = "completed"
        elif job_id in self.failed_jobs:
            status = "failed"
        elif job_id in self.running_jobs:
            status = "running"
        else:
            status = "queued"
        
        return {
            "status": status,
            "workflow_type": job.workflow_type.value,
            "priority": job.priority.value,
            "estimated_duration": job.estimated_duration,
            "scheduled_time": job.scheduled_time,
            "retry_count": job.retry_count,
            "max_retries": job.max_retries,
            "dependencies": job.dependencies,
            "resource_requirements": {rt.value: req for rt, req in job.resource_requirements.items()},
            "error": str(job.error) if job.error else None
        }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status."""
        return {
            "strategy": self.strategy.value,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "total_jobs": len(self.scheduled_jobs),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "queued_jobs": len(self.scheduled_jobs) - len(self.running_jobs) - len(self.completed_jobs) - len(self.failed_jobs),
            "performance_metrics": {
                "total_jobs_processed": self.total_jobs_processed,
                "average_job_duration": self.average_job_duration,
                "scheduler_efficiency": self.scheduler_efficiency,
            },
            "resource_state": {
                "cpu_usage": self.cpu_usage_history[-1] if self.cpu_usage_history else 0.0,
                "memory_usage": self.memory_usage_history[-1] if self.memory_usage_history else 0.0,
                "gpu_usage": self.gpu_usage_history[-1] if self.gpu_usage_history and self.enable_gpu_scheduling else 0.0,
            },
            "quantum_planner_status": self.quantum_planner.get_status()
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled or running job."""
        if job_id not in self.scheduled_jobs:
            return False
        
        if job_id in self.running_jobs:
            # Cancel running job
            future = self.running_jobs[job_id]
            cancelled = future.cancel()
            if cancelled:
                del self.running_jobs[job_id]
                self.failed_jobs.append(job_id)
                self.logger.info(f"Cancelled running job {job_id}")
            return cancelled
        
        elif job_id not in self.completed_jobs and job_id not in self.failed_jobs:
            # Remove from scheduled jobs
            self.failed_jobs.append(job_id)
            self.logger.info(f"Cancelled scheduled job {job_id}")
            return True
        
        return False
    
    async def shutdown(self):
        """Gracefully shutdown the quantum scheduler."""
        self.logger.info("Shutting down QuantumScheduler")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel all running jobs
        for job_id, future in self.running_jobs.items():
            future.cancel()
            self.logger.info(f"Cancelled job {job_id} during shutdown")
        
        # Wait for threads to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        if self.scheduling_thread and self.scheduling_thread.is_alive():
            self.scheduling_thread.join(timeout=5.0)
        
        # Shutdown quantum planner
        await self.quantum_planner.shutdown()
        
        self.logger.info("QuantumScheduler shutdown complete")