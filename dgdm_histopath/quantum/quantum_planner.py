"""
Quantum-Inspired Task Planner for autonomous medical AI workflows.

Leverages quantum computing principles including superposition, entanglement,
and quantum annealing for optimal task scheduling and resource allocation.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import time

from dgdm_histopath.utils.logging import get_logger
from dgdm_histopath.utils.monitoring import monitor_operation
from dgdm_histopath.utils.validation import InputValidator


class TaskPriority(Enum):
    """Task priority levels using quantum-inspired states."""
    CRITICAL = "critical"      # |1⟩ state - immediate execution
    HIGH = "high"             # |+⟩ superposition state  
    NORMAL = "normal"         # |0⟩ ground state
    LOW = "low"              # |−⟩ mixed state


class ResourceType(Enum):
    """Available computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class Task:
    """Represents a computational task with quantum properties."""
    id: str
    name: str
    priority: TaskPriority
    estimated_duration: float  # in seconds
    resource_requirements: Dict[ResourceType, float]
    dependencies: List[str] = field(default_factory=list)
    quantum_state: complex = field(default=1.0+0j)  # Quantum amplitude
    entangled_tasks: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def is_completed(self) -> bool:
        return self.completed_at is not None
    
    @property
    def is_running(self) -> bool:
        return self.started_at is not None and not self.is_completed
    
    @property
    def execution_time(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class ResourceState:
    """Current state of computational resources."""
    cpu_usage: float = 0.0      # 0.0 to 1.0
    gpu_usage: float = 0.0      # 0.0 to 1.0  
    memory_usage: float = 0.0   # 0.0 to 1.0
    storage_usage: float = 0.0  # 0.0 to 1.0
    network_usage: float = 0.0  # 0.0 to 1.0
    
    def can_allocate(self, requirements: Dict[ResourceType, float]) -> bool:
        """Check if resources can be allocated for task."""
        resource_map = {
            ResourceType.CPU: self.cpu_usage,
            ResourceType.GPU: self.gpu_usage,
            ResourceType.MEMORY: self.memory_usage,
            ResourceType.STORAGE: self.storage_usage,
            ResourceType.NETWORK: self.network_usage,
        }
        
        for resource_type, required in requirements.items():
            current = resource_map.get(resource_type, 0.0)
            if current + required > 1.0:
                return False
        return True


class QuantumTaskPlanner:
    """
    Quantum-inspired autonomous task planner for medical AI workflows.
    
    Uses quantum computing principles to optimize task scheduling:
    - Superposition: Tasks exist in multiple execution states
    - Entanglement: Dependent tasks affect each other's scheduling  
    - Quantum Annealing: Optimal scheduling through energy minimization
    - Measurement: Collapse to optimal execution plan
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 8,
        resource_monitoring_interval: float = 1.0,
        quantum_coherence_time: float = 10.0,
        enable_quantum_acceleration: bool = True
    ):
        self.logger = get_logger(__name__)
        self.validator = InputValidator()
        
        # Core planning parameters
        self.max_concurrent_tasks = max_concurrent_tasks
        self.resource_monitoring_interval = resource_monitoring_interval
        self.quantum_coherence_time = quantum_coherence_time
        self.enable_quantum_acceleration = enable_quantum_acceleration
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[str] = []
        
        # Resource management
        self.resource_state = ResourceState()
        self.resource_history: List[ResourceState] = []
        
        # Quantum state management
        self.quantum_register = np.zeros((32,), dtype=complex)  # 32-qubit register
        self.entanglement_matrix = np.eye(32, dtype=complex)
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.total_execution_time = 0.0
        self.optimization_score = 0.0
        
        # Executors for parallel processing
        self.thread_executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.process_executor = ProcessPoolExecutor(max_workers=max_concurrent_tasks//2)
        
        self.logger.info(f"QuantumTaskPlanner initialized with {max_concurrent_tasks} concurrent tasks")
    
    @monitor_operation("add_task")
    def add_task(
        self,
        task_id: str,
        name: str,
        priority: TaskPriority,
        estimated_duration: float,
        resource_requirements: Dict[ResourceType, float],
        dependencies: Optional[List[str]] = None,
        entangled_tasks: Optional[List[str]] = None
    ) -> Task:
        """Add a new task to the quantum planning system."""
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")
        
        # Validate inputs
        self.validator.validate_positive_number(estimated_duration, "estimated_duration")
        for resource_type, requirement in resource_requirements.items():
            self.validator.validate_range(requirement, 0.0, 1.0, f"resource_requirement_{resource_type.value}")
        
        # Create task with quantum properties
        task = Task(
            id=task_id,
            name=name,
            priority=priority,
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements,
            dependencies=dependencies or [],
            entangled_tasks=entangled_tasks or [],
            quantum_state=self._calculate_quantum_state(priority, estimated_duration)
        )
        
        # Add to quantum register
        self._update_quantum_register(task)
        
        # Store task
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        self.logger.info(f"Added task {task_id} with priority {priority.value}")
        return task
    
    def _calculate_quantum_state(self, priority: TaskPriority, duration: float) -> complex:
        """Calculate quantum state amplitude based on task properties."""
        # Map priority to quantum amplitude
        priority_amplitudes = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 0.8,
            TaskPriority.NORMAL: 0.6,
            TaskPriority.LOW: 0.4,
        }
        
        # Duration affects phase
        phase = -math.log(duration + 1) / 10  # Shorter tasks get less negative phase
        amplitude = priority_amplitudes[priority]
        
        return amplitude * math.cos(phase) + 1j * amplitude * math.sin(phase)
    
    def _update_quantum_register(self, task: Task):
        """Update quantum register with new task state."""
        # Hash task ID to qubit index
        qubit_idx = hash(task.id) % len(self.quantum_register)
        
        # Set qubit state
        self.quantum_register[qubit_idx] = task.quantum_state
        
        # Handle entanglement
        for entangled_id in task.entangled_tasks:
            if entangled_id in self.tasks:
                entangled_qubit = hash(entangled_id) % len(self.quantum_register)
                # Create entanglement through CNOT-like operation
                self.entanglement_matrix[qubit_idx, entangled_qubit] = 0.7071 + 0.7071j
    
    @monitor_operation("quantum_optimize_schedule")
    async def quantum_optimize_schedule(self) -> List[str]:
        """
        Use quantum annealing to find optimal task execution order.
        
        Returns optimized task execution sequence.
        """
        if not self.task_queue:
            return []
        
        self.logger.info("Starting quantum optimization of task schedule")
        
        # Create Hamiltonian matrix for optimization
        n_tasks = len(self.task_queue)
        hamiltonian = self._create_scheduling_hamiltonian()
        
        # Quantum annealing simulation
        optimized_order = await self._quantum_anneal(hamiltonian)
        
        # Validate and repair schedule for dependencies
        valid_schedule = self._repair_schedule_dependencies(optimized_order)
        
        # Calculate optimization score
        self.optimization_score = self._calculate_schedule_score(valid_schedule)
        
        self.logger.info(f"Quantum optimization complete. Score: {self.optimization_score:.3f}")
        return valid_schedule
    
    def _create_scheduling_hamiltonian(self) -> np.ndarray:
        """Create Hamiltonian matrix for quantum annealing optimization."""
        n_tasks = len(self.task_queue)
        H = np.zeros((n_tasks, n_tasks), dtype=complex)
        
        for i, task_id in enumerate(self.task_queue):
            task = self.tasks[task_id]
            
            # Diagonal terms (task priorities and resource costs)
            priority_weight = {
                TaskPriority.CRITICAL: -10.0,
                TaskPriority.HIGH: -5.0,
                TaskPriority.NORMAL: -1.0,
                TaskPriority.LOW: 0.0,
            }[task.priority]
            
            resource_cost = sum(task.resource_requirements.values())
            H[i, i] = priority_weight - resource_cost
            
            # Off-diagonal terms (task interactions)
            for j, other_task_id in enumerate(self.task_queue):
                if i != j:
                    other_task = self.tasks[other_task_id]
                    
                    # Dependency constraints
                    if task_id in other_task.dependencies:
                        H[i, j] = -100.0  # Strong coupling for dependencies
                    
                    # Entanglement effects
                    if other_task_id in task.entangled_tasks:
                        H[i, j] = -2.0 * abs(task.quantum_state * other_task.quantum_state.conjugate())
        
        return H
    
    async def _quantum_anneal(self, hamiltonian: np.ndarray) -> List[str]:
        """Simulate quantum annealing to find optimal solution."""
        n_tasks = len(self.task_queue)
        
        # Initialize with superposition state
        state = np.ones(n_tasks, dtype=complex) / math.sqrt(n_tasks)
        
        # Annealing parameters
        n_steps = 100
        initial_field = 10.0
        final_field = 0.01
        
        for step in range(n_steps):
            # Linear annealing schedule
            s = step / (n_steps - 1)
            transverse_field = initial_field * (1 - s) + final_field * s
            
            # Time evolution operator
            H_total = hamiltonian - transverse_field * np.eye(n_tasks)
            dt = 0.1
            evolution_op = scipy.linalg.expm(-1j * H_total * dt)
            
            # Evolve state
            state = evolution_op @ state
            
            # Add small amount of decoherence
            state += 0.001 * np.random.normal(0, 1, n_tasks) * (1 + 1j)
            state /= np.linalg.norm(state)
            
            await asyncio.sleep(0.001)  # Yield control
        
        # Measure final state (collapse to classical solution)
        probabilities = np.abs(state) ** 2
        
        # Generate order based on measurement probabilities
        task_indices = np.argsort(-probabilities)
        return [self.task_queue[i] for i in task_indices]
    
    def _repair_schedule_dependencies(self, schedule: List[str]) -> List[str]:
        """Repair schedule to satisfy dependency constraints."""
        repaired = []
        remaining = schedule.copy()
        
        while remaining:
            # Find tasks with satisfied dependencies
            ready_tasks = []
            for task_id in remaining:
                task = self.tasks[task_id]
                if all(dep in repaired for dep in task.dependencies):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Handle circular dependencies by picking highest priority task
                task_priorities = [(self.tasks[tid].priority, tid) for tid in remaining]
                task_priorities.sort(key=lambda x: list(TaskPriority).index(x[0]))
                ready_tasks = [task_priorities[0][1]]
            
            # Add first ready task
            next_task = ready_tasks[0]
            repaired.append(next_task)
            remaining.remove(next_task)
        
        return repaired
    
    def _calculate_schedule_score(self, schedule: List[str]) -> float:
        """Calculate optimization score for a given schedule."""
        score = 0.0
        simulated_time = 0.0
        simulated_resources = ResourceState()
        
        for task_id in schedule:
            task = self.tasks[task_id]
            
            # Priority bonus
            priority_bonus = {
                TaskPriority.CRITICAL: 10.0,
                TaskPriority.HIGH: 5.0,
                TaskPriority.NORMAL: 1.0,
                TaskPriority.LOW: 0.0,
            }[task.priority]
            
            # Resource utilization penalty
            resource_penalty = sum(task.resource_requirements.values()) * 0.1
            
            # Dependency satisfaction bonus
            dependency_bonus = 2.0 if all(dep in schedule[:schedule.index(task_id)] 
                                         for dep in task.dependencies) else -10.0
            
            score += priority_bonus - resource_penalty + dependency_bonus
            simulated_time += task.estimated_duration
        
        # Time efficiency bonus
        ideal_time = sum(self.tasks[tid].estimated_duration for tid in schedule) / self.max_concurrent_tasks
        time_efficiency = max(0, 2.0 - simulated_time / ideal_time)
        
        return score + time_efficiency
    
    @monitor_operation("execute_schedule")  
    async def execute_schedule(self, schedule: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute tasks according to optimized schedule."""
        if schedule is None:
            schedule = await self.quantum_optimize_schedule()
        
        if not schedule:
            return {"status": "success", "tasks_executed": 0, "total_time": 0.0}
        
        self.logger.info(f"Executing schedule with {len(schedule)} tasks")
        
        execution_results = {
            "status": "success",
            "tasks_executed": 0,
            "total_time": 0.0,
            "failed_tasks": [],
            "performance_metrics": {}
        }
        
        start_time = time.time()
        
        # Execute tasks with resource management
        for task_id in schedule:
            try:
                await self._execute_single_task(task_id)
                execution_results["tasks_executed"] += 1
                self.completed_tasks.append(task_id)
                
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {str(e)}")
                execution_results["failed_tasks"].append({"task_id": task_id, "error": str(e)})
        
        execution_results["total_time"] = time.time() - start_time
        execution_results["performance_metrics"] = self._calculate_performance_metrics()
        
        self.logger.info(f"Schedule execution complete: {execution_results['tasks_executed']} tasks in {execution_results['total_time']:.2f}s")
        return execution_results
    
    async def _execute_single_task(self, task_id: str):
        """Execute a single task with resource management."""
        task = self.tasks[task_id]
        
        # Wait for resource availability
        while not self.resource_state.can_allocate(task.resource_requirements):
            await asyncio.sleep(0.1)
            self._update_resource_state()
        
        # Allocate resources
        self._allocate_resources(task)
        
        # Execute task
        task.started_at = time.time()
        self.logger.debug(f"Starting execution of task {task_id}")
        
        try:
            # Simulate task execution (replace with actual task logic)
            await asyncio.sleep(task.estimated_duration * 0.1)  # Accelerated simulation
            
            task.completed_at = time.time()
            self.total_tasks_processed += 1
            self.total_execution_time += task.execution_time or 0
            
            self.logger.debug(f"Completed task {task_id} in {task.execution_time:.2f}s")
            
        finally:
            # Release resources
            self._release_resources(task)
    
    def _allocate_resources(self, task: Task):
        """Allocate resources for task execution."""
        for resource_type, requirement in task.resource_requirements.items():
            if resource_type == ResourceType.CPU:
                self.resource_state.cpu_usage += requirement
            elif resource_type == ResourceType.GPU:
                self.resource_state.gpu_usage += requirement
            elif resource_type == ResourceType.MEMORY:
                self.resource_state.memory_usage += requirement
            elif resource_type == ResourceType.STORAGE:
                self.resource_state.storage_usage += requirement
            elif resource_type == ResourceType.NETWORK:
                self.resource_state.network_usage += requirement
    
    def _release_resources(self, task: Task):
        """Release resources after task completion."""
        for resource_type, requirement in task.resource_requirements.items():
            if resource_type == ResourceType.CPU:
                self.resource_state.cpu_usage = max(0, self.resource_state.cpu_usage - requirement)
            elif resource_type == ResourceType.GPU:
                self.resource_state.gpu_usage = max(0, self.resource_state.gpu_usage - requirement)
            elif resource_type == ResourceType.MEMORY:
                self.resource_state.memory_usage = max(0, self.resource_state.memory_usage - requirement)
            elif resource_type == ResourceType.STORAGE:
                self.resource_state.storage_usage = max(0, self.resource_state.storage_usage - requirement)
            elif resource_type == ResourceType.NETWORK:
                self.resource_state.network_usage = max(0, self.resource_state.network_usage - requirement)
    
    def _update_resource_state(self):
        """Update current resource utilization state."""
        # In production, this would query actual system resources
        # For now, simulate gradual resource recovery
        decay_factor = 0.95
        self.resource_state.cpu_usage *= decay_factor
        self.resource_state.gpu_usage *= decay_factor
        self.resource_state.memory_usage *= decay_factor
        self.resource_state.storage_usage *= decay_factor
        self.resource_state.network_usage *= decay_factor
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if self.total_tasks_processed == 0:
            return {}
        
        avg_execution_time = self.total_execution_time / self.total_tasks_processed
        
        return {
            "average_execution_time": avg_execution_time,
            "total_tasks_processed": self.total_tasks_processed,
            "optimization_score": self.optimization_score,
            "quantum_coherence": abs(np.sum(self.quantum_register)) / len(self.quantum_register),
            "resource_efficiency": 1.0 - np.mean([
                self.resource_state.cpu_usage,
                self.resource_state.gpu_usage,
                self.resource_state.memory_usage,
                self.resource_state.storage_usage,
                self.resource_state.network_usage,
            ])
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive planner status."""
        return {
            "total_tasks": len(self.tasks),
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "resource_state": {
                "cpu_usage": self.resource_state.cpu_usage,
                "gpu_usage": self.resource_state.gpu_usage,
                "memory_usage": self.resource_state.memory_usage,
                "storage_usage": self.resource_state.storage_usage,
                "network_usage": self.resource_state.network_usage,
            },
            "quantum_state": {
                "register_coherence": abs(np.sum(self.quantum_register)) / len(self.quantum_register),
                "entanglement_strength": np.trace(self.entanglement_matrix).real,
            },
            "performance_metrics": self._calculate_performance_metrics()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the quantum task planner."""
        self.logger.info("Shutting down QuantumTaskPlanner")
        
        # Cancel running tasks
        for task_future in self.running_tasks.values():
            task_future.cancel()
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        self.logger.info("QuantumTaskPlanner shutdown complete")


# Compatibility imports for scipy if needed
try:
    import scipy.linalg
except ImportError:
    # Fallback implementation
    def _matrix_exp_fallback(A, dt):
        """Simple fallback for matrix exponential."""
        return np.eye(A.shape[0]) + A * dt  # First-order approximation
    
    class _ScipyFallback:
        @staticmethod
        def expm(A):
            return _matrix_exp_fallback(A, 1.0)
    
    scipy = type('scipy', (), {'linalg': _ScipyFallback})()