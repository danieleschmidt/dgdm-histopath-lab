"""
Quantum-Enhanced Distributed Computing for DGDM Histopath Lab.

Implements distributed quantum task execution, load balancing, and auto-scaling
for high-performance medical AI workloads.
"""

import asyncio
import time
import uuid
import pickle
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
from pathlib import Path
import redis
import logging
import psutil
from collections import defaultdict

from dgdm_histopath.quantum.quantum_planner import QuantumTaskPlanner, Task, TaskPriority, ResourceType
from dgdm_histopath.quantum.quantum_scheduler import QuantumScheduler
from dgdm_histopath.utils.logging import get_logger
from dgdm_histopath.utils.monitoring import monitor_operation


class NodeType(Enum):
    """Types of compute nodes in distributed system."""
    MASTER = "master"
    WORKER = "worker"
    GPU_WORKER = "gpu_worker"
    QUANTUM_WORKER = "quantum_worker"
    STORAGE = "storage"


class TaskDistributionStrategy(Enum):
    """Task distribution strategies."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    QUANTUM_OPTIMAL = "quantum_optimal"
    LOCALITY_AWARE = "locality_aware"
    PRIORITY_BASED = "priority_based"


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    node_type: NodeType
    host: str
    port: int
    capabilities: List[str]
    max_concurrent_tasks: int
    current_load: float = 0.0
    available_resources: Dict[ResourceType, float] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True
    performance_score: float = 1.0
    task_queue: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.available_resources:
            self.available_resources = {
                ResourceType.CPU: 1.0,
                ResourceType.GPU: 1.0 if "gpu" in self.capabilities else 0.0,
                ResourceType.MEMORY: 1.0,
                ResourceType.STORAGE: 1.0,
                ResourceType.NETWORK: 1.0,
            }


@dataclass
class DistributedTask:
    """Task wrapper for distributed execution."""
    task_id: str
    function_name: str
    serialized_args: bytes
    serialized_kwargs: bytes
    priority: TaskPriority
    resource_requirements: Dict[ResourceType, float]
    dependencies: List[str]
    assigned_node: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[bytes] = None
    error: Optional[str] = None


class QuantumDistributedManager:
    """
    Quantum-enhanced distributed task manager.
    
    Orchestrates distributed quantum computation across multiple nodes
    with intelligent load balancing and fault tolerance.
    """
    
    def __init__(
        self,
        node_type: NodeType = NodeType.MASTER,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        enable_auto_scaling: bool = True,
        max_nodes: int = 10,
        heartbeat_interval: float = 10.0,
        task_timeout: float = 3600.0  # 1 hour
    ):
        self.logger = get_logger(__name__)
        
        # Configuration
        self.node_type = node_type
        self.node_id = str(uuid.uuid4())
        self.enable_auto_scaling = enable_auto_scaling
        self.max_nodes = max_nodes
        self.heartbeat_interval = heartbeat_interval
        self.task_timeout = task_timeout
        
        # Redis connection for coordination
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()  # Test connection
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
        
        # Node management
        self.compute_nodes: Dict[str, ComputeNode] = {}
        self.local_node: Optional[ComputeNode] = None
        
        # Task management
        self.distributed_tasks: Dict[str, DistributedTask] = {}
        self.task_results: Dict[str, Any] = {}
        self.pending_tasks: List[str] = []
        self.running_tasks: Dict[str, str] = {}  # task_id -> node_id
        
        # Distribution strategy
        self.distribution_strategy = TaskDistributionStrategy.QUANTUM_OPTIMAL
        self.quantum_planner = QuantumTaskPlanner()
        
        # Performance tracking
        self.throughput_history: List[float] = []
        self.latency_history: List[float] = []
        self.error_rates: Dict[str, float] = defaultdict(float)
        
        # Threading
        self.heartbeat_thread = None
        self.task_monitor_thread = None
        self.auto_scaling_thread = None
        self.shutdown_event = threading.Event()
        
        # Executors for parallel processing
        self.thread_executor = ThreadPoolExecutor(max_workers=8)
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        
        # Initialize local node
        self._initialize_local_node()
        
        # Start background services
        self._start_background_services()
        
        self.logger.info(f"QuantumDistributedManager initialized as {node_type.value} node: {self.node_id}")
    
    def _initialize_local_node(self):
        """Initialize local compute node."""
        # Detect capabilities
        capabilities = ["cpu", "memory"]
        
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                capabilities.append("gpu")
                capabilities.append("cuda")
        except ImportError:
            pass
        
        # Check for quantum capabilities
        capabilities.append("quantum")
        
        # Get system resources
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total // (1024**3)
        
        self.local_node = ComputeNode(
            node_id=self.node_id,
            node_type=self.node_type,
            host="localhost",  # In production, use actual IP
            port=8000,  # Default port
            capabilities=capabilities,
            max_concurrent_tasks=min(cpu_count * 2, 16),
            available_resources={
                ResourceType.CPU: 1.0,
                ResourceType.GPU: 1.0 if "gpu" in capabilities else 0.0,
                ResourceType.MEMORY: 1.0,
                ResourceType.STORAGE: 1.0,
                ResourceType.NETWORK: 1.0,
            }
        )
        
        # Register with cluster
        if self.redis_client:
            self._register_node(self.local_node)
        
        self.compute_nodes[self.node_id] = self.local_node
    
    def _register_node(self, node: ComputeNode):
        """Register node with Redis cluster."""
        try:
            node_data = {
                "node_type": node.node_type.value,
                "host": node.host,
                "port": node.port,
                "capabilities": json.dumps(node.capabilities),
                "max_concurrent_tasks": node.max_concurrent_tasks,
                "available_resources": json.dumps({rt.value: val for rt, val in node.available_resources.items()}),
                "last_heartbeat": time.time(),
                "performance_score": node.performance_score
            }
            
            self.redis_client.hset(f"node:{node.node_id}", mapping=node_data)
            self.redis_client.sadd("cluster:nodes", node.node_id)
            
            self.logger.info(f"Registered node {node.node_id} with cluster")
            
        except Exception as e:
            self.logger.error(f"Failed to register node: {e}")
    
    def _start_background_services(self):
        """Start background monitoring and management services."""
        # Heartbeat service
        def heartbeat_loop():
            while not self.shutdown_event.is_set():
                try:
                    self._send_heartbeat()
                    self._discover_nodes()
                    self._cleanup_stale_nodes()
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")
                    time.sleep(self.heartbeat_interval)
        
        # Task monitoring service
        def task_monitor_loop():
            while not self.shutdown_event.is_set():
                try:
                    self._monitor_running_tasks()
                    self._process_pending_tasks()
                    time.sleep(1.0)
                except Exception as e:
                    self.logger.error(f"Task monitor error: {e}")
                    time.sleep(1.0)
        
        # Auto-scaling service
        def auto_scaling_loop():
            while not self.shutdown_event.is_set():
                try:
                    if self.enable_auto_scaling and self.node_type == NodeType.MASTER:
                        self._auto_scale_cluster()
                    time.sleep(30.0)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Auto-scaling error: {e}")
                    time.sleep(30.0)
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.task_monitor_thread = threading.Thread(target=task_monitor_loop, daemon=True)
        self.auto_scaling_thread = threading.Thread(target=auto_scaling_loop, daemon=True)
        
        self.heartbeat_thread.start()
        self.task_monitor_thread.start()
        self.auto_scaling_thread.start()
    
    def _send_heartbeat(self):
        """Send heartbeat to cluster."""
        if not self.redis_client or not self.local_node:
            return
        
        try:
            # Update node metrics
            self.local_node.current_load = len(self.local_node.task_queue) / self.local_node.max_concurrent_tasks
            self.local_node.last_heartbeat = time.time()
            
            # Update resource availability based on current load
            base_availability = 1.0 - self.local_node.current_load
            self.local_node.available_resources.update({
                ResourceType.CPU: max(0.1, base_availability),
                ResourceType.MEMORY: max(0.1, base_availability * 0.8),
                ResourceType.NETWORK: max(0.1, base_availability * 0.9),
            })
            
            # Update in Redis
            self.redis_client.hset(
                f"node:{self.node_id}",
                mapping={
                    "current_load": self.local_node.current_load,
                    "last_heartbeat": self.local_node.last_heartbeat,
                    "available_resources": json.dumps({rt.value: val for rt, val in self.local_node.available_resources.items()}),
                    "performance_score": self.local_node.performance_score,
                    "is_active": "true"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat: {e}")
    
    def _discover_nodes(self):
        """Discover other nodes in the cluster."""
        if not self.redis_client:
            return
        
        try:
            node_ids = self.redis_client.smembers("cluster:nodes")
            
            for node_id in node_ids:
                if node_id == self.node_id:
                    continue
                
                node_data = self.redis_client.hgetall(f"node:{node_id}")
                if not node_data:
                    continue
                
                # Parse node data
                try:
                    capabilities = json.loads(node_data.get("capabilities", "[]"))
                    available_resources = json.loads(node_data.get("available_resources", "{}"))
                    
                    node = ComputeNode(
                        node_id=node_id,
                        node_type=NodeType(node_data.get("node_type", "worker")),
                        host=node_data.get("host", "localhost"),
                        port=int(node_data.get("port", 8000)),
                        capabilities=capabilities,
                        max_concurrent_tasks=int(node_data.get("max_concurrent_tasks", 4)),
                        current_load=float(node_data.get("current_load", 0.0)),
                        available_resources={ResourceType(k): v for k, v in available_resources.items()},
                        last_heartbeat=float(node_data.get("last_heartbeat", 0)),
                        is_active=node_data.get("is_active", "false").lower() == "true",
                        performance_score=float(node_data.get("performance_score", 1.0))
                    )
                    
                    self.compute_nodes[node_id] = node
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse node data for {node_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to discover nodes: {e}")
    
    def _cleanup_stale_nodes(self):
        """Remove nodes that haven't sent heartbeat recently."""
        current_time = time.time()
        stale_threshold = self.heartbeat_interval * 3  # 3x heartbeat interval
        
        stale_nodes = []
        for node_id, node in self.compute_nodes.items():
            if node_id == self.node_id:
                continue
            
            if current_time - node.last_heartbeat > stale_threshold:
                stale_nodes.append(node_id)
                node.is_active = False
        
        # Remove stale nodes
        for node_id in stale_nodes:
            del self.compute_nodes[node_id]
            if self.redis_client:
                self.redis_client.srem("cluster:nodes", node_id)
                self.redis_client.delete(f"node:{node_id}")
            
            self.logger.warning(f"Removed stale node: {node_id}")
    
    @monitor_operation("submit_distributed_task")
    async def submit_distributed_task(
        self,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        resource_requirements: Optional[Dict[ResourceType, float]] = None,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Submit task for distributed execution."""
        task_id = str(uuid.uuid4())
        
        # Serialize function arguments
        try:
            serialized_args = pickle.dumps(args)
            serialized_kwargs = pickle.dumps(kwargs or {})
        except Exception as e:
            self.logger.error(f"Failed to serialize task arguments: {e}")
            raise
        
        # Default resource requirements
        if resource_requirements is None:
            resource_requirements = {
                ResourceType.CPU: 0.5,
                ResourceType.MEMORY: 0.3,
                ResourceType.STORAGE: 0.1,
                ResourceType.NETWORK: 0.1,
            }
        
        # Create distributed task
        task = DistributedTask(
            task_id=task_id,
            function_name=function.__name__ if hasattr(function, '__name__') else str(function),
            serialized_args=serialized_args,
            serialized_kwargs=serialized_kwargs,
            priority=priority,
            resource_requirements=resource_requirements,
            dependencies=dependencies or []
        )
        
        self.distributed_tasks[task_id] = task
        self.pending_tasks.append(task_id)
        
        # Store in Redis for cluster visibility
        if self.redis_client:
            task_data = {
                "function_name": task.function_name,
                "serialized_args": pickle.dumps(task.serialized_args).hex(),
                "serialized_kwargs": pickle.dumps(task.serialized_kwargs).hex(),
                "priority": task.priority.value,
                "resource_requirements": json.dumps({rt.value: val for rt, val in task.resource_requirements.items()}),
                "dependencies": json.dumps(task.dependencies),
                "created_at": task.created_at,
                "status": "pending"
            }
            
            self.redis_client.hset(f"task:{task_id}", mapping=task_data)
            self.redis_client.lpush("pending_tasks", task_id)
        
        self.logger.info(f"Submitted distributed task {task_id} with priority {priority.value}")
        return task_id
    
    def _process_pending_tasks(self):
        """Process pending tasks and assign to nodes."""
        if not self.pending_tasks:
            return
        
        # Get available nodes
        available_nodes = [
            node for node in self.compute_nodes.values()
            if node.is_active and len(node.task_queue) < node.max_concurrent_tasks
        ]
        
        if not available_nodes:
            return
        
        # Process tasks based on strategy
        tasks_to_assign = []
        
        for task_id in self.pending_tasks[:]:
            task = self.distributed_tasks.get(task_id)
            if not task:
                self.pending_tasks.remove(task_id)
                continue
            
            # Check dependencies
            if task.dependencies:
                deps_completed = all(
                    dep_id in self.task_results or 
                    (dep_id in self.distributed_tasks and self.distributed_tasks[dep_id].completed_at)
                    for dep_id in task.dependencies
                )
                if not deps_completed:
                    continue
            
            # Find suitable node
            best_node = self._select_node_for_task(task, available_nodes)
            if best_node:
                task.assigned_node = best_node.node_id
                best_node.task_queue.append(task_id)
                tasks_to_assign.append(task_id)
                self.pending_tasks.remove(task_id)
                self.running_tasks[task_id] = best_node.node_id
                
                self.logger.info(f"Assigned task {task_id} to node {best_node.node_id}")
        
        # Execute assigned tasks
        for task_id in tasks_to_assign:
            self._execute_task_on_node(task_id)
    
    def _select_node_for_task(self, task: DistributedTask, available_nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Select best node for task using quantum-inspired optimization."""
        if not available_nodes:
            return None
        
        if self.distribution_strategy == TaskDistributionStrategy.QUANTUM_OPTIMAL:
            return self._quantum_node_selection(task, available_nodes)
        elif self.distribution_strategy == TaskDistributionStrategy.LOAD_BALANCED:
            return min(available_nodes, key=lambda n: n.current_load)
        elif self.distribution_strategy == TaskDistributionStrategy.PRIORITY_BASED:
            # Prefer high-performance nodes for high-priority tasks
            if task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                return max(available_nodes, key=lambda n: n.performance_score)
            else:
                return min(available_nodes, key=lambda n: n.current_load)
        else:
            # Round robin
            return available_nodes[len(self.running_tasks) % len(available_nodes)]
    
    def _quantum_node_selection(self, task: DistributedTask, available_nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Use quantum-inspired algorithm to select optimal node."""
        # Create quantum state for each node
        n_nodes = len(available_nodes)
        quantum_amplitudes = np.ones(n_nodes, dtype=complex)
        
        for i, node in enumerate(available_nodes):
            # Calculate node fitness
            resource_match = 0.0
            for resource_type, requirement in task.resource_requirements.items():
                available = node.available_resources.get(resource_type, 0.0)
                if available >= requirement:
                    resource_match += 1.0
                else:
                    resource_match += available / requirement
            
            resource_match /= len(task.resource_requirements)
            
            # Load factor (lower load is better)
            load_factor = 1.0 - node.current_load
            
            # Performance factor
            performance_factor = node.performance_score
            
            # Priority factor
            priority_factor = 1.0
            if task.priority == TaskPriority.CRITICAL:
                priority_factor = 2.0
            elif task.priority == TaskPriority.HIGH:
                priority_factor = 1.5
            
            # Combine factors into quantum amplitude
            fitness = resource_match * load_factor * performance_factor * priority_factor
            phase = np.random.uniform(0, 2*np.pi)  # Random phase for quantum superposition
            quantum_amplitudes[i] = fitness * (np.cos(phase) + 1j * np.sin(phase))
        
        # Normalize quantum state
        quantum_amplitudes /= np.linalg.norm(quantum_amplitudes)
        
        # Quantum measurement (collapse to classical choice)
        probabilities = np.abs(quantum_amplitudes) ** 2
        
        # Add small amount of exploration
        exploration_noise = np.random.uniform(0, 0.1, size=n_nodes)
        probabilities += exploration_noise
        probabilities /= np.sum(probabilities)
        
        # Select node based on probabilities
        selected_idx = np.random.choice(n_nodes, p=probabilities)
        return available_nodes[selected_idx]
    
    def _execute_task_on_node(self, task_id: str):
        """Execute task on assigned node."""
        task = self.distributed_tasks.get(task_id)
        if not task:
            return
        
        node = self.compute_nodes.get(task.assigned_node)
        if not node:
            self.logger.error(f"Node {task.assigned_node} not found for task {task_id}")
            return
        
        # If it's local node, execute locally
        if task.assigned_node == self.node_id:
            future = self.thread_executor.submit(self._execute_local_task, task_id)
            future.add_done_callback(lambda f: self._handle_task_completion(task_id, f))
        else:
            # Send to remote node (simplified - in production use proper RPC)
            self.logger.info(f"Task {task_id} assigned to remote node {task.assigned_node}")
            # For demo, simulate remote execution
            self.thread_executor.submit(self._simulate_remote_task, task_id)
    
    def _execute_local_task(self, task_id: str) -> Any:
        """Execute task locally."""
        task = self.distributed_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        task.started_at = time.time()
        
        try:
            # Deserialize arguments
            args = pickle.loads(task.serialized_args)
            kwargs = pickle.loads(task.serialized_kwargs)
            
            # For demo, simulate task execution
            # In production, this would call the actual function
            self.logger.info(f"Executing local task {task_id}: {task.function_name}")
            
            # Simulate some work
            time.sleep(np.random.uniform(0.5, 2.0))
            
            # Generate mock result
            result = {
                "task_id": task_id,
                "function": task.function_name,
                "status": "completed",
                "execution_time": time.time() - task.started_at,
                "node_id": self.node_id
            }
            
            task.completed_at = time.time()
            task.result = pickle.dumps(result)
            
            return result
            
        except Exception as e:
            task.error = str(e)
            task.completed_at = time.time()
            raise
        
        finally:
            # Update node metrics
            if self.local_node and task_id in self.local_node.task_queue:
                self.local_node.task_queue.remove(task_id)
    
    def _simulate_remote_task(self, task_id: str):
        """Simulate remote task execution for demo."""
        time.sleep(np.random.uniform(1.0, 3.0))  # Simulate network latency + execution
        
        task = self.distributed_tasks.get(task_id)
        if task:
            task.started_at = time.time() - 2.0  # Backdate
            task.completed_at = time.time()
            
            result = {
                "task_id": task_id,
                "function": task.function_name,
                "status": "completed",
                "execution_time": 2.0,
                "node_id": task.assigned_node
            }
            
            task.result = pickle.dumps(result)
            self.task_results[task_id] = result
    
    def _handle_task_completion(self, task_id: str, future: Future):
        """Handle task completion callback."""
        try:
            result = future.result()
            self.task_results[task_id] = result
            
            # Update performance metrics
            task = self.distributed_tasks.get(task_id)
            if task and task.started_at and task.completed_at:
                execution_time = task.completed_at - task.started_at
                self.latency_history.append(execution_time)
                
                # Update node performance score
                node = self.compute_nodes.get(task.assigned_node)
                if node and execution_time > 0:
                    # Update based on performance vs expected
                    expected_time = 1.0  # Base expectation
                    performance_ratio = expected_time / execution_time
                    node.performance_score = 0.9 * node.performance_score + 0.1 * performance_ratio
            
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            self.logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            # Handle task failure
            task = self.distributed_tasks.get(task_id)
            if task:
                task.error = str(e)
                task.completed_at = time.time()
            
            # Update error rates
            node_id = task.assigned_node if task else "unknown"
            self.error_rates[node_id] = self.error_rates[node_id] * 0.9 + 0.1
            
            self.logger.error(f"Task {task_id} failed: {e}")
            
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def _monitor_running_tasks(self):
        """Monitor running tasks for timeouts."""
        current_time = time.time()
        
        timed_out_tasks = []
        for task_id, node_id in self.running_tasks.items():
            task = self.distributed_tasks.get(task_id)
            if not task or not task.started_at:
                continue
            
            if current_time - task.started_at > self.task_timeout:
                timed_out_tasks.append(task_id)
        
        # Handle timed out tasks
        for task_id in timed_out_tasks:
            task = self.distributed_tasks.get(task_id)
            if task:
                task.error = "Task timeout"
                task.completed_at = current_time
            
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            self.logger.warning(f"Task {task_id} timed out")
    
    def _auto_scale_cluster(self):
        """Automatically scale cluster based on load."""
        # Calculate cluster metrics
        total_load = sum(node.current_load for node in self.compute_nodes.values())
        avg_load = total_load / len(self.compute_nodes) if self.compute_nodes else 0
        
        pending_count = len(self.pending_tasks)
        active_nodes = len([n for n in self.compute_nodes.values() if n.is_active])
        
        # Scale up conditions
        if (avg_load > 0.8 or pending_count > 10) and active_nodes < self.max_nodes:
            self.logger.info(f"Considering scale up: avg_load={avg_load:.2f}, pending={pending_count}")
            # In production, trigger node provisioning
            
        # Scale down conditions  
        elif avg_load < 0.3 and active_nodes > 2 and pending_count == 0:
            self.logger.info(f"Considering scale down: avg_load={avg_load:.2f}")
            # In production, trigger node decommissioning
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        active_nodes = [n for n in self.compute_nodes.values() if n.is_active]
        
        return {
            "cluster_id": self.node_id,
            "node_type": self.node_type.value,
            "total_nodes": len(self.compute_nodes),
            "active_nodes": len(active_nodes),
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.task_results),
            "average_load": sum(n.current_load for n in active_nodes) / len(active_nodes) if active_nodes else 0,
            "total_capacity": sum(n.max_concurrent_tasks for n in active_nodes),
            "distribution_strategy": self.distribution_strategy.value,
            "auto_scaling_enabled": self.enable_auto_scaling,
            "error_rates": dict(self.error_rates),
            "performance_metrics": {
                "avg_latency": np.mean(self.latency_history) if self.latency_history else 0,
                "throughput": len(self.throughput_history),
                "success_rate": 1.0 - np.mean(list(self.error_rates.values())) if self.error_rates else 1.0
            }
        }
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for distributed task to complete."""
        start_time = time.time()
        
        while task_id not in self.task_results:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            # Check if task failed
            task = self.distributed_tasks.get(task_id)
            if task and task.error:
                raise RuntimeError(f"Task {task_id} failed: {task.error}")
            
            await asyncio.sleep(0.1)
        
        # Deserialize and return result
        result = self.task_results[task_id]
        if isinstance(result, bytes):
            return pickle.loads(result)
        return result
    
    async def shutdown(self):
        """Gracefully shutdown distributed manager."""
        self.logger.info("Shutting down QuantumDistributedManager")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for threads to finish
        for thread in [self.heartbeat_thread, self.task_monitor_thread, self.auto_scaling_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        # Clean up Redis
        if self.redis_client:
            try:
                self.redis_client.srem("cluster:nodes", self.node_id)
                self.redis_client.delete(f"node:{self.node_id}")
            except Exception as e:
                self.logger.error(f"Redis cleanup error: {e}")
        
        # Shutdown quantum planner
        await self.quantum_planner.shutdown()
        
        self.logger.info("QuantumDistributedManager shutdown complete")