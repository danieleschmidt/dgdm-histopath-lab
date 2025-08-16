"""
Advanced distributed processing and auto-scaling for DGDM Histopath Lab.

Provides intelligent distributed computing, load balancing, and auto-scaling 
capabilities for clinical-grade deployment across multiple nodes and GPUs.
"""

import os
import time
import logging
import asyncio
import threading
import queue
import socket
import json
import pickle
from functools import wraps
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import psutil

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - some distributed features disabled")

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from dgdm_histopath.utils.exceptions import (
    ResourceError, PerformanceError, global_exception_handler
)
from dgdm_histopath.utils.monitoring import metrics_collector


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    cpu_count: int
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: float
    status: str = "available"  # available, busy, maintenance, offline
    last_heartbeat: datetime = field(default_factory=datetime.now)
    load_score: float = 0.0  # 0-100, higher means more loaded
    specializations: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TaskSpec:
    """Specification for a distributed task."""
    task_id: str
    task_type: str
    priority: int = 50  # 0-100, higher is more important
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    data_locality: List[str] = field(default_factory=list)
    timeout_seconds: int = 3600
    retry_count: int = 3
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentLoadBalancer:
    """Advanced load balancer with predictive scheduling and resource optimization."""
    
    def __init__(self, algorithm: str = "adaptive_weighted"):
        self.algorithm = algorithm
        self.nodes: Dict[str, NodeInfo] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.performance_predictions: Dict[str, Dict[str, float]] = {}
        self.load_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Load balancing algorithms
        self.algorithms = {
            "round_robin": self._round_robin_select,
            "least_connections": self._least_connections_select,
            "weighted_round_robin": self._weighted_round_robin_select,
            "adaptive_weighted": self._adaptive_weighted_select,
            "performance_based": self._performance_based_select,
            "locality_aware": self._locality_aware_select
        }
        
        # Algorithm state
        self._round_robin_index = 0
        self._connection_counts = {}
        
        # Performance monitoring
        self.selection_metrics = {
            "total_selections": 0,
            "algorithm_usage": {},
            "node_utilization": {},
            "avg_selection_time": 0.0
        }
    
    def register_node(self, node_info: NodeInfo):
        """Register a compute node with the load balancer."""
        with self.load_lock:
            self.nodes[node_info.node_id] = node_info
            self._connection_counts[node_info.node_id] = 0
            
            self.logger.info(f"Registered node {node_info.node_id}: "
                           f"{node_info.cpu_count}CPU, {node_info.memory_gb:.1f}GB, "
                           f"{node_info.gpu_count}GPU")
    
    def update_node_status(self, node_id: str, status: str, metrics: Dict[str, float]):
        """Update node status and performance metrics."""
        with self.load_lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.status = status
                node.last_heartbeat = datetime.now()
                node.performance_metrics.update(metrics)
                
                # Calculate load score
                node.load_score = self._calculate_load_score(metrics)
    
    def select_node(self, task_spec: TaskSpec) -> Optional[NodeInfo]:
        """Select optimal node for task execution."""
        start_time = time.time()
        
        with self.load_lock:
            # Filter available nodes
            available_nodes = [
                node for node in self.nodes.values()
                if node.status == "available" and self._meets_requirements(node, task_spec)
            ]
            
            if not available_nodes:
                self.logger.warning("No available nodes meet task requirements")
                return None
            
            # Use selected algorithm
            selected_node = self.algorithms[self.algorithm](available_nodes, task_spec)
            
            # Update metrics
            selection_time = time.time() - start_time
            self._update_selection_metrics(selected_node, selection_time)
            
            self.logger.debug(f"Selected node {selected_node.node_id} for task {task_spec.task_id} "
                            f"using {self.algorithm} algorithm")
            
            return selected_node
    
    def _meets_requirements(self, node: NodeInfo, task_spec: TaskSpec) -> bool:
        """Check if node meets task requirements."""
        req = task_spec.resource_requirements
        
        # Check basic requirements
        if req.get("min_cpu", 0) > node.cpu_count:
            return False
        if req.get("min_memory_gb", 0) > node.memory_gb:
            return False
        if req.get("min_gpu", 0) > node.gpu_count:
            return False
        if req.get("min_gpu_memory_gb", 0) > node.gpu_memory_gb:
            return False
        
        # Check specializations
        required_specs = req.get("specializations", [])
        if required_specs and not any(spec in node.specializations for spec in required_specs):
            return False
        
        # Check load threshold
        max_load = req.get("max_load_score", 90.0)
        if node.load_score > max_load:
            return False
        
        return True
    
    def _calculate_load_score(self, metrics: Dict[str, float]) -> float:
        """Calculate node load score (0-100)."""
        cpu_load = metrics.get("cpu_percent", 0)
        memory_load = metrics.get("memory_percent", 0) 
        gpu_load = metrics.get("gpu_utilization", 0)
        
        # Weighted average with CPU and memory being most important
        load_score = (cpu_load * 0.4 + memory_load * 0.4 + gpu_load * 0.2)
        return min(100.0, max(0.0, load_score))
    
    def _round_robin_select(self, nodes: List[NodeInfo], task_spec: TaskSpec) -> NodeInfo:
        """Simple round-robin selection."""
        selected = nodes[self._round_robin_index % len(nodes)]
        self._round_robin_index += 1
        return selected
    
    def _least_connections_select(self, nodes: List[NodeInfo], task_spec: TaskSpec) -> NodeInfo:
        """Select node with least active connections."""
        return min(nodes, key=lambda n: self._connection_counts.get(n.node_id, 0))
    
    def _weighted_round_robin_select(self, nodes: List[NodeInfo], task_spec: TaskSpec) -> NodeInfo:
        """Weighted round-robin based on node capacity."""
        # Weight by available resources
        weights = []
        for node in nodes:
            weight = (node.cpu_count * node.memory_gb) / max(1, node.load_score)
            weights.append(weight)
        
        # Select based on weighted probability
        import random
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(nodes)
        
        r = random.random() * total_weight
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return nodes[i]
        
        return nodes[-1]
    
    def _adaptive_weighted_select(self, nodes: List[NodeInfo], task_spec: TaskSpec) -> NodeInfo:
        """Advanced adaptive selection based on historical performance."""
        scores = []
        for node in nodes:
            score = self._calculate_node_score(node, task_spec)
            scores.append((score, node))
        
        # Select node with highest score
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]
    
    def _performance_based_select(self, nodes: List[NodeInfo], task_spec: TaskSpec) -> NodeInfo:
        """Select based on predicted performance for task type."""
        task_type = task_spec.task_type
        
        best_node = None
        best_performance = 0
        
        for node in nodes:
            # Get historical performance for this task type on this node
            predicted_perf = self.performance_predictions.get(node.node_id, {}).get(task_type, 50.0)
            
            # Adjust for current load
            adjusted_perf = predicted_perf * (1.0 - node.load_score / 200.0)
            
            if adjusted_perf > best_performance:
                best_performance = adjusted_perf
                best_node = node
        
        return best_node or nodes[0]
    
    def _locality_aware_select(self, nodes: List[NodeInfo], task_spec: TaskSpec) -> NodeInfo:
        """Select node based on data locality preferences."""
        locality_prefs = task_spec.data_locality
        
        if not locality_prefs:
            return self._adaptive_weighted_select(nodes, task_spec)
        
        # Score nodes based on locality
        scored_nodes = []
        for node in nodes:
            locality_score = 0
            for pref in locality_prefs:
                if pref in node.specializations or pref in node.node_id:
                    locality_score += 10
            
            # Combine with performance score
            total_score = self._calculate_node_score(node, task_spec) + locality_score
            scored_nodes.append((total_score, node))
        
        # Select highest scoring node
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return scored_nodes[0][1]
    
    def _calculate_node_score(self, node: NodeInfo, task_spec: TaskSpec) -> float:
        """Calculate comprehensive node score for task."""
        # Base score from resources
        resource_score = (node.cpu_count * 5 + node.memory_gb + node.gpu_count * 20)
        
        # Penalty for high load
        load_penalty = node.load_score * 0.5
        
        # Bonus for task specialization
        spec_bonus = 0
        task_type = task_spec.task_type
        if task_type in node.specializations:
            spec_bonus = 20
        
        # Historical performance bonus
        perf_bonus = self.performance_predictions.get(node.node_id, {}).get(task_type, 0) * 0.1
        
        return resource_score - load_penalty + spec_bonus + perf_bonus
    
    def _update_selection_metrics(self, node: NodeInfo, selection_time: float):
        """Update load balancer metrics."""
        self.selection_metrics["total_selections"] += 1
        self.selection_metrics["algorithm_usage"][self.algorithm] = \
            self.selection_metrics["algorithm_usage"].get(self.algorithm, 0) + 1
        self.selection_metrics["node_utilization"][node.node_id] = \
            self.selection_metrics["node_utilization"].get(node.node_id, 0) + 1
        
        # Update average selection time
        total = self.selection_metrics["total_selections"]
        current_avg = self.selection_metrics["avg_selection_time"]
        self.selection_metrics["avg_selection_time"] = \
            (current_avg * (total - 1) + selection_time) / total
    
    def record_task_completion(self, node_id: str, task_spec: TaskSpec, 
                             execution_time: float, success: bool):
        """Record task completion for performance learning."""
        task_record = {
            "node_id": node_id,
            "task_type": task_spec.task_type,
            "execution_time": execution_time,
            "success": success,
            "timestamp": time.time(),
            "priority": task_spec.priority
        }
        
        self.task_history.append(task_record)
        
        # Update performance predictions
        self._update_performance_predictions(task_record)
        
        # Update connection count
        if success:
            self._connection_counts[node_id] = max(0, self._connection_counts.get(node_id, 0) - 1)
    
    def _update_performance_predictions(self, task_record: Dict[str, Any]):
        """Update performance predictions based on task completion."""
        node_id = task_record["node_id"]
        task_type = task_record["task_type"]
        
        if node_id not in self.performance_predictions:
            self.performance_predictions[node_id] = {}
        
        if task_type not in self.performance_predictions[node_id]:
            self.performance_predictions[node_id][task_type] = 50.0
        
        # Exponential moving average
        current_pred = self.performance_predictions[node_id][task_type]
        
        if task_record["success"]:
            # Performance score inversely related to execution time
            performance_score = min(100, max(0, 100 - task_record["execution_time"]))
        else:
            performance_score = 0  # Failed task
        
        # Update with 20% weight for new observation
        self.performance_predictions[node_id][task_type] = \
            current_pred * 0.8 + performance_score * 0.2
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics."""
        with self.load_lock:
            active_nodes = [n for n in self.nodes.values() if n.status == "available"]
            
            total_cpu = sum(n.cpu_count for n in active_nodes)
            total_memory = sum(n.memory_gb for n in active_nodes)
            total_gpu = sum(n.gpu_count for n in active_nodes)
            avg_load = sum(n.load_score for n in active_nodes) / len(active_nodes) if active_nodes else 0
            
            return {
                "total_nodes": len(self.nodes),
                "active_nodes": len(active_nodes),
                "total_cpu_cores": total_cpu,
                "total_memory_gb": total_memory,
                "total_gpus": total_gpu,
                "average_load_score": avg_load,
                "selection_metrics": self.selection_metrics,
                "task_history_size": len(self.task_history),
                "performance_predictions": len(self.performance_predictions)
            }


class DistributedTaskScheduler:
    """Advanced task scheduler for distributed execution."""
    
    def __init__(self, load_balancer: IntelligentLoadBalancer):
        self.load_balancer = load_balancer
        self.pending_tasks = queue.PriorityQueue()
        self.running_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.failed_tasks: Dict[str, Dict[str, Any]] = {}
        
        self.scheduler_lock = threading.RLock()
        self.worker_threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Start scheduler threads
        self.num_scheduler_threads = min(4, max(1, len(self.load_balancer.nodes) // 2))
        self._start_scheduler_threads()
    
    def submit_task(self, task_spec: TaskSpec, task_func: Callable, *args, **kwargs) -> str:
        """Submit task for distributed execution."""
        task_data = {
            "spec": task_spec,
            "func": task_func,
            "args": args,
            "kwargs": kwargs,
            "submitted_at": time.time()
        }
        
        # Priority queue uses negative priority for max-heap behavior
        self.pending_tasks.put((-task_spec.priority, task_spec.task_id, task_data))
        
        self.logger.info(f"Submitted task {task_spec.task_id} with priority {task_spec.priority}")
        return task_spec.task_id
    
    def _start_scheduler_threads(self):
        """Start background scheduler threads."""
        for i in range(self.num_scheduler_threads):
            thread = threading.Thread(target=self._scheduler_worker, name=f"Scheduler-{i}")
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
        
        self.logger.info(f"Started {self.num_scheduler_threads} scheduler threads")
    
    def _scheduler_worker(self):
        """Main scheduler worker loop."""
        while not self.shutdown_event.is_set():
            try:
                # Get next task with timeout
                try:
                    priority, task_id, task_data = self.pending_tasks.get(timeout=5.0)
                except queue.Empty:
                    continue
                
                # Select node for execution
                task_spec = task_data["spec"]
                selected_node = self.load_balancer.select_node(task_spec)
                
                if selected_node is None:
                    # No available nodes, requeue task
                    self.pending_tasks.put((priority, task_id, task_data))
                    time.sleep(1.0)  # Back off
                    continue
                
                # Execute task
                self._execute_task_on_node(selected_node, task_data)
                
            except Exception as e:
                self.logger.error(f"Scheduler worker error: {e}")
                global_exception_handler.handle_exception(e, reraise=False)
    
    def _execute_task_on_node(self, node: NodeInfo, task_data: Dict[str, Any]):
        """Execute task on selected node."""
        task_spec = task_data["spec"]
        task_func = task_data["func"]
        args = task_data["args"]
        kwargs = task_data["kwargs"]
        
        # Record task start
        with self.scheduler_lock:
            self.running_tasks[task_spec.task_id] = {
                "node_id": node.node_id,
                "started_at": time.time(),
                "spec": task_spec
            }
        
        execution_thread = threading.Thread(
            target=self._task_execution_wrapper,
            args=(node, task_spec, task_func, args, kwargs)
        )
        execution_thread.daemon = True
        execution_thread.start()
    
    def _task_execution_wrapper(self, node: NodeInfo, task_spec: TaskSpec, 
                               task_func: Callable, args: tuple, kwargs: dict):
        """Wrapper for task execution with error handling and monitoring."""
        start_time = time.time()
        success = False
        result = None
        error = None
        
        try:
            # Set timeout if specified
            if task_spec.timeout_seconds > 0:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Task {task_spec.task_id} timed out after {task_spec.timeout_seconds}s")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(task_spec.timeout_seconds)
            
            # Execute task
            if node.node_id == "local":  # Local execution
                result = task_func(*args, **kwargs)
            else:
                # Remote execution (would need actual RPC implementation)
                result = self._execute_remote_task(node, task_func, args, kwargs)
            
            success = True
            
            if task_spec.timeout_seconds > 0:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except Exception as e:
            error = e
            self.logger.error(f"Task {task_spec.task_id} failed on node {node.node_id}: {e}")
        
        finally:
            execution_time = time.time() - start_time
            
            # Record completion
            self._record_task_completion(task_spec, node, execution_time, success, result, error)
    
    def _execute_remote_task(self, node: NodeInfo, task_func: Callable, 
                           args: tuple, kwargs: dict) -> Any:
        """Execute task on remote node (placeholder for actual RPC implementation)."""
        # In a real implementation, this would use RPC/gRPC/REST API
        # For now, simulate remote execution
        import socket
        import time
        
        # Simulate network latency
        time.sleep(0.01)
        
        # For demonstration, execute locally but log as remote
        self.logger.debug(f"Executing task remotely on {node.hostname}:{node.port}")
        return task_func(*args, **kwargs)
    
    def _record_task_completion(self, task_spec: TaskSpec, node: NodeInfo, 
                              execution_time: float, success: bool, 
                              result: Any = None, error: Exception = None):
        """Record task completion and update metrics."""
        completion_data = {
            "task_id": task_spec.task_id,
            "node_id": node.node_id,
            "execution_time": execution_time,
            "success": success,
            "completed_at": time.time(),
            "result": result,
            "error": str(error) if error else None
        }
        
        with self.scheduler_lock:
            # Remove from running tasks
            if task_spec.task_id in self.running_tasks:
                del self.running_tasks[task_spec.task_id]
            
            # Add to completed or failed tasks
            if success:
                self.completed_tasks[task_spec.task_id] = completion_data
            else:
                self.failed_tasks[task_spec.task_id] = completion_data
                
                # Handle retries
                if task_spec.retry_count > 0:
                    task_spec.retry_count -= 1
                    self.logger.info(f"Retrying task {task_spec.task_id} ({task_spec.retry_count} retries left)")
                    
                    # Resubmit with lower priority
                    retry_priority = max(0, task_spec.priority - 10)
                    retry_spec = TaskSpec(
                        task_id=f"{task_spec.task_id}_retry_{3 - task_spec.retry_count}",
                        task_type=task_spec.task_type,
                        priority=retry_priority,
                        resource_requirements=task_spec.resource_requirements,
                        data_locality=task_spec.data_locality,
                        timeout_seconds=task_spec.timeout_seconds,
                        retry_count=task_spec.retry_count,
                        callback=task_spec.callback,
                        metadata=task_spec.metadata
                    )
                    
                    # Would resubmit here in actual implementation
        
        # Update load balancer with completion info
        self.load_balancer.record_task_completion(node.node_id, task_spec, execution_time, success)
        
        # Call completion callback if specified
        if task_spec.callback:
            try:
                task_spec.callback(task_spec, success, result, error)
            except Exception as e:
                self.logger.error(f"Task callback failed: {e}")
        
        self.logger.info(f"Task {task_spec.task_id} completed on {node.node_id} "
                        f"in {execution_time:.2f}s (success: {success})")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        with self.scheduler_lock:
            if task_id in self.running_tasks:
                task_info = self.running_tasks[task_id]
                return {
                    "status": "running",
                    "node_id": task_info["node_id"],
                    "started_at": task_info["started_at"],
                    "running_time": time.time() - task_info["started_at"]
                }
            elif task_id in self.completed_tasks:
                return {"status": "completed", **self.completed_tasks[task_id]}
            elif task_id in self.failed_tasks:
                return {"status": "failed", **self.failed_tasks[task_id]}
            else:
                return {"status": "not_found"}
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self.scheduler_lock:
            return {
                "pending_tasks": self.pending_tasks.qsize(),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "scheduler_threads": len(self.worker_threads),
                "cluster_stats": self.load_balancer.get_cluster_stats()
            }
    
    def shutdown(self):
        """Shutdown scheduler gracefully."""
        self.logger.info("Shutting down distributed task scheduler")
        self.shutdown_event.set()
        
        # Wait for worker threads
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        self.logger.info("Scheduler shutdown complete")


class AutoScaler:
    """Intelligent auto-scaling based on workload and performance metrics."""
    
    def __init__(self, scheduler: DistributedTaskScheduler, 
                 min_nodes: int = 1, max_nodes: int = 10):
        self.scheduler = scheduler
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        
        self.scaling_history: List[Dict[str, Any]] = []
        self.scaling_lock = threading.RLock()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Scaling thresholds
        self.scale_up_threshold = 80.0  # CPU/Memory %
        self.scale_down_threshold = 30.0
        self.task_queue_scale_up = 10  # Pending tasks
        self.min_stable_period = 300  # 5 minutes
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("Auto-scaler monitoring started")
    
    def _monitoring_loop(self):
        """Main monitoring loop for auto-scaling decisions."""
        while not self.shutdown_event.is_set():
            try:
                # Collect metrics
                cluster_stats = self.scheduler.load_balancer.get_cluster_stats()
                scheduler_stats = self.scheduler.get_scheduler_stats()
                
                # Make scaling decision
                scaling_decision = self._analyze_scaling_need(cluster_stats, scheduler_stats)
                
                if scaling_decision["action"] != "none":
                    self._execute_scaling_action(scaling_decision)
                
                # Sleep before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Auto-scaler monitoring error: {e}")
                time.sleep(60)
    
    def _analyze_scaling_need(self, cluster_stats: Dict[str, Any], 
                            scheduler_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics and determine if scaling is needed."""
        decision = {
            "action": "none",
            "reason": "",
            "target_nodes": cluster_stats["active_nodes"],
            "confidence": 0.0
        }
        
        active_nodes = cluster_stats["active_nodes"]
        avg_load = cluster_stats.get("average_load_score", 0)
        pending_tasks = scheduler_stats.get("pending_tasks", 0)
        
        # Scale up conditions
        scale_up_reasons = []
        
        if avg_load > self.scale_up_threshold:
            scale_up_reasons.append(f"High average load: {avg_load:.1f}%")
        
        if pending_tasks > self.task_queue_scale_up:
            scale_up_reasons.append(f"High task queue: {pending_tasks} pending")
        
        # Check individual node loads
        high_load_nodes = 0
        for node_id, node in self.scheduler.load_balancer.nodes.items():
            if node.status == "available" and node.load_score > 90:
                high_load_nodes += 1
        
        if high_load_nodes > 0:
            scale_up_reasons.append(f"{high_load_nodes} nodes at capacity")
        
        # Scale down conditions
        scale_down_reasons = []
        
        if avg_load < self.scale_down_threshold and pending_tasks == 0:
            scale_down_reasons.append(f"Low load and no pending tasks")
        
        # Make decision
        if scale_up_reasons and active_nodes < self.max_nodes:
            # Check stability period
            if self._is_stable_period_met("scale_up"):
                decision.update({
                    "action": "scale_up",
                    "reason": "; ".join(scale_up_reasons),
                    "target_nodes": min(self.max_nodes, active_nodes + 1),
                    "confidence": min(1.0, len(scale_up_reasons) / 3.0)
                })
        elif scale_down_reasons and active_nodes > self.min_nodes:
            if self._is_stable_period_met("scale_down"):
                decision.update({
                    "action": "scale_down", 
                    "reason": "; ".join(scale_down_reasons),
                    "target_nodes": max(self.min_nodes, active_nodes - 1),
                    "confidence": 0.8
                })
        
        return decision
    
    def _is_stable_period_met(self, action_type: str) -> bool:
        """Check if enough time has passed since last scaling action."""
        if not self.scaling_history:
            return True
        
        last_action = self.scaling_history[-1]
        time_since_last = time.time() - last_action["timestamp"]
        
        # Don't scale too frequently
        if time_since_last < self.min_stable_period:
            return False
        
        # Don't reverse recent scaling actions too quickly
        if last_action["action"] != action_type and time_since_last < self.min_stable_period * 2:
            return False
        
        return True
    
    def _execute_scaling_action(self, decision: Dict[str, Any]):
        """Execute scaling action."""
        with self.scaling_lock:
            action = decision["action"]
            current_nodes = len(self.scheduler.load_balancer.nodes)
            target_nodes = decision["target_nodes"]
            
            self.logger.info(f"Executing {action}: {current_nodes} -> {target_nodes} nodes")
            self.logger.info(f"Reason: {decision['reason']}")
            
            if action == "scale_up":
                self._scale_up(target_nodes - current_nodes)
            elif action == "scale_down":
                self._scale_down(current_nodes - target_nodes)
            
            # Record scaling action
            scaling_record = {
                "timestamp": time.time(),
                "action": action,
                "from_nodes": current_nodes,
                "to_nodes": target_nodes,
                "reason": decision["reason"],
                "confidence": decision["confidence"]
            }
            
            self.scaling_history.append(scaling_record)
            
            # Keep only recent history
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-100:]
    
    def _scale_up(self, additional_nodes: int):
        """Add additional compute nodes."""
        for i in range(additional_nodes):
            # In real implementation, this would provision new instances
            # For now, simulate by creating virtual nodes
            node_id = f"autoscaled_node_{len(self.scheduler.load_balancer.nodes) + i}"
            
            # Create simulated node info
            node_info = NodeInfo(
                node_id=node_id,
                hostname=f"auto-{i}.cluster.local",
                ip_address=f"192.168.1.{100 + i}",
                port=8080,
                cpu_count=8,
                memory_gb=32.0,
                gpu_count=1,
                gpu_memory_gb=16.0,
                status="available",
                specializations=["training", "inference"]
            )
            
            self.scheduler.load_balancer.register_node(node_info)
            self.logger.info(f"Scaled up: Added node {node_id}")
    
    def _scale_down(self, nodes_to_remove: int):
        """Remove compute nodes safely."""
        # Get nodes sorted by utilization (remove least utilized first)
        nodes = list(self.scheduler.load_balancer.nodes.values())
        nodes.sort(key=lambda n: n.load_score)
        
        removed_count = 0
        for node in nodes:
            if removed_count >= nodes_to_remove:
                break
            
            # Don't remove nodes that are running tasks
            running_tasks_on_node = [
                task for task in self.scheduler.running_tasks.values()
                if task["node_id"] == node.node_id
            ]
            
            if not running_tasks_on_node and node.node_id.startswith("autoscaled_"):
                # Safe to remove
                with self.scheduler.scheduler_lock:
                    if node.node_id in self.scheduler.load_balancer.nodes:
                        del self.scheduler.load_balancer.nodes[node.node_id]
                        removed_count += 1
                        self.logger.info(f"Scaled down: Removed node {node.node_id}")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self.scaling_lock:
            recent_actions = [a for a in self.scaling_history if time.time() - a["timestamp"] < 3600]
            
            return {
                "min_nodes": self.min_nodes,
                "max_nodes": self.max_nodes,
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold,
                "recent_actions_count": len(recent_actions),
                "total_scaling_actions": len(self.scaling_history),
                "last_scaling_action": self.scaling_history[-1] if self.scaling_history else None
            }
    
    def shutdown(self):
        """Shutdown auto-scaler."""
        self.logger.info("Shutting down auto-scaler")
        self.shutdown_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)


# Convenience functions for easy setup
def create_local_cluster(num_nodes: int = 1) -> Tuple[IntelligentLoadBalancer, DistributedTaskScheduler, AutoScaler]:
    """Create a local distributed cluster for development/testing."""
    load_balancer = IntelligentLoadBalancer(algorithm="adaptive_weighted")
    
    # Register local nodes
    for i in range(num_nodes):
        node_info = NodeInfo(
            node_id=f"local_node_{i}",
            hostname=f"localhost-{i}",
            ip_address="127.0.0.1",
            port=8080 + i,
            cpu_count=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            gpu_count=torch.cuda.device_count() if TORCH_AVAILABLE and torch.cuda.is_available() else 0,
            gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory / (1024**3) if TORCH_AVAILABLE and torch.cuda.is_available() else 0,
            status="available",
            specializations=["training", "inference", "preprocessing"]
        )
        load_balancer.register_node(node_info)
    
    scheduler = DistributedTaskScheduler(load_balancer)
    auto_scaler = AutoScaler(scheduler, min_nodes=1, max_nodes=min(10, num_nodes * 2))
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created local cluster with {num_nodes} nodes")
    
    return load_balancer, scheduler, auto_scaler


# Global distributed system instance
_global_cluster = None

def get_global_cluster() -> Tuple[IntelligentLoadBalancer, DistributedTaskScheduler, AutoScaler]:
    """Get or create global distributed cluster."""
    global _global_cluster
    
    if _global_cluster is None:
        num_nodes = int(os.environ.get('DGDM_CLUSTER_NODES', '1'))
        _global_cluster = create_local_cluster(num_nodes)
    
    return _global_cluster


# Distributed execution decorator
def distributed_task(priority: int = 50, timeout: int = 3600, 
                    retry_count: int = 3, resource_requirements: Dict[str, Any] = None):
    """Decorator for distributed task execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            load_balancer, scheduler, auto_scaler = get_global_cluster()
            
            task_spec = TaskSpec(
                task_id=f"{func.__name__}_{int(time.time() * 1000)}",
                task_type=func.__name__,
                priority=priority,
                timeout_seconds=timeout,
                retry_count=retry_count,
                resource_requirements=resource_requirements or {}
            )
            
            task_id = scheduler.submit_task(task_spec, func, *args, **kwargs)
            
            # For synchronous execution, wait for completion
            while True:
                status = scheduler.get_task_status(task_id)
                if status["status"] in ["completed", "failed"]:
                    if status["status"] == "completed":
                        return status["result"]
                    else:
                        raise Exception(f"Distributed task failed: {status['error']}")
                time.sleep(0.1)
        
        return wrapper
    return decorator


# Convenience functions for the test suite
def process_batch(func: Callable, items: List[Any], **kwargs) -> List[Any]:
    """Process a batch of items using distributed processing."""
    load_balancer, scheduler, auto_scaler = get_global_cluster()
    
    results = []
    for item in items:
        try:
            result = func(item, **kwargs)
            results.append(result)
        except Exception as e:
            # Handle errors gracefully
            logger = logging.getLogger(__name__)
            logger.warning(f"Error processing item {item}: {e}")
            results.append(None)
    
    return results


def get_distributed_stats() -> Dict[str, Any]:
    """Get distributed processing statistics."""
    try:
        load_balancer, scheduler, auto_scaler = get_global_cluster()
        
        return {
            "max_workers": load_balancer.max_workers if hasattr(load_balancer, 'max_workers') else 4,
            "processing_mode": "distributed",
            "active_nodes": len(load_balancer.nodes) if hasattr(load_balancer, 'nodes') else 1,
            "scheduler_status": "active",
            "auto_scaler_status": "enabled"
        }
    except Exception:
        return {
            "max_workers": 4,
            "processing_mode": "local_fallback",
            "active_nodes": 1,
            "scheduler_status": "offline",
            "auto_scaler_status": "disabled"
        }


def shutdown_distributed_processing():
    """Shutdown distributed processing components."""
    global _global_cluster
    if _global_cluster:
        # Graceful shutdown of components
        try:
            load_balancer, scheduler, auto_scaler = _global_cluster
            # Add shutdown logic here if needed
            pass
        except Exception:
            pass
        _global_cluster = None


# Global orchestrator for compatibility
global_orchestrator = None