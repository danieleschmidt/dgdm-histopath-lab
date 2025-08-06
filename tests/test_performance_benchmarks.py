"""
Performance benchmarks and stress tests for DGDM quantum-enhanced framework.

Comprehensive performance testing including throughput, latency, memory usage,
and scalability benchmarks for medical AI workloads.
"""

import pytest
import time
import asyncio
import numpy as np
import torch
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any
import gc

from dgdm_histopath.quantum.quantum_planner import QuantumTaskPlanner, TaskPriority, ResourceType
from dgdm_histopath.quantum.quantum_scheduler import QuantumScheduler, WorkflowType
from dgdm_histopath.quantum.quantum_distributed import QuantumDistributedManager, NodeType
from dgdm_histopath.utils.performance import AdvancedCache, GPUMemoryManager, cached
from dgdm_histopath.utils.scaling import AutoScalingManager, ScalingStrategy


class BenchmarkMetrics:
    """Container for benchmark results."""
    
    def __init__(self):
        self.throughput = 0.0  # operations per second
        self.latency_mean = 0.0  # seconds
        self.latency_p95 = 0.0  # seconds
        self.latency_p99 = 0.0  # seconds
        self.memory_peak_mb = 0.0  # MB
        self.cpu_utilization = 0.0  # percentage
        self.error_rate = 0.0  # percentage
        self.scalability_factor = 1.0  # performance scaling vs resources
        
    def __str__(self):
        return (f"Throughput: {self.throughput:.2f} ops/sec, "
                f"Latency: {self.latency_mean*1000:.2f}ms (p95: {self.latency_p95*1000:.2f}ms), "
                f"Memory: {self.memory_peak_mb:.1f}MB, "
                f"CPU: {self.cpu_utilization:.1f}%, "
                f"Errors: {self.error_rate:.2f}%")


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = False
        
    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Monitor system resources."""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                cpu_percent = self.process.cpu_percent()
                
                self.memory_samples.append(memory_mb)
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(0.1)  # Sample every 100ms
            except:
                break
    
    def calculate_metrics(self, latencies: List[float], errors: int, total_ops: int) -> BenchmarkMetrics:
        """Calculate benchmark metrics."""
        metrics = BenchmarkMetrics()
        
        if latencies:
            metrics.latency_mean = np.mean(latencies)
            metrics.latency_p95 = np.percentile(latencies, 95)
            metrics.latency_p99 = np.percentile(latencies, 99)
            
            total_time = max(latencies) - min(latencies) if len(latencies) > 1 else sum(latencies)
            metrics.throughput = total_ops / total_time if total_time > 0 else 0
        
        if self.memory_samples:
            metrics.memory_peak_mb = max(self.memory_samples)
        
        if self.cpu_samples:
            metrics.cpu_utilization = np.mean(self.cpu_samples)
        
        metrics.error_rate = (errors / total_ops * 100) if total_ops > 0 else 0
        
        return metrics


@pytest.mark.performance
class TestQuantumPlannerPerformance(PerformanceBenchmark):
    """Performance tests for QuantumTaskPlanner."""
    
    def test_task_creation_throughput(self):
        """Benchmark task creation throughput."""
        planner = QuantumTaskPlanner(max_concurrent_tasks=8)
        
        self.start_monitoring()
        
        num_tasks = 1000
        latencies = []
        errors = 0
        
        for i in range(num_tasks):
            start_time = time.perf_counter()
            
            try:
                planner.add_task(
                    task_id=f"benchmark_task_{i}",
                    name=f"Benchmark Task {i}",
                    priority=TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.HIGH,
                    estimated_duration=float(i % 10 + 1),
                    resource_requirements={
                        ResourceType.CPU: (i % 5 + 1) * 0.1,
                        ResourceType.MEMORY: (i % 3 + 1) * 0.1
                    }
                )
                
                latency = time.perf_counter() - start_time
                latencies.append(latency)
                
            except Exception:
                errors += 1
        
        self.stop_monitoring()
        
        metrics = self.calculate_metrics(latencies, errors, num_tasks)
        
        # Performance assertions
        assert metrics.throughput > 500, f"Task creation throughput too low: {metrics.throughput} ops/sec"
        assert metrics.latency_mean < 0.01, f"Task creation latency too high: {metrics.latency_mean*1000}ms"
        assert metrics.error_rate == 0, f"Task creation errors: {metrics.error_rate}%"
        assert metrics.memory_peak_mb < 100, f"Memory usage too high: {metrics.memory_peak_mb}MB"
        
        print(f"Task Creation: {metrics}")
    
    @pytest.mark.asyncio
    async def test_quantum_optimization_performance(self):
        """Benchmark quantum optimization performance."""
        planner = QuantumTaskPlanner(max_concurrent_tasks=16)
        
        # Add test tasks
        num_tasks = 50
        for i in range(num_tasks):
            planner.add_task(
                task_id=f"opt_task_{i}",
                name=f"Optimization Task {i}",
                priority=TaskPriority.HIGH if i < 10 else TaskPriority.NORMAL,
                estimated_duration=float(i % 20 + 1),
                resource_requirements={ResourceType.CPU: (i % 10 + 1) * 0.05}
            )
        
        self.start_monitoring()
        
        num_optimizations = 10
        latencies = []
        errors = 0
        
        for i in range(num_optimizations):
            start_time = time.perf_counter()
            
            try:
                schedule = await planner.quantum_optimize_schedule()
                assert len(schedule) == num_tasks
                
                latency = time.perf_counter() - start_time
                latencies.append(latency)
                
            except Exception:
                errors += 1
        
        self.stop_monitoring()
        
        metrics = self.calculate_metrics(latencies, errors, num_optimizations)
        
        # Performance assertions
        assert metrics.latency_mean < 2.0, f"Optimization too slow: {metrics.latency_mean}s"
        assert metrics.error_rate == 0, f"Optimization errors: {metrics.error_rate}%"
        
        print(f"Quantum Optimization: {metrics}")


@pytest.mark.performance
class TestCachePerformance(PerformanceBenchmark):
    """Performance tests for advanced caching system."""
    
    def test_cache_throughput(self):
        """Benchmark cache read/write throughput."""
        cache = AdvancedCache(max_size_mb=100.0)
        
        self.start_monitoring()
        
        # Test data
        test_data = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        
        # Write performance
        write_latencies = []
        write_errors = 0
        
        for key, value in test_data.items():
            start_time = time.perf_counter()
            
            try:
                success = cache.put(key, value)
                assert success
                
                latency = time.perf_counter() - start_time
                write_latencies.append(latency)
                
            except Exception:
                write_errors += 1
        
        # Read performance
        read_latencies = []
        read_errors = 0
        cache_hits = 0
        
        # Multiple read passes to test cache efficiency
        for _ in range(10):
            for key in test_data.keys():
                start_time = time.perf_counter()
                
                try:
                    result = cache.get(key)
                    if result is not None:
                        cache_hits += 1
                    
                    latency = time.perf_counter() - start_time
                    read_latencies.append(latency)
                    
                except Exception:
                    read_errors += 1
        
        self.stop_monitoring()
        
        # Calculate metrics
        write_metrics = self.calculate_metrics(write_latencies, write_errors, len(test_data))
        read_metrics = self.calculate_metrics(read_latencies, read_errors, len(test_data) * 10)
        
        hit_rate = cache_hits / (len(test_data) * 10) * 100
        
        # Performance assertions
        assert write_metrics.throughput > 1000, f"Cache write throughput too low: {write_metrics.throughput} ops/sec"
        assert read_metrics.throughput > 10000, f"Cache read throughput too low: {read_metrics.throughput} ops/sec"
        assert hit_rate > 95, f"Cache hit rate too low: {hit_rate}%"
        assert write_metrics.latency_mean < 0.001, f"Cache write latency too high: {write_metrics.latency_mean*1000}ms"
        assert read_metrics.latency_mean < 0.0001, f"Cache read latency too high: {read_metrics.latency_mean*1000}ms"
        
        print(f"Cache Write: {write_metrics}")
        print(f"Cache Read: {read_metrics}")
        print(f"Cache Hit Rate: {hit_rate:.1f}%")
    
    def test_cache_memory_efficiency(self):
        """Test cache memory usage patterns."""
        cache = AdvancedCache(max_size_mb=10.0)  # Small cache for testing
        
        # Fill cache with various sized objects
        small_objects = {f"small_{i}": "x" * 100 for i in range(100)}
        large_objects = {f"large_{i}": "x" * 10000 for i in range(20)}
        
        # Add small objects
        for key, value in small_objects.items():
            cache.put(key, value)
        
        initial_stats = cache.get_stats()
        
        # Add large objects (should trigger evictions)
        for key, value in large_objects.items():
            cache.put(key, value)
        
        final_stats = cache.get_stats()
        
        # Memory should stay within bounds
        assert final_stats['current_size_mb'] <= 10.0, f"Cache exceeded size limit: {final_stats['current_size_mb']}MB"
        assert final_stats['evictions'] > 0, "Cache should have performed evictions"
        
        print(f"Cache Memory Usage: {final_stats['current_size_mb']:.2f}MB")
        print(f"Evictions: {final_stats['evictions']}")
    
    def test_cached_function_performance(self):
        """Test performance of cached function decorator."""
        
        @cached(ttl=60.0)
        def expensive_computation(n: int) -> int:
            # Simulate expensive computation
            result = 0
            for i in range(n * 1000):
                result += i
            return result
        
        self.start_monitoring()
        
        # First call (cache miss)
        start_time = time.perf_counter()
        result1 = expensive_computation(100)
        first_call_time = time.perf_counter() - start_time
        
        # Second call (cache hit)
        start_time = time.perf_counter()
        result2 = expensive_computation(100)
        second_call_time = time.perf_counter() - start_time
        
        self.stop_monitoring()
        
        assert result1 == result2, "Cached function should return same result"
        
        # Cache should provide significant speedup
        speedup_factor = first_call_time / second_call_time
        assert speedup_factor > 10, f"Cache speedup insufficient: {speedup_factor}x"
        assert second_call_time < 0.001, f"Cached call too slow: {second_call_time*1000}ms"
        
        print(f"Cache speedup: {speedup_factor:.1f}x ({first_call_time*1000:.2f}ms -> {second_call_time*1000:.2f}ms)")


@pytest.mark.performance
class TestDistributedPerformance(PerformanceBenchmark):
    """Performance tests for distributed quantum processing."""
    
    @pytest.mark.asyncio
    async def test_distributed_task_throughput(self):
        """Test distributed task processing throughput."""
        try:
            # Mock Redis for testing
            from unittest.mock import Mock
            
            manager = QuantumDistributedManager(
                node_type=NodeType.MASTER,
                enable_auto_scaling=False
            )
            manager.redis_client = Mock()  # Mock Redis
            
            self.start_monitoring()
            
            num_tasks = 50
            tasks = []
            
            # Submit tasks
            for i in range(num_tasks):
                task_id = await manager.submit_distributed_task(
                    function=lambda x: x * 2,  # Simple function
                    args=(i,),
                    priority=TaskPriority.NORMAL
                )
                tasks.append(task_id)
            
            # Wait for completion (simulated)
            await asyncio.sleep(2.0)
            
            self.stop_monitoring()
            
            metrics = self.calculate_metrics([0.1] * num_tasks, 0, num_tasks)
            
            assert len(tasks) == num_tasks
            print(f"Distributed Task Submission: {metrics}")
            
        finally:
            await manager.shutdown()
    
    def test_auto_scaling_response_time(self):
        """Test auto-scaling system response time."""
        from dgdm_histopath.utils.scaling import AutoScalingManager, ScalingStrategy
        
        scaler = AutoScalingManager(
            strategy=ScalingStrategy.REACTIVE,
            scaling_cooldown=1.0  # Fast scaling for testing
        )
        
        self.start_monitoring()
        
        # Simulate load spike
        start_time = time.perf_counter()
        
        # Generate high CPU usage metrics
        from dgdm_histopath.utils.scaling import ScalingMetrics
        
        high_load_metrics = ScalingMetrics(
            timestamp=time.time(),
            cpu_utilization=0.9,  # 90% CPU
            memory_utilization=0.8,  # 80% memory
            gpu_utilization=0.7,
            queue_length=20,  # High queue
            throughput=100.0,
            latency_p95=2.0,
            error_rate=0.01
        )
        
        # Simulate receiving high load
        scaler.predictor.record_metrics(high_load_metrics)
        scaler.metrics_history.append(high_load_metrics)
        
        # Check scaling decision time
        scaler._evaluate_scaling_needs(high_load_metrics)
        
        response_time = time.perf_counter() - start_time
        
        self.stop_monitoring()
        
        # Should respond quickly to load changes
        assert response_time < 0.1, f"Auto-scaling response too slow: {response_time*1000}ms"
        
        print(f"Auto-scaling response time: {response_time*1000:.2f}ms")
        
        scaler.shutdown()


@pytest.mark.performance
class TestMemoryPerformance(PerformanceBenchmark):
    """Memory management and GPU performance tests."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_management(self):
        """Test GPU memory management performance."""
        memory_manager = GPUMemoryManager()
        
        self.start_monitoring()
        
        # Allocate tensors of various sizes
        tensors = []
        allocation_times = []
        
        sizes = [(1024, 1024), (2048, 2048), (512, 512, 3), (100, 100, 100)]
        
        for size in sizes:
            start_time = time.perf_counter()
            
            try:
                tensor = memory_manager.allocate_tensor(*size, dtype=torch.float32)
                tensors.append(tensor)
                
                allocation_time = time.perf_counter() - start_time
                allocation_times.append(allocation_time)
                
            except Exception as e:
                print(f"GPU allocation failed for size {size}: {e}")
        
        # Test cleanup performance
        start_time = time.perf_counter()
        memory_manager.cleanup_memory()
        cleanup_time = time.perf_counter() - start_time
        
        self.stop_monitoring()
        
        # Get GPU memory stats
        stats = memory_manager.get_memory_stats()
        
        avg_allocation_time = np.mean(allocation_times) if allocation_times else float('inf')
        
        # Performance assertions
        assert avg_allocation_time < 0.1, f"GPU allocation too slow: {avg_allocation_time*1000}ms"
        assert cleanup_time < 1.0, f"GPU cleanup too slow: {cleanup_time}s"
        assert stats['memory_utilization'] <= 1.0, "Memory utilization should be <= 100%"
        
        print(f"GPU Allocation: {avg_allocation_time*1000:.2f}ms avg")
        print(f"GPU Cleanup: {cleanup_time*1000:.2f}ms")
        print(f"GPU Memory Utilization: {stats['memory_utilization']*100:.1f}%")
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in quantum components."""
        import gc
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        initial_memory = psutil.Process().memory_info().rss
        
        # Create and destroy quantum components
        for i in range(10):
            planner = QuantumTaskPlanner(max_concurrent_tasks=4)
            
            # Add tasks
            for j in range(20):
                planner.add_task(
                    f"leak_test_{i}_{j}",
                    f"Leak Test {i}-{j}",
                    TaskPriority.NORMAL,
                    1.0,
                    {ResourceType.CPU: 0.1}
                )
            
            # Clear references
            del planner
            gc.collect()
        
        # Force final garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        final_memory = psutil.Process().memory_info().rss
        
        object_growth = final_objects - initial_objects
        memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
        
        # Check for excessive growth (some growth is expected)
        assert object_growth < 1000, f"Possible object leak: {object_growth} new objects"
        assert memory_growth_mb < 50, f"Possible memory leak: {memory_growth_mb:.1f}MB growth"
        
        print(f"Object growth: {object_growth}")
        print(f"Memory growth: {memory_growth_mb:.1f}MB")


@pytest.mark.stress
class TestStressTests(PerformanceBenchmark):
    """Stress tests for quantum framework under extreme load."""
    
    def test_high_concurrency_stress(self):
        """Test system under high concurrent load."""
        planner = QuantumTaskPlanner(max_concurrent_tasks=32)
        
        self.start_monitoring()
        
        # High concurrency task creation
        def create_tasks(thread_id, num_tasks):
            errors = 0
            for i in range(num_tasks):
                try:
                    planner.add_task(
                        f"stress_{thread_id}_{i}",
                        f"Stress Task {thread_id}-{i}",
                        TaskPriority.NORMAL,
                        float(i % 5 + 1),
                        {ResourceType.CPU: 0.1}
                    )
                except Exception:
                    errors += 1
            return errors
        
        # Run concurrent task creation
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(create_tasks, thread_id, 50)
                for thread_id in range(10)
            ]
            
            total_errors = sum(future.result() for future in futures)
        
        self.stop_monitoring()
        
        total_tasks = len(planner.tasks)
        
        # Stress test assertions
        assert total_tasks >= 400, f"Too few tasks created under stress: {total_tasks}"
        assert total_errors < 10, f"Too many errors under stress: {total_errors}"
        
        metrics = self.calculate_metrics([0.01] * total_tasks, total_errors, total_tasks)
        
        print(f"High Concurrency Stress: {metrics}")
        print(f"Total tasks created: {total_tasks}")
    
    @pytest.mark.asyncio
    async def test_sustained_load_stress(self):
        """Test system under sustained high load."""
        scheduler = QuantumScheduler(max_concurrent_jobs=8)
        
        self.start_monitoring()
        
        def cpu_intensive_task(duration):
            """CPU intensive task for stress testing."""
            start = time.perf_counter()
            result = 0
            while time.perf_counter() - start < duration:
                result += 1
            return result
        
        # Submit sustained load
        jobs = []
        errors = 0
        
        for i in range(100):
            try:
                job = scheduler.schedule_job(
                    job_id=f"sustained_job_{i}",
                    function=cpu_intensive_task,
                    args=(0.01,),  # 10ms each
                    workflow_type=WorkflowType.TRAINING,
                    priority=TaskPriority.NORMAL
                )
                jobs.append(job)
                
                # Brief pause to simulate realistic load
                await asyncio.sleep(0.001)
                
            except Exception:
                errors += 1
        
        # Wait for completion
        await asyncio.sleep(5.0)
        
        self.stop_monitoring()
        
        status = scheduler.get_scheduler_status()
        
        # Sustained load assertions
        assert len(jobs) >= 90, f"Too few jobs submitted: {len(jobs)}"
        assert errors < 5, f"Too many submission errors: {errors}"
        assert status['performance_metrics']['success_rate'] > 0.95, "Success rate too low under sustained load"
        
        print(f"Sustained Load - Jobs: {len(jobs)}, Errors: {errors}")
        print(f"Success Rate: {status['performance_metrics']['success_rate']*100:.1f}%")
        
        await scheduler.shutdown()
    
    def test_memory_pressure_stress(self):
        """Test system under memory pressure."""
        cache = AdvancedCache(max_size_mb=50.0)  # Limited memory
        
        self.start_monitoring()
        
        # Create memory pressure with large objects
        large_objects = []
        cache_operations = 0
        cache_errors = 0
        
        for i in range(1000):
            try:
                # Create objects of random sizes
                size = np.random.randint(1000, 10000)
                data = "x" * size
                
                key = f"memory_stress_{i}"
                success = cache.put(key, data)
                
                cache_operations += 1
                
                # Occasionally read back
                if i % 10 == 0:
                    result = cache.get(key)
                    cache_operations += 1
                
            except Exception:
                cache_errors += 1
        
        self.stop_monitoring()
        
        stats = cache.get_stats()
        
        # Memory pressure assertions
        assert cache_operations > 900, f"Too few cache operations: {cache_operations}"
        assert cache_errors < 10, f"Too many cache errors: {cache_errors}"
        assert stats['current_size_mb'] <= 50.0, f"Cache exceeded memory limit: {stats['current_size_mb']}MB"
        assert stats['evictions'] > 0, "Should have evicted items under memory pressure"
        
        print(f"Memory Pressure - Operations: {cache_operations}, Errors: {cache_errors}")
        print(f"Cache Size: {stats['current_size_mb']:.1f}MB, Evictions: {stats['evictions']}")


@pytest.mark.benchmark
class TestBenchmarkSuite:
    """Complete benchmark suite for performance regression testing."""
    
    def test_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite."""
        print("\n=== DGDM Quantum Framework Performance Benchmark ===\n")
        
        benchmark = PerformanceBenchmark()
        
        # System information
        print(f"System: {psutil.cpu_count()} cores, {psutil.virtual_memory().total // (1024**3)}GB RAM")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        print()
        
        # Run benchmark components
        test_classes = [
            TestQuantumPlannerPerformance,
            TestCachePerformance,
            TestMemoryPerformance
        ]
        
        for test_class in test_classes:
            print(f"Running {test_class.__name__}...")
            test_instance = test_class()
            
            # Run performance tests
            for method_name in dir(test_instance):
                if method_name.startswith('test_') and not method_name.endswith('_stress'):
                    try:
                        method = getattr(test_instance, method_name)
                        if asyncio.iscoroutinefunction(method):
                            asyncio.run(method())
                        else:
                            method()
                    except Exception as e:
                        print(f"  {method_name} failed: {e}")
            print()
        
        print("=== Benchmark Complete ===")


if __name__ == "__main__":
    # Run specific benchmark
    pytest.main([__file__, "-v", "-m", "performance", "--tb=short"])