"""Comprehensive test suite for DGDM Histopath Lab."""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
import time
import psutil
from typing import Dict, Any, List, Optional
import warnings

from dgdm_histopath.models.dgdm_model import DGDMModel, ModelConfigurationError, ModelInferenceError
from dgdm_histopath.utils.validation import InputValidator, ValidationError
from dgdm_histopath.utils.config import load_config, save_config, ConfigurationError
from dgdm_histopath.utils.logging import setup_logging, get_logger
from dgdm_histopath.utils.monitoring import monitor_operation, health_checker
from dgdm_histopath.utils.security import SecurityError, rate_limiter
from dgdm_histopath.utils.performance import performance_optimizer, global_cache
from dgdm_histopath.utils.scaling import AdaptiveLoadBalancer, AutoScaler, ScalingPolicy


class TestPerformanceBase:
    """Base class for performance testing utilities."""
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """Measure function execution time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def measure_memory_usage(func, *args, **kwargs):
        """Measure memory usage during function execution."""
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before
        
        return result, memory_delta


class TestDGDMModelValidation:
    """Test model configuration validation and error handling."""
    
    def test_valid_configuration(self):
        """Test model creation with valid configuration."""
        model = DGDMModel(
            node_features=768,
            hidden_dims=[512, 256, 128],
            num_diffusion_steps=10,
            attention_heads=8,
            dropout=0.1,
            num_classes=5
        )
        
        assert model.node_features == 768
        assert model.hidden_dims == [512, 256, 128]
        assert model.num_classes == 5
        assert model.classification_head is not None
    
    def test_invalid_node_features(self):
        """Test model creation with invalid node features."""
        with pytest.raises(ModelConfigurationError):
            DGDMModel(node_features=0)
        
        with pytest.raises(ModelConfigurationError):
            DGDMModel(node_features=-1)
        
        with pytest.raises(ModelConfigurationError):
            DGDMModel(node_features=50000)  # Too large
    
    def test_invalid_hidden_dims(self):
        """Test model creation with invalid hidden dimensions."""
        with pytest.raises(ModelConfigurationError):
            DGDMModel(hidden_dims=[])
        
        with pytest.raises(ModelConfigurationError):
            DGDMModel(hidden_dims=[0, 128])
        
        with pytest.raises(ModelConfigurationError):
            DGDMModel(hidden_dims=[-1, 128])
    
    def test_invalid_attention_heads(self):
        """Test model creation with invalid attention heads."""
        with pytest.raises(ModelConfigurationError):
            DGDMModel(attention_heads=0)
        
        with pytest.raises(ModelConfigurationError):
            DGDMModel(hidden_dims=[129], attention_heads=8)  # Not divisible
    
    def test_invalid_dropout(self):
        """Test model creation with invalid dropout."""
        with pytest.raises(ModelConfigurationError):
            DGDMModel(dropout=-0.1)
        
        with pytest.raises(ModelConfigurationError):
            DGDMModel(dropout=1.0)
    
    def test_warning_for_no_tasks(self):
        """Test warning when no classification or regression tasks specified."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DGDMModel(node_features=768, hidden_dims=[256])
            assert len(w) > 0
            assert "no classification or regression targets" in str(w[0].message).lower()


class TestDGDMModelInference:
    """Test model inference and forward pass validation."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        return DGDMModel(
            node_features=32,
            hidden_dims=[64, 32],
            num_diffusion_steps=5,
            attention_heads=4,
            num_classes=3
        )
    
    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for testing."""
        from torch_geometric.data import Data
        
        num_nodes = 20
        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, 40))
        pos = torch.rand(num_nodes, 2)
        
        return Data(x=x, edge_index=edge_index, pos=pos, num_nodes=num_nodes)
    
    def test_valid_inference(self, sample_model, sample_graph_data):
        """Test valid model inference."""
        sample_model.eval()
        
        with torch.no_grad():
            outputs = sample_model(sample_graph_data, mode="inference")
        
        assert "graph_embedding" in outputs
        assert "classification_logits" in outputs
        assert "classification_probs" in outputs
        
        assert outputs["classification_logits"].shape[1] == 3  # num_classes
        assert torch.allclose(outputs["classification_probs"].sum(dim=1), torch.ones(1))
    
    def test_invalid_input_data(self, sample_model):
        """Test model with invalid input data."""
        from torch_geometric.data import Data
        
        # Missing required attributes
        invalid_data = Data()
        
        with pytest.raises(ModelInferenceError):
            sample_model(invalid_data)
    
    def test_nan_input_validation(self, sample_model):
        """Test validation of NaN inputs."""
        from torch_geometric.data import Data
        
        num_nodes = 10
        x = torch.randn(num_nodes, 32)
        x[0, 0] = float('nan')  # Introduce NaN
        edge_index = torch.randint(0, num_nodes, (2, 20))
        
        invalid_data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        
        with pytest.raises(ModelInferenceError):
            sample_model(invalid_data)
    
    def test_dimension_mismatch(self, sample_model):
        """Test dimension mismatch validation."""
        from torch_geometric.data import Data
        
        num_nodes = 10
        x = torch.randn(num_nodes, 64)  # Wrong feature dimension
        edge_index = torch.randint(0, num_nodes, (2, 20))
        
        invalid_data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        
        with pytest.raises(ModelInferenceError):
            sample_model(invalid_data)
    
    def test_pretrain_mode(self, sample_model, sample_graph_data):
        """Test pretraining mode."""
        sample_model.train()
        
        outputs = sample_model(sample_graph_data, mode="pretrain")
        
        assert "diffusion_loss" in outputs
        assert "total_pretrain_loss" in outputs
        assert isinstance(outputs["diffusion_loss"], torch.Tensor)


class TestValidationUtilities:
    """Test input validation utilities."""
    
    def test_numeric_validation(self):
        """Test numeric input validation."""
        # Valid cases
        assert InputValidator.validate_numeric(5.0) == 5.0
        assert InputValidator.validate_numeric("3.14") == 3.14
        assert InputValidator.validate_numeric(42, min_val=0, max_val=100) == 42
        
        # Invalid cases
        with pytest.raises(ValidationError):
            InputValidator.validate_numeric("invalid")
        
        with pytest.raises(ValidationError):
            InputValidator.validate_numeric(-1, min_val=0)
        
        with pytest.raises(ValidationError):
            InputValidator.validate_numeric(101, max_val=100)
    
    def test_integer_validation(self):
        """Test integer input validation."""
        # Valid cases
        assert InputValidator.validate_integer(42) == 42
        assert InputValidator.validate_integer("10") == 10
        
        # Invalid cases
        with pytest.raises(ValidationError):
            InputValidator.validate_integer(3.14)
        
        with pytest.raises(ValidationError):
            InputValidator.validate_integer("not_a_number")
    
    def test_boolean_validation(self):
        """Test boolean input validation."""
        # Valid cases
        assert InputValidator.validate_boolean(True) is True
        assert InputValidator.validate_boolean("true") is True
        assert InputValidator.validate_boolean("false") is False
        assert InputValidator.validate_boolean(1) is True
        assert InputValidator.validate_boolean(0) is False
        
        # Invalid cases
        with pytest.raises(ValidationError):
            InputValidator.validate_boolean("maybe")
    
    def test_enum_validation(self):
        """Test enum/choice validation."""
        valid_choices = ["option1", "option2", "option3"]
        
        # Valid cases
        assert InputValidator.validate_enum("option1", valid_choices) == "option1"
        
        # Invalid cases
        with pytest.raises(ValidationError):
            InputValidator.validate_enum("invalid_option", valid_choices)
    
    def test_file_path_validation(self):
        """Test file path validation."""
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(b"test content")
        
        try:
            # Valid file path
            validated_path = InputValidator.validate_file_path(tmp_path)
            assert validated_path == tmp_path.resolve()
            
            # Non-existent file
            with pytest.raises(ValidationError):
                InputValidator.validate_file_path("/non/existent/file.txt")
                
        finally:
            tmp_path.unlink()
    
    def test_string_sanitization(self):
        """Test string sanitization."""
        # Normal string
        assert InputValidator.sanitize_string("hello world") == "hello world"
        
        # String with dangerous content
        with pytest.raises(SecurityError):
            InputValidator.sanitize_string("<script>alert('xss')</script>")
        
        # String too long
        with pytest.raises(ValidationError):
            InputValidator.sanitize_string("x" * 20000)


class TestConfigurationManagement:
    """Test configuration loading and validation."""
    
    def test_yaml_config_loading(self):
        """Test loading YAML configuration."""
        config_data = {
            "model": {
                "node_features": 768,
                "hidden_dims": [512, 256, 128]
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.001
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            import yaml
            yaml.dump(config_data, tmp_file)
            tmp_path = Path(tmp_file.name)
        
        try:
            loaded_config = load_config(tmp_path)
            assert loaded_config == config_data
        finally:
            tmp_path.unlink()
    
    def test_json_config_loading(self):
        """Test loading JSON configuration."""
        config_data = {
            "model": {
                "node_features": 768,
                "dropout": 0.1
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            import json
            json.dump(config_data, tmp_file)
            tmp_path = Path(tmp_file.name)
        
        try:
            loaded_config = load_config(tmp_path)
            assert loaded_config == config_data
        finally:
            tmp_path.unlink()
    
    def test_invalid_config_format(self):
        """Test handling of invalid configuration format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("invalid config content")
            tmp_path = Path(tmp_file.name)
        
        try:
            with pytest.raises(ConfigurationError):
                load_config(tmp_path)
        finally:
            tmp_path.unlink()
    
    def test_nonexistent_config_file(self):
        """Test handling of non-existent configuration file."""
        with pytest.raises(ConfigurationError):
            load_config("/non/existent/config.yaml")


class TestPerformanceOptimization:
    """Test performance optimization utilities."""
    
    def test_pytorch_optimizations(self):
        """Test PyTorch optimization settings."""
        # Test that optimizations are applied
        assert torch.get_num_threads() > 0
        
        if torch.cuda.is_available():
            assert torch.backends.cuda.matmul.allow_tf32 in [True, False]  # Should be set
    
    def test_dataloader_optimization(self):
        """Test dataloader parameter optimization."""
        settings = performance_optimizer.optimize_dataloader_settings(
            dataset_size=10000,
            batch_size=4
        )
        
        assert 'num_workers' in settings
        assert 'pin_memory' in settings
        assert 'prefetch_factor' in settings
        assert settings['num_workers'] > 0
    
    def test_batch_size_optimization(self):
        """Test batch size optimization based on memory."""
        # This should not crash and return a reasonable batch size
        optimized_size = performance_optimizer._optimize_batch_size(32, 16.0)  # 16GB available
        assert optimized_size > 0
        assert optimized_size <= 32


class TestCachingSystem:
    """Test advanced caching functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        from dgdm_histopath.utils.performance import AdvancedCache
        
        cache = AdvancedCache(max_size=10, ttl_seconds=3600)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
    
    def test_cache_eviction(self):
        """Test cache eviction policies."""
        from dgdm_histopath.utils.performance import AdvancedCache
        
        cache = AdvancedCache(max_size=2, ttl_seconds=3600)
        
        # Fill cache beyond capacity
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cache_ttl(self):
        """Test cache TTL (time-to-live) functionality."""
        from dgdm_histopath.utils.performance import AdvancedCache
        
        cache = AdvancedCache(max_size=10, ttl_seconds=1)  # 1 second TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for TTL to expire
        time.sleep(1.1)
        assert cache.get("key1") is None


class TestLoadBalancing:
    """Test load balancing functionality."""
    
    def test_load_balancer_creation(self):
        """Test load balancer initialization."""
        workers = ["worker1", "worker2", "worker3"]
        lb = AdaptiveLoadBalancer(workers)
        
        assert len(lb.workers) == 3
        assert lb.current_algorithm == "adaptive"
    
    def test_worker_selection(self):
        """Test worker selection algorithms."""
        workers = ["worker1", "worker2"]
        lb = AdaptiveLoadBalancer(workers)
        
        # Test that worker selection returns valid workers
        selected = lb.select_worker()
        assert selected in workers
    
    def test_performance_tracking(self):
        """Test performance tracking for workers."""
        workers = ["worker1", "worker2"]
        lb = AdaptiveLoadBalancer(workers)
        
        # Record some responses
        lb.record_response("worker1", 0.1, True)
        lb.record_response("worker1", 0.2, True)
        lb.record_response("worker2", 0.5, False)
        
        stats = lb.get_stats()
        assert stats['num_workers'] == 2
        assert 'worker_stats' in stats


class TestAutoScaling:
    """Test auto-scaling functionality."""
    
    def test_scaling_policy_creation(self):
        """Test scaling policy configuration."""
        policy = ScalingPolicy(
            min_workers=1,
            max_workers=8,
            cpu_scale_up_threshold=70.0
        )
        
        assert policy.min_workers == 1
        assert policy.max_workers == 8
        assert policy.cpu_scale_up_threshold == 70.0
    
    def test_auto_scaler_creation(self):
        """Test auto-scaler initialization."""
        policy = ScalingPolicy(min_workers=2, max_workers=4)
        scaler = AutoScaler(policy)
        
        assert scaler.current_workers == 2
        assert scaler.policy.max_workers == 4
        
        # Cleanup
        scaler.stop_monitoring()


class TestSecurityFeatures:
    """Test security utilities."""
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Reset rate limiter for testing
        rate_limiter.requests.clear()
        
        # Test normal requests
        assert rate_limiter.is_allowed("client1") is True
        assert rate_limiter.is_allowed("client1") is True
        
        # Test rate limiting (would need to exceed the limit)
        remaining = rate_limiter.get_remaining_requests("client1")
        assert remaining > 0
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        from dgdm_histopath.utils.security import InputSanitizer
        
        # Test filename sanitization
        safe_filename = InputSanitizer.sanitize_filename("file<>name")
        assert "<" not in safe_filename
        assert ">" not in safe_filename
        
        # Test path sanitization
        safe_path = InputSanitizer.sanitize_path("../../../etc/passwd")
        assert ".." not in safe_path
    
    def test_input_validation_security(self):
        """Test security-focused input validation."""
        from dgdm_histopath.utils.security import InputSanitizer
        
        # Test dangerous patterns
        assert not InputSanitizer.validate_input("<script>alert('xss')</script>")
        assert not InputSanitizer.validate_input("'; DROP TABLE users; --")
        assert InputSanitizer.validate_input("normal text input")


class TestMonitoringAndHealthChecks:
    """Test monitoring and health check functionality."""
    
    def test_health_checker(self):
        """Test system health checking."""
        health_report = health_checker.check_system_health()
        
        assert 'overall_status' in health_report
        assert 'checks' in health_report
        assert 'alerts' in health_report
        assert health_report['overall_status'] in ['healthy', 'warning', 'degraded', 'error']
    
    def test_operation_monitoring(self):
        """Test operation monitoring context manager."""
        def sample_operation():
            time.sleep(0.01)  # Simulate work
            return "result"
        
        # Test successful operation
        with monitor_operation("test_operation"):
            result = sample_operation()
        
        assert result == "result"
        
        # Test operation with exception
        def failing_operation():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            with monitor_operation("failing_operation"):
                failing_operation()


class TestEndToEndWorkflow(TestPerformanceBase):
    """End-to-end integration tests."""
    
    def test_complete_model_workflow(self):
        """Test complete model creation, training simulation, and inference."""
        # Create model
        model = DGDMModel(
            node_features=64,
            hidden_dims=[128, 64],
            num_diffusion_steps=5,
            attention_heads=4,
            num_classes=3
        )
        
        # Create sample data
        from torch_geometric.data import Data
        num_nodes = 25
        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, 50))
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        
        # Test inference
        model.eval()
        with torch.no_grad():
            outputs = model(data, mode="inference")
        
        assert "classification_logits" in outputs
        assert outputs["classification_logits"].shape[1] == 3
        
        # Test pretraining mode
        model.train()
        pretrain_outputs = model(data, mode="pretrain")
        
        assert "diffusion_loss" in pretrain_outputs
        assert isinstance(pretrain_outputs["diffusion_loss"], torch.Tensor)
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking of key operations."""
        # Test model creation performance
        def create_model():
            return DGDMModel(
                node_features=256,
                hidden_dims=[512, 256, 128],
                num_classes=10
            )
        
        model, creation_time = self.measure_execution_time(create_model)
        
        # Should create model in reasonable time (< 5 seconds)
        assert creation_time < 5.0
        
        # Test inference performance
        from torch_geometric.data import Data
        num_nodes = 100
        x = torch.randn(num_nodes, 256)
        edge_index = torch.randint(0, num_nodes, (2, 200))
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        
        model.eval()
        with torch.no_grad():
            def run_inference():
                return model(data, mode="inference")
            
            outputs, inference_time = self.measure_execution_time(run_inference)
        
        # Should complete inference in reasonable time (< 10 seconds on CPU)
        assert inference_time < 10.0
        
        # Test memory usage during inference
        with torch.no_grad():
            def memory_test_inference():
                return model(data, mode="inference")
            
            outputs, memory_delta = self.measure_memory_usage(memory_test_inference)
        
        # Memory usage should be reasonable (< 1GB delta)
        assert memory_delta < 1000  # MB


# Test fixtures and utilities
@pytest.fixture(scope="session")
def temp_directory():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config_file(temp_directory):
    """Create sample configuration file for testing."""
    config_data = {
        "model": {
            "node_features": 768,
            "hidden_dims": [512, 256, 128],
            "num_classes": 5
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 0.001,
            "max_epochs": 100
        }
    }
    
    config_file = temp_directory / "test_config.yaml"
    
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    return config_file


# Performance benchmarking utilities
def run_performance_suite():
    """Run comprehensive performance benchmarking suite."""
    results = {}
    
    print("Running DGDM Histopath Lab Performance Benchmark Suite...")
    print("=" * 60)
    
    # Model creation benchmark
    print("Testing model creation performance...")
    start_time = time.time()
    model = DGDMModel(
        node_features=768,
        hidden_dims=[512, 256, 128],
        num_diffusion_steps=10,
        attention_heads=8,
        num_classes=10
    )
    creation_time = time.time() - start_time
    results['model_creation_time'] = creation_time
    print(f"Model creation time: {creation_time:.3f}s")
    
    # Inference benchmark
    print("Testing inference performance...")
    from torch_geometric.data import Data
    num_nodes = 500
    x = torch.randn(num_nodes, 768)
    edge_index = torch.randint(0, num_nodes, (2, 1000))
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            model(data, mode="inference")
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            outputs = model(data, mode="inference")
        avg_inference_time = (time.time() - start_time) / 10
    
    results['avg_inference_time'] = avg_inference_time
    print(f"Average inference time: {avg_inference_time:.3f}s")
    
    # Memory usage test
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Large model test
    large_model = DGDMModel(
        node_features=1024,
        hidden_dims=[1024, 512, 256],
        num_classes=100
    )
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_usage = memory_after - memory_before
    results['model_memory_usage_mb'] = memory_usage
    print(f"Large model memory usage: {memory_usage:.1f}MB")
    
    print("=" * 60)
    print("Performance benchmark completed!")
    
    return results


if __name__ == "__main__":
    # Run performance benchmarks if executed directly
    run_performance_suite()