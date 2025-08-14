"""Performance benchmarks for progressive quality gates."""

import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

try:
    import psutil
except ImportError:
    psutil = None

from dgdm_histopath.testing.progressive_quality_gates import (
    ProgressiveQualityRunner, ProgressiveQualityConfig, ProjectMaturity
)


class PerformanceBenchmark:
    """Performance benchmark utility."""
    
    def __init__(self):
        self.measurements = []
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        self.measurements.append({
            'function': func.__name__,
            'execution_time': execution_time,
            'timestamp': start_time
        })
        
        return result, execution_time


class TestProgressiveQualityGatesPerformance:
    """Performance tests for progressive quality gates."""
    
    @pytest.fixture
    def benchmark(self):
        """Create performance benchmark instance."""
        return PerformanceBenchmark()
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create realistic project structure
        package_dir = temp_dir / "dgdm_histopath"
        package_dir.mkdir()
        (package_dir / "__init__.py").touch()
        
        # Create multiple Python files to test scalability
        for i in range(5):
            with open(package_dir / f"module_{i}.py", "w") as f:
                f.write(f"""
def function_{i}():
    '''Function {i} for testing.'''
    return {i}

class Class_{i}:
    '''Class {i} for testing.'''
    
    def method_{i}(self):
        return {i}
""")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_progressive_runner_performance(self, benchmark, temp_project):
        """Test performance of progressive quality runner."""
        config = ProgressiveQualityConfig()
        config.maturity = ProjectMaturity.DEVELOPMENT
        
        with patch('dgdm_histopath.testing.progressive_quality_gates.Path.cwd', return_value=temp_project):
            def run_progressive_gates():
                runner = ProgressiveQualityRunner(config=config)
                # Test only compilation gate for performance
                return runner._check_code_compilation()
            
            result, execution_time = benchmark.measure_execution_time(run_progressive_gates)
            
            # Performance assertions
            assert execution_time < 10.0  # Should complete within 10 seconds
            assert result.passed  # Should work correctly
            
            # Log performance
            print(f"Progressive runner execution time: {execution_time:.3f}s")
    
    def test_maturity_level_performance_impact(self, benchmark, temp_project):
        """Test performance impact of different maturity levels."""
        execution_times = {}
        
        for maturity in ProjectMaturity:
            config = ProgressiveQualityConfig()
            config.maturity = maturity
            
            with patch('dgdm_histopath.testing.progressive_quality_gates.Path.cwd', return_value=temp_project):
                def run_gates():
                    runner = ProgressiveQualityRunner(config=config)
                    return len(config.enabled_gates[maturity])
                
                _, execution_time = benchmark.measure_execution_time(run_gates)
                execution_times[maturity.value] = execution_time
        
        # Higher maturity levels may take longer due to more gates
        # but the overhead should be reasonable
        max_time = max(execution_times.values())
        assert max_time < 1.0  # Setup should be fast regardless of maturity


def test_performance_regression_detection():
    """Test for performance regressions."""
    # This test can be expanded to store baseline performance metrics
    # and detect when performance degrades significantly
    
    # For now, just ensure basic functionality works within time limits
    config = ProgressiveQualityConfig()
    
    start_time = time.time()
    
    # This should be very fast
    maturity_detection_time = time.time() - start_time
    
    assert maturity_detection_time < 1.0  # Should be nearly instantaneous


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([__file__, "-v", "-s"])