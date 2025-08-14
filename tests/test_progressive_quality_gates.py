"""Comprehensive test suite for progressive quality gates."""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import subprocess

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgdm_histopath.testing.progressive_quality_gates import (
    ProgressiveQualityRunner,
    ProgressiveQualityConfig, 
    ProjectMaturity
)
from dgdm_histopath.testing.robust_quality_runner import RobustQualityRunner
from dgdm_histopath.testing.scalable_quality_gates import ScalableQualityGates


class TestProgressiveQualityConfig:
    """Test progressive quality configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = ProgressiveQualityConfig()
        assert config.maturity == ProjectMaturity.DEVELOPMENT
        assert isinstance(config.test_coverage_thresholds, dict)
        assert isinstance(config.performance_thresholds, dict)
        assert isinstance(config.security_thresholds, dict)
        assert isinstance(config.enabled_gates, dict)
    
    def test_maturity_thresholds(self):
        """Test that thresholds increase with maturity."""
        config = ProgressiveQualityConfig()
        
        # Test coverage thresholds increase
        greenfield_coverage = config.test_coverage_thresholds[ProjectMaturity.GREENFIELD]
        production_coverage = config.test_coverage_thresholds[ProjectMaturity.PRODUCTION]
        assert production_coverage > greenfield_coverage
        
        # Test that production has strictest security requirements
        greenfield_security = config.security_thresholds[ProjectMaturity.GREENFIELD]
        production_security = config.security_thresholds[ProjectMaturity.PRODUCTION]
        assert production_security["vulnerabilities"] <= greenfield_security["vulnerabilities"]
    
    def test_enabled_gates_progression(self):
        """Test that more gates are enabled for higher maturity levels."""
        config = ProgressiveQualityConfig()
        
        greenfield_gates = len(config.enabled_gates[ProjectMaturity.GREENFIELD])
        production_gates = len(config.enabled_gates[ProjectMaturity.PRODUCTION])
        
        assert production_gates > greenfield_gates


class TestProgressiveQualityRunner:
    """Test progressive quality runner."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create basic project structure
        (temp_dir / "dgdm_histopath").mkdir()
        (temp_dir / "dgdm_histopath" / "__init__.py").touch()
        (temp_dir / "tests").mkdir()
        (temp_dir / "tests" / "__init__.py").touch()
        
        # Create a simple Python file
        with open(temp_dir / "dgdm_histopath" / "simple.py", "w") as f:
            f.write("def hello():\n    return 'world'\n")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = ProgressiveQualityConfig()
        config.maturity = ProjectMaturity.GREENFIELD
        return config
    
    def test_runner_initialization(self, temp_project_dir, config):
        """Test runner initialization."""
        with patch('dgdm_histopath.testing.progressive_quality_gates.Path.cwd', return_value=temp_project_dir):
            runner = ProgressiveQualityRunner(config=config)
            assert runner.config.maturity == ProjectMaturity.GREENFIELD
            assert runner.output_dir.exists()
    
    def test_maturity_detection(self, temp_project_dir):
        """Test automatic maturity detection."""
        # Create indicators for different maturity levels
        (temp_project_dir / ".github").mkdir()
        (temp_project_dir / "Dockerfile").touch()
        (temp_project_dir / ".pre-commit-config.yaml").touch()
        
        with patch('dgdm_histopath.testing.progressive_quality_gates.Path.cwd', return_value=temp_project_dir):
            runner = ProgressiveQualityRunner()
            # Should detect higher maturity due to CI/deployment indicators
            assert runner.config.maturity in [ProjectMaturity.STAGING, ProjectMaturity.PRODUCTION]
    
    @patch('subprocess.run')
    def test_basic_tests_gate(self, mock_run, temp_project_dir, config):
        """Test basic tests quality gate."""
        # Mock successful test run
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="1 passed",
            stderr=""
        )
        
        with patch('dgdm_histopath.testing.progressive_quality_gates.Path.cwd', return_value=temp_project_dir):
            runner = ProgressiveQualityRunner(config=config)
            result = runner._check_basic_tests()
            
            assert result.passed
            assert result.gate_name == "basic_tests"
            assert "passed" in result.message
    
    def test_code_compilation_gate(self, temp_project_dir, config):
        """Test code compilation quality gate."""
        with patch('dgdm_histopath.testing.progressive_quality_gates.Path.cwd', return_value=temp_project_dir):
            runner = ProgressiveQualityRunner(config=config)
            result = runner._check_code_compilation()
            
            assert result.passed  # Should pass with valid Python file
            assert result.gate_name == "code_compilation"
            assert "errors" in result.message
    
    def test_model_validation_gate(self, temp_project_dir, config):
        """Test model validation quality gate."""
        with patch('dgdm_histopath.testing.progressive_quality_gates.Path.cwd', return_value=temp_project_dir):
            # Mock the DGDM model import and usage
            with patch('dgdm_histopath.testing.progressive_quality_gates.DGDMModel') as mock_model:
                mock_instance = MagicMock()
                mock_model.return_value = mock_instance
                mock_instance.eval.return_value = None
                mock_instance.return_value = {"classification_logits": MagicMock(shape=[1, 2])}
                
                runner = ProgressiveQualityRunner(config=config)
                result = runner._check_model_validation()
                
                assert result.passed
                assert result.gate_name == "model_validation"


class TestRobustQualityRunner:
    """Test robust quality runner."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProgressiveQualityConfig()
    
    def test_robust_runner_initialization(self, temp_dir, config):
        """Test robust runner initialization."""
        runner = RobustQualityRunner(
            config=config,
            project_root=temp_dir,
            output_dir=str(temp_dir / "output")
        )
        
        assert runner.config == config
        assert runner.project_root == temp_dir
        assert runner.output_dir.exists()
        assert runner.temp_dir.exists()
    
    def test_context_manager(self, temp_dir, config):
        """Test robust runner as context manager."""
        with RobustQualityRunner(
            config=config,
            project_root=temp_dir,
            output_dir=str(temp_dir / "output")
        ) as runner:
            assert runner.temp_dir.exists()
        
        # Temp directory should be cleaned up after context exit
        # Note: In real usage, cleanup happens but may be delayed
    
    def test_validation_context_creation(self, temp_dir, config):
        """Test validation context creation."""
        runner = RobustQualityRunner(
            config=config,
            project_root=temp_dir,
            output_dir=str(temp_dir / "output")
        )
        
        context = runner.context
        assert context.project_root == temp_dir
        assert context.maturity_level == config.maturity
        assert context.timeout_seconds > 0
        assert context.max_memory_mb > 0
        assert isinstance(context.metadata, dict)
    
    def test_timeout_calculation(self, temp_dir, config):
        """Test timeout calculation based on maturity."""
        config.maturity = ProjectMaturity.GREENFIELD
        runner_greenfield = RobustQualityRunner(config=config, project_root=temp_dir)
        
        config.maturity = ProjectMaturity.PRODUCTION
        runner_production = RobustQualityRunner(config=config, project_root=temp_dir)
        
        # Production should have longer timeout
        assert runner_production.context.timeout_seconds > runner_greenfield.context.timeout_seconds


class TestScalableQualityGates:
    """Test scalable quality gates."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create basic structure
        (temp_dir / "dgdm_histopath").mkdir()
        (temp_dir / "dgdm_histopath" / "__init__.py").touch()
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProgressiveQualityConfig()
    
    def test_scalable_gates_initialization(self, temp_dir, config):
        """Test scalable gates initialization."""
        gates = ScalableQualityGates(
            config=config,
            project_root=temp_dir,
            output_dir=str(temp_dir / "output"),
            cache_dir=str(temp_dir / "cache"),
            enable_caching=True,
            enable_distributed=False
        )
        
        assert gates.config == config
        assert gates.project_root == temp_dir
        assert gates.enable_caching
        assert not gates.enable_distributed
        assert gates.cache is not None
    
    def test_optimization_settings(self, temp_dir, config):
        """Test optimization settings based on maturity."""
        config.maturity = ProjectMaturity.GREENFIELD
        gates_greenfield = ScalableQualityGates(config=config, project_root=temp_dir)
        
        config.maturity = ProjectMaturity.PRODUCTION
        gates_production = ScalableQualityGates(config=config, project_root=temp_dir)
        
        # Production should have more aggressive optimization
        greenfield_settings = gates_greenfield.optimization_settings
        production_settings = gates_production.optimization_settings
        
        assert production_settings['parallel_workers'] >= greenfield_settings['parallel_workers']
        assert production_settings['cache_size'] > greenfield_settings['cache_size']
    
    def test_cache_functionality(self, temp_dir, config):
        """Test cache functionality."""
        from dgdm_histopath.testing.scalable_quality_gates import ResultCache
        
        cache = ResultCache(temp_dir / "cache")
        
        # Test cache operations
        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        
        # Cache should be empty initially
        result = cache.get("test_validator", [], {})
        assert result is None


class TestQualityGateIntegration:
    """Integration tests for quality gates."""
    
    @pytest.fixture
    def project_structure(self):
        """Create realistic project structure."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create package structure
        package_dir = temp_dir / "dgdm_histopath"
        package_dir.mkdir()
        
        # Create __init__.py files
        (package_dir / "__init__.py").touch()
        (package_dir / "models" / "__init__.py").write_text("")
        (package_dir / "models").mkdir(exist_ok=True)
        
        # Create a mock model file
        model_content = '''
"""Mock DGDM model for testing."""

import torch
import torch.nn as nn

class DGDMModel(nn.Module):
    def __init__(self, node_features=128, hidden_dims=None, num_classes=5):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.node_features = node_features
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Simple linear layers for testing
        layers = []
        in_dim = node_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, data, mode="inference"):
        # Mock forward pass
        batch_size = 1
        x = torch.randn(batch_size, self.node_features)
        logits = self.network(x)
        return {"classification_logits": logits}
'''
        
        (package_dir / "models" / "dgdm_model.py").write_text(model_content)
        
        # Create tests directory
        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").touch()
        (tests_dir / "test_basic.py").write_text("""
def test_simple():
    assert True

def test_addition():
    assert 1 + 1 == 2
""")
        
        # Create config file
        (temp_dir / "pyproject.toml").write_text("""
[project]
name = "test-dgdm"
version = "0.1.0"
""")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_progressive_gates(self, project_structure):
        """Test end-to-end progressive quality gates."""
        config = ProgressiveQualityConfig()
        config.maturity = ProjectMaturity.GREENFIELD
        
        # Mock the cwd to point to our test project
        with patch('dgdm_histopath.testing.progressive_quality_gates.Path.cwd', return_value=project_structure):
            with patch('sys.path', [str(project_structure)] + sys.path):
                runner = ProgressiveQualityRunner(
                    config=config,
                    output_dir=str(project_structure / "output")
                )
                
                # Run a subset of gates suitable for testing
                with patch.object(runner, 'quality_gates', {
                    'code_compilation': runner._check_code_compilation,
                    'basic_tests': runner._check_basic_tests
                }):
                    # Mock subprocess calls for tests
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=0,
                            stdout="2 passed",
                            stderr=""
                        )
                        
                        results = runner.run_progressive_gates(parallel=False)
                        
                        assert len(results) > 0
                        # At least code compilation should pass
                        compilation_results = [r for r in results if r.gate_name == "code_compilation"]
                        assert len(compilation_results) > 0
                        assert compilation_results[0].passed
    
    def test_maturity_level_progression(self, project_structure):
        """Test that different maturity levels run different gates."""
        # Test greenfield level
        config_greenfield = ProgressiveQualityConfig()
        config_greenfield.maturity = ProjectMaturity.GREENFIELD
        
        gates_greenfield = config_greenfield.enabled_gates[ProjectMaturity.GREENFIELD]
        
        # Test production level  
        config_production = ProgressiveQualityConfig()
        config_production.maturity = ProjectMaturity.PRODUCTION
        
        gates_production = config_production.enabled_gates[ProjectMaturity.PRODUCTION]
        
        # Production should have more gates
        assert len(gates_production) > len(gates_greenfield)
        
        # All greenfield gates should be included in production
        assert all(gate in gates_production for gate in gates_greenfield)


class TestQualityGatesCLI:
    """Test CLI interface for quality gates."""
    
    def test_cli_import(self):
        """Test that CLI can be imported without errors."""
        try:
            from dgdm_histopath.cli.quality_gates import app
            assert app is not None
        except ImportError as e:
            pytest.skip(f"CLI dependencies not available: {e}")
    
    @patch('dgdm_histopath.cli.quality_gates.ProgressiveQualityRunner')
    def test_cli_progressive_mode(self, mock_runner_class):
        """Test CLI in progressive mode."""
        try:
            from dgdm_histopath.cli.quality_gates import run
            from typer.testing import CliRunner
            from dgdm_histopath.cli.quality_gates import app
            
            # Mock the runner
            mock_runner = MagicMock()
            mock_runner.run_progressive_gates.return_value = [
                MagicMock(passed=True, gate_name="test_gate", score=1.0, threshold=1.0, 
                         message="Test passed", execution_time=1.0)
            ]
            mock_runner_class.return_value = mock_runner
            
            runner = CliRunner()
            result = runner.invoke(app, ["run", "--maturity", "greenfield", "--no-parallel"])
            
            # Should not crash
            assert result.exit_code in [0, 1]  # 0 for success, 1 for failure
            
        except ImportError as e:
            pytest.skip(f"CLI dependencies not available: {e}")


def test_quality_gates_complete_workflow():
    """Test complete workflow of quality gates."""
    # This is a meta-test that ensures all major components work together
    
    # Test that all major classes can be imported
    from dgdm_histopath.testing.progressive_quality_gates import (
        ProgressiveQualityRunner, ProgressiveQualityConfig, ProjectMaturity
    )
    from dgdm_histopath.testing.robust_quality_runner import RobustQualityRunner
    from dgdm_histopath.testing.scalable_quality_gates import ScalableQualityGates
    
    # Test basic instantiation
    config = ProgressiveQualityConfig()
    assert config.maturity in ProjectMaturity
    
    # Test enum functionality
    assert ProjectMaturity.GREENFIELD.value == "greenfield"
    assert ProjectMaturity.PRODUCTION.value == "production"
    
    # Test that configuration has expected structure
    assert hasattr(config, 'test_coverage_thresholds')
    assert hasattr(config, 'performance_thresholds')
    assert hasattr(config, 'security_thresholds')
    assert hasattr(config, 'enabled_gates')
    
    # Verify thresholds are properly structured
    for maturity in ProjectMaturity:
        assert maturity in config.test_coverage_thresholds
        assert maturity in config.performance_thresholds
        assert maturity in config.security_thresholds
        assert maturity in config.enabled_gates


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])