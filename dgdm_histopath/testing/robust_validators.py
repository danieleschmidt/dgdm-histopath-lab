"""Robust validators with comprehensive error handling and detailed reporting."""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from contextlib import contextmanager
import subprocess
import threading
import queue
import tempfile
import shutil
import psutil
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from .progressive_quality_gates import QualityGateResult, ProjectMaturity
from ..utils.logging import get_logger
from ..utils.monitoring import monitor_operation


@dataclass
class ValidationContext:
    """Context information for validation."""
    project_root: Path
    temp_dir: Path
    maturity_level: ProjectMaturity
    timeout_seconds: float
    max_memory_mb: float
    enable_gpu: bool
    parallel_workers: int
    validation_id: str
    metadata: Dict[str, Any]


@dataclass 
class RobustValidationResult:
    """Enhanced validation result with comprehensive details."""
    validator_name: str
    passed: bool
    score: float
    threshold: float
    message: str
    details: Dict[str, Any]
    execution_time: float
    memory_peak_mb: float
    cpu_usage_percent: float
    warnings: List[str]
    errors: List[str]
    artifacts: Dict[str, str]  # artifact_name -> file_path
    recovery_attempted: bool
    recovery_successful: bool
    validation_context: ValidationContext


class RobustValidator(ABC):
    """Abstract base class for robust validators."""
    
    def __init__(self, context: ValidationContext):
        self.context = context
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.start_time = 0.0
        self.start_memory = 0.0
        self.peak_memory = 0.0
        self.warnings = []
        self.errors = []
        self.artifacts = {}
        
    @abstractmethod
    def validate(self) -> RobustValidationResult:
        """Perform validation and return result."""
        pass
    
    @contextmanager
    def _resource_monitor(self):
        """Context manager to monitor resource usage."""
        process = psutil.Process()
        self.start_time = time.time()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start memory monitoring thread
        memory_queue = queue.Queue()
        stop_monitoring = threading.Event()
        
        def monitor_memory():
            while not stop_monitoring.is_set():
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_queue.put(current_memory)
                    time.sleep(0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            yield
        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=1.0)
            
            # Calculate peak memory
            max_memory = self.start_memory
            while not memory_queue.empty():
                try:
                    memory = memory_queue.get_nowait()
                    max_memory = max(max_memory, memory)
                except queue.Empty:
                    break
            self.peak_memory = max_memory
    
    def _run_with_timeout(self, func: Callable, timeout: float) -> Any:
        """Run function with timeout and proper error handling."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                self.errors.append(f"Validation timed out after {timeout:.1f}s")
                raise TimeoutError(f"Validation timed out after {timeout:.1f}s")
    
    def _save_artifact(self, name: str, content: Union[str, bytes, Dict], extension: str = ".txt"):
        """Save validation artifact."""
        artifact_dir = self.context.temp_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_path = artifact_dir / f"{self.context.validation_id}_{name}{extension}"
        
        try:
            if isinstance(content, dict):
                with open(artifact_path, 'w') as f:
                    json.dump(content, f, indent=2)
            elif isinstance(content, str):
                with open(artifact_path, 'w') as f:
                    f.write(content)
            elif isinstance(content, bytes):
                with open(artifact_path, 'wb') as f:
                    f.write(content)
            
            self.artifacts[name] = str(artifact_path)
            self.logger.debug(f"Saved artifact '{name}' to {artifact_path}")
            
        except Exception as e:
            self.warnings.append(f"Failed to save artifact '{name}': {e}")
    
    def _attempt_recovery(self, error: Exception) -> bool:
        """Attempt to recover from validation error."""
        self.logger.warning(f"Attempting recovery from error: {error}")
        
        # Basic recovery strategies
        recovery_strategies = [
            self._clear_cache_recovery,
            self._reset_permissions_recovery,
            self._cleanup_temp_files_recovery
        ]
        
        for strategy in recovery_strategies:
            try:
                if strategy():
                    self.logger.info(f"Recovery successful with strategy: {strategy.__name__}")
                    return True
            except Exception as recovery_error:
                self.warnings.append(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
        
        return False
    
    def _clear_cache_recovery(self) -> bool:
        """Clear various caches that might cause issues."""
        cache_dirs = [
            Path.home() / ".cache" / "pip",
            Path(".pytest_cache"),
            Path("__pycache__"),
            Path(".mypy_cache")
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
        
        return True
    
    def _reset_permissions_recovery(self) -> bool:
        """Reset file permissions that might cause issues."""
        try:
            # Reset permissions on common problematic files
            for pattern in ["*.py", "*.json", "*.yaml", "*.yml"]:
                for file_path in self.context.project_root.rglob(pattern):
                    if file_path.is_file():
                        file_path.chmod(0o644)
            return True
        except Exception:
            return False
    
    def _cleanup_temp_files_recovery(self) -> bool:
        """Clean up temporary files that might interfere."""
        temp_patterns = ["*.tmp", "*.temp", ".coverage*", "coverage.xml"]
        
        for pattern in temp_patterns:
            for temp_file in self.context.project_root.rglob(pattern):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                except Exception:
                    continue
        return True


class CodeCompilationValidator(RobustValidator):
    """Robust code compilation validator."""
    
    def validate(self) -> RobustValidationResult:
        with self._resource_monitor():
            try:
                return self._run_with_timeout(self._perform_validation, self.context.timeout_seconds)
            except TimeoutError:
                return self._create_timeout_result()
            except Exception as e:
                return self._handle_validation_error(e)
    
    def _perform_validation(self) -> RobustValidationResult:
        compilation_errors = []
        compilation_warnings = []
        files_checked = 0
        
        # Find all Python files
        python_files = list(self.context.project_root.glob("**/*.py"))
        
        for py_file in python_files:
            if any(skip in str(py_file) for skip in ['.venv', '__pycache__', '.git', 'build', 'dist']):
                continue
                
            files_checked += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for encoding issues
                if not content.isascii():
                    self.warnings.append(f"{py_file}: Contains non-ASCII characters")
                
                # Compile the code
                compile(content, str(py_file), 'exec')
                
                # Check for common issues
                self._check_code_quality(py_file, content, compilation_warnings)
                
            except SyntaxError as e:
                compilation_errors.append({
                    'file': str(py_file),
                    'line': e.lineno,
                    'column': e.offset,
                    'message': e.msg,
                    'text': e.text
                })
            except UnicodeDecodeError as e:
                compilation_errors.append({
                    'file': str(py_file),
                    'line': None,
                    'column': None,
                    'message': f"Encoding error: {e}",
                    'text': None
                })
            except Exception as e:
                self.warnings.append(f"{py_file}: Unexpected error during compilation check: {e}")
        
        # Save detailed results
        self._save_artifact("compilation_errors", compilation_errors, ".json")
        self._save_artifact("compilation_warnings", compilation_warnings, ".json")
        
        # Create result
        passed = len(compilation_errors) == 0
        score = files_checked - len(compilation_errors)
        
        return RobustValidationResult(
            validator_name="code_compilation",
            passed=passed,
            score=score,
            threshold=files_checked,
            message=f"Compilation: {len(compilation_errors)} errors in {files_checked} files",
            details={
                'files_checked': files_checked,
                'compilation_errors': compilation_errors,
                'compilation_warnings': compilation_warnings,
                'error_summary': self._summarize_errors(compilation_errors)
            },
            execution_time=time.time() - self.start_time,
            memory_peak_mb=self.peak_memory,
            cpu_usage_percent=psutil.cpu_percent(),
            warnings=self.warnings.copy(),
            errors=self.errors.copy(),
            artifacts=self.artifacts.copy(),
            recovery_attempted=False,
            recovery_successful=False,
            validation_context=self.context
        )
    
    def _check_code_quality(self, file_path: Path, content: str, warnings_list: List[Dict]):
        """Check for code quality issues."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for overly long lines
            if len(line) > 120:
                warnings_list.append({
                    'file': str(file_path),
                    'line': i,
                    'type': 'line_length',
                    'message': f"Line too long ({len(line)} chars)"
                })
            
            # Check for suspicious patterns
            if 'eval(' in line:
                warnings_list.append({
                    'file': str(file_path),
                    'line': i,
                    'type': 'security',
                    'message': "Use of eval() detected"
                })
            
            # Check for TODO/FIXME comments
            if any(keyword in line.upper() for keyword in ['TODO', 'FIXME', 'XXX', 'HACK']):
                warnings_list.append({
                    'file': str(file_path),
                    'line': i,
                    'type': 'maintenance',
                    'message': f"Maintenance comment: {line.strip()}"
                })
    
    def _summarize_errors(self, errors: List[Dict]) -> Dict[str, Any]:
        """Summarize compilation errors for quick analysis."""
        if not errors:
            return {}
        
        error_types = {}
        files_with_errors = set()
        
        for error in errors:
            error_type = error.get('message', '').split(':')[0] if ':' in error.get('message', '') else 'unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1
            files_with_errors.add(error['file'])
        
        return {
            'total_errors': len(errors),
            'files_affected': len(files_with_errors),
            'error_types': error_types,
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }
    
    def _create_timeout_result(self) -> RobustValidationResult:
        """Create result for timeout scenario."""
        return RobustValidationResult(
            validator_name="code_compilation",
            passed=False,
            score=0.0,
            threshold=1.0,
            message=f"Code compilation check timed out after {self.context.timeout_seconds}s",
            details={'timeout': True},
            execution_time=self.context.timeout_seconds,
            memory_peak_mb=self.peak_memory,
            cpu_usage_percent=psutil.cpu_percent(),
            warnings=self.warnings.copy(),
            errors=self.errors.copy(),
            artifacts=self.artifacts.copy(),
            recovery_attempted=False,
            recovery_successful=False,
            validation_context=self.context
        )
    
    def _handle_validation_error(self, error: Exception) -> RobustValidationResult:
        """Handle validation errors with recovery attempts."""
        recovery_attempted = False
        recovery_successful = False
        
        # Attempt recovery for certain error types
        if isinstance(error, (PermissionError, OSError, IOError)):
            recovery_attempted = True
            recovery_successful = self._attempt_recovery(error)
            
            if recovery_successful:
                try:
                    return self._run_with_timeout(self._perform_validation, self.context.timeout_seconds)
                except Exception as retry_error:
                    self.errors.append(f"Retry after recovery failed: {retry_error}")
        
        self.errors.append(f"Validation failed: {error}")
        self._save_artifact("error_traceback", traceback.format_exc(), ".txt")
        
        return RobustValidationResult(
            validator_name="code_compilation",
            passed=False,
            score=0.0,
            threshold=1.0,
            message=f"Code compilation validation failed: {str(error)[:100]}",
            details={'error': str(error), 'traceback': traceback.format_exc()},
            execution_time=time.time() - self.start_time,
            memory_peak_mb=self.peak_memory,
            cpu_usage_percent=psutil.cpu_percent(),
            warnings=self.warnings.copy(),
            errors=self.errors.copy(),
            artifacts=self.artifacts.copy(),
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful,
            validation_context=self.context
        )


class ModelValidationValidator(RobustValidator):
    """Robust model validation with comprehensive testing."""
    
    def validate(self) -> RobustValidationResult:
        with self._resource_monitor():
            try:
                return self._run_with_timeout(self._perform_validation, self.context.timeout_seconds)
            except TimeoutError:
                return self._create_timeout_result("Model validation timed out")
            except Exception as e:
                return self._handle_validation_error(e, "model_validation")
    
    def _perform_validation(self) -> RobustValidationResult:
        validation_results = {}
        
        try:
            # Import model with proper error handling
            validation_results['import_test'] = self._test_model_import()
            validation_results['instantiation_test'] = self._test_model_instantiation()
            validation_results['forward_pass_test'] = self._test_forward_pass()
            validation_results['device_compatibility_test'] = self._test_device_compatibility()
            validation_results['memory_efficiency_test'] = self._test_memory_efficiency()
            validation_results['serialization_test'] = self._test_model_serialization()
            
        except Exception as e:
            self.errors.append(f"Model validation failed: {e}")
            self._save_artifact("validation_error", traceback.format_exc(), ".txt")
        
        # Analyze results
        total_tests = len(validation_results)
        passed_tests = len([r for r in validation_results.values() if r.get('passed', False)])
        
        # Save detailed results
        self._save_artifact("model_validation_results", validation_results, ".json")
        
        passed = passed_tests == total_tests
        
        return RobustValidationResult(
            validator_name="model_validation",
            passed=passed,
            score=passed_tests,
            threshold=total_tests,
            message=f"Model validation: {passed_tests}/{total_tests} tests passed",
            details=validation_results,
            execution_time=time.time() - self.start_time,
            memory_peak_mb=self.peak_memory,
            cpu_usage_percent=psutil.cpu_percent(),
            warnings=self.warnings.copy(),
            errors=self.errors.copy(),
            artifacts=self.artifacts.copy(),
            recovery_attempted=False,
            recovery_successful=False,
            validation_context=self.context
        )
    
    def _test_model_import(self) -> Dict[str, Any]:
        """Test model import."""
        try:
            from dgdm_histopath.models.dgdm_model import DGDMModel
            return {'passed': True, 'message': 'Model import successful'}
        except ImportError as e:
            self.errors.append(f"Model import failed: {e}")
            return {'passed': False, 'message': f'Import failed: {e}'}
        except Exception as e:
            self.warnings.append(f"Unexpected error during model import: {e}")
            return {'passed': False, 'message': f'Unexpected error: {e}'}
    
    def _test_model_instantiation(self) -> Dict[str, Any]:
        """Test model instantiation with various configurations."""
        try:
            from dgdm_histopath.models.dgdm_model import DGDMModel
            
            # Test different configurations
            configs = [
                {'node_features': 64, 'hidden_dims': [128, 64], 'num_classes': 2},
                {'node_features': 128, 'hidden_dims': [256, 128], 'num_classes': 5},
                {'node_features': 256, 'hidden_dims': [512, 256, 128], 'num_classes': 10}
            ]
            
            instantiation_results = []
            
            for i, config in enumerate(configs):
                try:
                    model = DGDMModel(**config)
                    param_count = sum(p.numel() for p in model.parameters())
                    instantiation_results.append({
                        'config_id': i,
                        'config': config,
                        'parameter_count': param_count,
                        'success': True
                    })
                except Exception as e:
                    instantiation_results.append({
                        'config_id': i,
                        'config': config,
                        'error': str(e),
                        'success': False
                    })
            
            successful_instantiations = len([r for r in instantiation_results if r['success']])
            
            return {
                'passed': successful_instantiations == len(configs),
                'message': f'{successful_instantiations}/{len(configs)} configurations successful',
                'details': instantiation_results
            }
            
        except Exception as e:
            return {'passed': False, 'message': f'Instantiation test failed: {e}'}
    
    def _test_forward_pass(self) -> Dict[str, Any]:
        """Test forward pass with various input sizes."""
        try:
            from dgdm_histopath.models.dgdm_model import DGDMModel
            from torch_geometric.data import Data
            
            model = DGDMModel(node_features=128, hidden_dims=[256, 128], num_classes=5)
            model.eval()
            
            # Test different graph sizes
            test_cases = [
                {'num_nodes': 10, 'num_edges': 20},
                {'num_nodes': 50, 'num_edges': 100},
                {'num_nodes': 100, 'num_edges': 200}
            ]
            
            forward_results = []
            
            for case in test_cases:
                try:
                    x = torch.randn(case['num_nodes'], 128)
                    edge_index = torch.randint(0, case['num_nodes'], (2, case['num_edges']))
                    data = Data(x=x, edge_index=edge_index, num_nodes=case['num_nodes'])
                    
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = model(data, mode="inference")
                    inference_time = time.time() - start_time
                    
                    # Validate outputs
                    assert isinstance(outputs, dict)
                    assert "classification_logits" in outputs
                    assert outputs["classification_logits"].shape[1] == 5
                    
                    forward_results.append({
                        'case': case,
                        'inference_time': inference_time,
                        'output_shape': list(outputs["classification_logits"].shape),
                        'success': True
                    })
                    
                except Exception as e:
                    forward_results.append({
                        'case': case,
                        'error': str(e),
                        'success': False
                    })
            
            successful_passes = len([r for r in forward_results if r['success']])
            
            return {
                'passed': successful_passes == len(test_cases),
                'message': f'{successful_passes}/{len(test_cases)} forward passes successful',
                'details': forward_results
            }
            
        except Exception as e:
            return {'passed': False, 'message': f'Forward pass test failed: {e}'}
    
    def _test_device_compatibility(self) -> Dict[str, Any]:
        """Test device compatibility (CPU/GPU)."""
        device_results = {}
        
        # Test CPU
        try:
            from dgdm_histopath.models.dgdm_model import DGDMModel
            from torch_geometric.data import Data
            
            model = DGDMModel(node_features=64, hidden_dims=[128, 64], num_classes=2)
            model = model.to('cpu')
            
            x = torch.randn(20, 64)
            edge_index = torch.randint(0, 20, (2, 40))
            data = Data(x=x, edge_index=edge_index, num_nodes=20)
            
            with torch.no_grad():
                outputs = model(data, mode="inference")
            
            device_results['cpu'] = {'passed': True, 'message': 'CPU compatibility confirmed'}
            
        except Exception as e:
            device_results['cpu'] = {'passed': False, 'message': f'CPU test failed: {e}'}
        
        # Test GPU if available
        if torch.cuda.is_available() and self.context.enable_gpu:
            try:
                model = model.to('cuda')
                data = data.to('cuda')
                
                with torch.no_grad():
                    outputs = model(data, mode="inference")
                
                device_results['gpu'] = {'passed': True, 'message': 'GPU compatibility confirmed'}
                
            except Exception as e:
                device_results['gpu'] = {'passed': False, 'message': f'GPU test failed: {e}'}
        else:
            device_results['gpu'] = {'passed': True, 'message': 'GPU test skipped (not available or disabled)'}
        
        all_passed = all(result['passed'] for result in device_results.values())
        
        return {
            'passed': all_passed,
            'message': f'Device compatibility: {len(device_results)} devices tested',
            'details': device_results
        }
    
    def _test_memory_efficiency(self) -> Dict[str, Any]:
        """Test memory efficiency and cleanup."""
        try:
            import gc
            
            # Get initial memory
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Create and delete multiple models
            memory_measurements = []
            
            for i in range(3):
                from dgdm_histopath.models.dgdm_model import DGDMModel
                
                model = DGDMModel(node_features=128, hidden_dims=[256, 128], num_classes=5)
                
                if torch.cuda.is_available() and self.context.enable_gpu:
                    model = model.to('cuda')
                    current_memory = torch.cuda.memory_allocated()
                else:
                    current_memory = 0
                
                memory_measurements.append(current_memory - initial_memory)
                
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Check for memory leaks
            final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_leak = (final_memory - initial_memory) / (1024 * 1024)  # MB
            
            return {
                'passed': memory_leak < 100,  # Less than 100MB leak is acceptable
                'message': f'Memory efficiency test: {memory_leak:.1f}MB potential leak',
                'details': {
                    'memory_measurements': memory_measurements,
                    'memory_leak_mb': memory_leak,
                    'initial_memory': initial_memory,
                    'final_memory': final_memory
                }
            }
            
        except Exception as e:
            return {'passed': False, 'message': f'Memory efficiency test failed: {e}'}
    
    def _test_model_serialization(self) -> Dict[str, Any]:
        """Test model serialization and deserialization."""
        try:
            from dgdm_histopath.models.dgdm_model import DGDMModel
            
            # Create model
            model = DGDMModel(node_features=128, hidden_dims=[256, 128], num_classes=5)
            
            # Save to temporary file
            temp_file = self.context.temp_dir / "test_model.pth"
            torch.save(model.state_dict(), temp_file)
            
            # Load model
            new_model = DGDMModel(node_features=128, hidden_dims=[256, 128], num_classes=5)
            new_model.load_state_dict(torch.load(temp_file, map_location='cpu'))
            
            # Verify parameters match
            original_params = list(model.parameters())
            loaded_params = list(new_model.parameters())
            
            params_match = all(
                torch.allclose(p1, p2, rtol=1e-5)
                for p1, p2 in zip(original_params, loaded_params)
            )
            
            return {
                'passed': params_match,
                'message': f'Serialization test: {"successful" if params_match else "failed"}',
                'details': {
                    'serialized_file_size': temp_file.stat().st_size,
                    'parameter_count': len(original_params),
                    'parameters_match': params_match
                }
            }
            
        except Exception as e:
            return {'passed': False, 'message': f'Serialization test failed: {e}'}
    
    def _create_timeout_result(self, message: str) -> RobustValidationResult:
        """Create timeout result."""
        return RobustValidationResult(
            validator_name="model_validation",
            passed=False,
            score=0.0,
            threshold=1.0,
            message=message,
            details={'timeout': True},
            execution_time=self.context.timeout_seconds,
            memory_peak_mb=self.peak_memory,
            cpu_usage_percent=psutil.cpu_percent(),
            warnings=self.warnings.copy(),
            errors=self.errors.copy(),
            artifacts=self.artifacts.copy(),
            recovery_attempted=False,
            recovery_successful=False,
            validation_context=self.context
        )
    
    def _handle_validation_error(self, error: Exception, validator_name: str) -> RobustValidationResult:
        """Handle validation errors."""
        self.errors.append(f"Validation failed: {error}")
        self._save_artifact("error_traceback", traceback.format_exc(), ".txt")
        
        return RobustValidationResult(
            validator_name=validator_name,
            passed=False,
            score=0.0,
            threshold=1.0,
            message=f"Validation failed: {str(error)[:100]}",
            details={'error': str(error), 'traceback': traceback.format_exc()},
            execution_time=time.time() - self.start_time,
            memory_peak_mb=self.peak_memory,
            cpu_usage_percent=psutil.cpu_percent(),
            warnings=self.warnings.copy(),
            errors=self.errors.copy(),
            artifacts=self.artifacts.copy(),
            recovery_attempted=False,
            recovery_successful=False,
            validation_context=self.context
        )