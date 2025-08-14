"""Robust quality gate runner with comprehensive error handling and recovery."""

import os
import sys
import json
import time
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import logging

try:
    import torch
except ImportError:
    torch = None

from .robust_validators import (
    RobustValidator, 
    ValidationContext, 
    RobustValidationResult,
    CodeCompilationValidator,
    ModelValidationValidator
)
from .progressive_quality_gates import ProjectMaturity, ProgressiveQualityConfig
from ..utils.logging import get_logger, setup_logging


class RobustQualityRunner:
    """Robust quality gate runner with advanced error handling and recovery."""
    
    def __init__(
        self, 
        config: ProgressiveQualityConfig = None,
        project_root: Path = None,
        output_dir: str = "./quality_reports",
        enable_recovery: bool = True,
        max_recovery_attempts: int = 2
    ):
        self.config = config or ProgressiveQualityConfig()
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_recovery = enable_recovery
        self.max_recovery_attempts = max_recovery_attempts
        
        self.logger = get_logger(__name__)
        self.validation_id = str(uuid.uuid4())[:8]
        
        # Create temporary directory for this validation run
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"dgdm_quality_{self.validation_id}_"))
        
        # Initialize validation context
        self.context = ValidationContext(
            project_root=self.project_root,
            temp_dir=self.temp_dir,
            maturity_level=self.config.maturity,
            timeout_seconds=self._calculate_timeout(),
            max_memory_mb=self._calculate_max_memory(),
            enable_gpu=torch.cuda.is_available() if torch is not None else False,
            parallel_workers=self._calculate_parallel_workers(),
            validation_id=self.validation_id,
            metadata=self._collect_metadata()
        )
        
        # Validator registry with robust implementations
        self.validator_registry: Dict[str, Type[RobustValidator]] = {
            'code_compilation': CodeCompilationValidator,
            'model_validation': ModelValidationValidator,
            # Additional validators can be added here
        }
        
        # Results storage
        self.validation_results: List[RobustValidationResult] = []
        self.execution_summary = {}
        
        self.logger.info(f"Robust Quality Runner initialized - ID: {self.validation_id}")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Maturity level: {self.config.maturity.value}")
        self.logger.info(f"Temp directory: {self.temp_dir}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def _calculate_timeout(self) -> float:
        """Calculate timeout based on maturity level."""
        base_timeout = 300  # 5 minutes
        multipliers = {
            ProjectMaturity.GREENFIELD: 0.5,
            ProjectMaturity.DEVELOPMENT: 1.0,
            ProjectMaturity.STAGING: 2.0,
            ProjectMaturity.PRODUCTION: 3.0
        }
        return base_timeout * multipliers.get(self.config.maturity, 1.0)
    
    def _calculate_max_memory(self) -> float:
        """Calculate max memory based on system and maturity level."""
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            
            # Use percentage based on maturity
            memory_percentages = {
                ProjectMaturity.GREENFIELD: 0.3,
                ProjectMaturity.DEVELOPMENT: 0.5,
                ProjectMaturity.STAGING: 0.7,
                ProjectMaturity.PRODUCTION: 0.8
            }
            
            return available_memory * memory_percentages.get(self.config.maturity, 0.5)
        except ImportError:
            return 2048.0  # Default 2GB
    
    def _calculate_parallel_workers(self) -> int:
        """Calculate optimal number of parallel workers."""
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            
            # Conservative approach based on maturity
            worker_ratios = {
                ProjectMaturity.GREENFIELD: 0.5,
                ProjectMaturity.DEVELOPMENT: 0.75,
                ProjectMaturity.STAGING: 1.0,
                ProjectMaturity.PRODUCTION: 1.0
            }
            
            workers = max(1, int(cpu_count * worker_ratios.get(self.config.maturity, 0.75)))
            return min(workers, 8)  # Cap at 8 workers
        except ImportError:
            return 2  # Conservative default
    
    def _collect_metadata(self) -> Dict[str, Any]:
        """Collect metadata about the validation environment."""
        metadata = {
            'timestamp': time.time(),
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': str(self.project_root),
        }
        
        try:
            import psutil
            metadata.update({
                'cpu_count': psutil.cpu_count(),
                'memory_total_mb': psutil.virtual_memory().total / (1024 * 1024),
                'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
            })
        except ImportError:
            pass
        
        if torch is not None:
            metadata.update({
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            })
        
        return metadata
    
    def run_validation(
        self, 
        validators: Optional[List[str]] = None,
        parallel: bool = True
    ) -> List[RobustValidationResult]:
        """Run robust validation with comprehensive error handling."""
        
        start_time = time.time()
        self.logger.info("Starting robust quality validation...")
        
        try:
            # Determine which validators to run
            if validators is None:
                enabled_gates = self.config.enabled_gates.get(self.config.maturity, [])
                validators_to_run = [v for v in enabled_gates if v in self.validator_registry]
            else:
                validators_to_run = [v for v in validators if v in self.validator_registry]
            
            if not validators_to_run:
                self.logger.warning("No validators to run")
                return []
            
            self.logger.info(f"Running {len(validators_to_run)} validators: {validators_to_run}")
            
            # Execute validators
            if parallel and len(validators_to_run) > 1:
                results = self._run_parallel_validation(validators_to_run)
            else:
                results = self._run_sequential_validation(validators_to_run)
            
            self.validation_results = results
            
            # Generate execution summary
            self._generate_execution_summary(time.time() - start_time)
            
            # Save comprehensive report
            self._save_comprehensive_report()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Critical error during validation: {e}")
            self.logger.error(traceback.format_exc())
            
            # Create error result
            error_result = self._create_critical_error_result(e, time.time() - start_time)
            self.validation_results = [error_result]
            return [error_result]
        finally:
            self.logger.info(f"Validation completed in {time.time() - start_time:.2f}s")
    
    def _run_parallel_validation(self, validators_to_run: List[str]) -> List[RobustValidationResult]:
        """Run validators in parallel with proper error isolation."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.context.parallel_workers) as executor:
            # Submit all validation tasks
            future_to_validator = {
                executor.submit(self._run_single_validator, validator_name): validator_name
                for validator_name in validators_to_run
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_validator, timeout=self.context.timeout_seconds * 2):
                validator_name = future_to_validator[future]
                
                try:
                    result = future.result(timeout=self.context.timeout_seconds)
                    results.append(result)
                    
                    status = "✅" if result.passed else "❌" 
                    self.logger.info(f"{status} {validator_name} completed")
                    
                except Exception as e:
                    self.logger.error(f"Validator {validator_name} failed: {e}")
                    error_result = self._create_validator_error_result(validator_name, e)
                    results.append(error_result)
        
        return results
    
    def _run_sequential_validation(self, validators_to_run: List[str]) -> List[RobustValidationResult]:
        """Run validators sequentially."""
        results = []
        
        for validator_name in validators_to_run:
            try:
                result = self._run_single_validator(validator_name)
                results.append(result)
                
                status = "✅" if result.passed else "❌"
                self.logger.info(f"{status} {validator_name} completed")
                
            except Exception as e:
                self.logger.error(f"Validator {validator_name} failed: {e}")
                error_result = self._create_validator_error_result(validator_name, e)
                results.append(error_result)
        
        return results
    
    def _run_single_validator(self, validator_name: str) -> RobustValidationResult:
        """Run a single validator with recovery attempts."""
        validator_class = self.validator_registry[validator_name]
        
        for attempt in range(self.max_recovery_attempts + 1):
            try:
                # Create fresh validator instance for each attempt
                validator = validator_class(self.context)
                
                self.logger.debug(f"Running {validator_name} (attempt {attempt + 1})")
                result = validator.validate()
                
                # If successful or recovery not enabled, return result
                if result.passed or not self.enable_recovery or attempt == self.max_recovery_attempts:
                    return result
                
                # If failed and recovery enabled, try recovery
                if not result.passed and self.enable_recovery and attempt < self.max_recovery_attempts:
                    self.logger.info(f"Attempting recovery for {validator_name} (attempt {attempt + 2})")
                    continue
                
            except Exception as e:
                if attempt == self.max_recovery_attempts:
                    raise
                self.logger.warning(f"Validator {validator_name} attempt {attempt + 1} failed: {e}")
        
        # Should not reach here, but safety fallback
        return self._create_validator_error_result(validator_name, Exception("Max recovery attempts exceeded"))
    
    def _create_validator_error_result(
        self, 
        validator_name: str, 
        error: Exception
    ) -> RobustValidationResult:
        """Create error result for validator failure."""
        return RobustValidationResult(
            validator_name=validator_name,
            passed=False,
            score=0.0,
            threshold=1.0,
            message=f"Validator execution failed: {str(error)[:100]}",
            details={
                'error': str(error),
                'traceback': traceback.format_exc(),
                'critical_failure': True
            },
            execution_time=0.0,
            memory_peak_mb=0.0,
            cpu_usage_percent=0.0,
            warnings=[f"Critical validator failure: {validator_name}"],
            errors=[str(error)],
            artifacts={},
            recovery_attempted=False,
            recovery_successful=False,
            validation_context=self.context
        )
    
    def _create_critical_error_result(
        self, 
        error: Exception, 
        execution_time: float
    ) -> RobustValidationResult:
        """Create result for critical validation runner failure."""
        return RobustValidationResult(
            validator_name="critical_failure",
            passed=False,
            score=0.0,
            threshold=1.0,
            message=f"Critical validation failure: {str(error)[:100]}",
            details={
                'error': str(error),
                'traceback': traceback.format_exc(),
                'critical_failure': True,
                'runner_failure': True
            },
            execution_time=execution_time,
            memory_peak_mb=0.0,
            cpu_usage_percent=0.0,
            warnings=["Critical validation runner failure"],
            errors=[str(error)],
            artifacts={},
            recovery_attempted=False,
            recovery_successful=False,
            validation_context=self.context
        )
    
    def _generate_execution_summary(self, total_execution_time: float):
        """Generate comprehensive execution summary."""
        passed_count = len([r for r in self.validation_results if r.passed])
        failed_count = len([r for r in self.validation_results if not r.passed])
        total_warnings = sum(len(r.warnings) for r in self.validation_results)
        total_errors = sum(len(r.errors) for r in self.validation_results)
        
        self.execution_summary = {
            'validation_id': self.validation_id,
            'timestamp': time.time(),
            'maturity_level': self.config.maturity.value,
            'total_execution_time': total_execution_time,
            'validators_run': len(self.validation_results),
            'validators_passed': passed_count,
            'validators_failed': failed_count,
            'overall_success': failed_count == 0,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'recovery_attempts': sum(1 for r in self.validation_results if r.recovery_attempted),
            'successful_recoveries': sum(1 for r in self.validation_results if r.recovery_successful),
            'peak_memory_usage_mb': max((r.memory_peak_mb for r in self.validation_results), default=0),
            'average_execution_time': total_execution_time / len(self.validation_results) if self.validation_results else 0,
            'context': {
                'project_root': str(self.context.project_root),
                'temp_dir': str(self.context.temp_dir),
                'timeout_seconds': self.context.timeout_seconds,
                'max_memory_mb': self.context.max_memory_mb,
                'parallel_workers': self.context.parallel_workers,
                'metadata': self.context.metadata
            }
        }
    
    def _save_comprehensive_report(self):
        """Save comprehensive validation report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Comprehensive report
        comprehensive_report = {
            'report_version': '2.0',
            'generated_at': timestamp,
            'validation_id': self.validation_id,
            'summary': self.execution_summary,
            'validator_results': [
                {
                    'validator_name': result.validator_name,
                    'passed': result.passed,
                    'score': result.score,
                    'threshold': result.threshold,
                    'message': result.message,
                    'execution_time': result.execution_time,
                    'memory_peak_mb': result.memory_peak_mb,
                    'cpu_usage_percent': result.cpu_usage_percent,
                    'warnings_count': len(result.warnings),
                    'errors_count': len(result.errors),
                    'artifacts_count': len(result.artifacts),
                    'recovery_attempted': result.recovery_attempted,
                    'recovery_successful': result.recovery_successful,
                    'details': result.details,
                    'warnings': result.warnings,
                    'errors': result.errors,
                    'artifacts': result.artifacts
                }
                for result in self.validation_results
            ]
        }
        
        # Save JSON report
        report_file = self.output_dir / f"robust_quality_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        # Save human-readable summary
        summary_file = self.output_dir / f"robust_quality_summary_{timestamp}.txt"
        self._save_human_readable_summary(summary_file, comprehensive_report)
        
        # Save CSV for analysis
        csv_file = self.output_dir / f"robust_quality_data_{timestamp}.csv"
        self._save_csv_report(csv_file)
        
        self.logger.info(f"Comprehensive report saved: {report_file}")
        self.logger.info(f"Human-readable summary: {summary_file}")
        self.logger.info(f"CSV data file: {csv_file}")
    
    def _save_human_readable_summary(self, summary_file: Path, report: Dict[str, Any]):
        """Save human-readable summary."""
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DGDM ROBUST QUALITY VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Validation ID: {report['validation_id']}\n")
            f.write(f"Generated: {report['generated_at']}\n")
            f.write(f"Project Maturity: {report['summary']['maturity_level'].upper()}\n")
            f.write(f"Overall Success: {'✅ PASSED' if report['summary']['overall_success'] else '❌ FAILED'}\n\n")
            
            f.write("EXECUTION SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Execution Time: {report['summary']['total_execution_time']:.2f}s\n")
            f.write(f"Validators Run: {report['summary']['validators_run']}\n")
            f.write(f"Passed: {report['summary']['validators_passed']}\n")
            f.write(f"Failed: {report['summary']['validators_failed']}\n")
            f.write(f"Warnings: {report['summary']['total_warnings']}\n")
            f.write(f"Errors: {report['summary']['total_errors']}\n")
            f.write(f"Recovery Attempts: {report['summary']['recovery_attempts']}\n")
            f.write(f"Successful Recoveries: {report['summary']['successful_recoveries']}\n")
            f.write(f"Peak Memory Usage: {report['summary']['peak_memory_usage_mb']:.1f} MB\n\n")
            
            f.write("VALIDATOR RESULTS\n")
            f.write("-" * 40 + "\n")
            
            for result in report['validator_results']:
                status = "✅ PASSED" if result['passed'] else "❌ FAILED"
                f.write(f"{status} {result['validator_name']}\n")
                f.write(f"  Message: {result['message']}\n")
                f.write(f"  Score: {result['score']:.2f}/{result['threshold']:.2f}\n")
                f.write(f"  Execution Time: {result['execution_time']:.2f}s\n")
                f.write(f"  Memory Peak: {result['memory_peak_mb']:.1f} MB\n")
                f.write(f"  Warnings: {result['warnings_count']}\n")
                f.write(f"  Errors: {result['errors_count']}\n")
                if result['recovery_attempted']:
                    recovery_status = "✅" if result['recovery_successful'] else "❌"
                    f.write(f"  Recovery: {recovery_status} {'Successful' if result['recovery_successful'] else 'Failed'}\n")
                f.write("\n")
    
    def _save_csv_report(self, csv_file: Path):
        """Save CSV report for data analysis."""
        try:
            import csv
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'validator_name', 'passed', 'score', 'threshold', 'execution_time',
                    'memory_peak_mb', 'cpu_usage_percent', 'warnings_count', 'errors_count',
                    'recovery_attempted', 'recovery_successful'
                ])
                
                # Data rows
                for result in self.validation_results:
                    writer.writerow([
                        result.validator_name,
                        result.passed,
                        result.score,
                        result.threshold,
                        result.execution_time,
                        result.memory_peak_mb,
                        result.cpu_usage_percent,
                        len(result.warnings),
                        len(result.errors),
                        result.recovery_attempted,
                        result.recovery_successful
                    ])
                    
        except ImportError:
            self.logger.warning("CSV module not available, skipping CSV report")
    
    def cleanup(self):
        """Clean up temporary resources."""
        try:
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
                self.logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temporary directory: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return self.execution_summary.copy()
    
    def has_failures(self) -> bool:
        """Check if any validators failed."""
        return any(not result.passed for result in self.validation_results)
    
    def get_failed_validators(self) -> List[str]:
        """Get list of failed validator names."""
        return [result.validator_name for result in self.validation_results if not result.passed]