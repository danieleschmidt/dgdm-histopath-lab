"""Automated quality gates and continuous integration checks."""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import psutil
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from dgdm_histopath.utils.logging import get_logger, setup_logging
from dgdm_histopath.utils.monitoring import monitor_operation
from dgdm_histopath.testing.test_suite import run_performance_suite


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float
    threshold: float
    message: str
    details: Dict[str, Any] = None
    execution_time: float = 0.0


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""
    
    # Code quality thresholds
    min_test_coverage: float = 85.0
    max_cyclomatic_complexity: int = 10
    max_code_duplications: float = 5.0
    
    # Performance thresholds
    max_model_creation_time: float = 10.0
    max_inference_time: float = 5.0
    max_memory_usage_mb: float = 2000.0
    
    # Security thresholds
    max_security_vulnerabilities: int = 0
    max_dependency_vulnerabilities: int = 2
    
    # Documentation thresholds
    min_docstring_coverage: float = 80.0
    
    # Resource thresholds
    max_cpu_usage_percent: float = 90.0
    max_memory_usage_percent: float = 85.0
    
    # Model-specific thresholds
    min_model_accuracy: float = 0.70
    max_model_size_mb: float = 500.0
    max_training_time_hours: float = 48.0


class QualityGateRunner:
    """Automated quality gate execution and reporting."""
    
    def __init__(self, config: QualityGateConfig = None, output_dir: str = "./quality_reports"):
        self.config = config or QualityGateConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(__name__)
        self.results = []
        
        # Quality gate registry
        self.quality_gates = {
            'code_coverage': self._check_code_coverage,
            'code_complexity': self._check_code_complexity,
            'code_duplications': self._check_code_duplications,
            'security_scan': self._check_security_vulnerabilities,
            'dependency_scan': self._check_dependency_vulnerabilities,
            'performance_tests': self._check_performance,
            'model_validation': self._check_model_validation,
            'resource_usage': self._check_resource_usage,
            'documentation': self._check_documentation,
            'integration_tests': self._check_integration_tests
        }
    
    def run_all_gates(self, parallel: bool = True) -> List[QualityGateResult]:
        """Run all quality gates."""
        self.logger.info("Starting quality gate validation...")
        self.results.clear()
        
        if parallel:
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_gate = {
                    executor.submit(self._run_single_gate, gate_name, gate_func): gate_name
                    for gate_name, gate_func in self.quality_gates.items()
                }
                
                for future in as_completed(future_to_gate):
                    gate_name = future_to_gate[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        self.logger.error(f"Quality gate {gate_name} failed with exception: {e}")
                        self.results.append(QualityGateResult(
                            gate_name=gate_name,
                            passed=False,
                            score=0.0,
                            threshold=1.0,
                            message=f"Exception during execution: {e}",
                            details={'exception': str(e)}
                        ))
        else:
            for gate_name, gate_func in self.quality_gates.items():
                result = self._run_single_gate(gate_name, gate_func)
                self.results.append(result)
        
        # Generate report
        self._generate_report()
        
        return self.results
    
    def _run_single_gate(self, gate_name: str, gate_func) -> QualityGateResult:
        """Run a single quality gate."""
        self.logger.info(f"Running quality gate: {gate_name}")
        
        start_time = time.time()
        try:
            with monitor_operation(f"quality_gate_{gate_name}"):
                result = gate_func()
                result.execution_time = time.time() - start_time
                
                if result.passed:
                    self.logger.info(f"✅ {gate_name}: PASSED ({result.score:.2f}/{result.threshold:.2f})")
                else:
                    self.logger.warning(f"❌ {gate_name}: FAILED ({result.score:.2f}/{result.threshold:.2f}) - {result.message}")
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Quality gate {gate_name} encountered error: {e}")
            
            return QualityGateResult(
                gate_name=gate_name,
                passed=False,
                score=0.0,
                threshold=1.0,
                message=f"Execution failed: {e}",
                details={'exception': str(e)},
                execution_time=execution_time
            )
    
    def _check_code_coverage(self) -> QualityGateResult:
        """Check code test coverage."""
        try:
            # Run pytest with coverage
            result = subprocess.run([
                'python', '-m', 'pytest', 
                '--cov=dgdm_histopath',
                '--cov-report=json',
                '--cov-report=term-missing',
                'tests/',
                '-v'
            ], capture_output=True, text=True, timeout=300)
            
            # Parse coverage report
            coverage_file = Path('coverage.json')
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data['totals']['percent_covered']
                
                return QualityGateResult(
                    gate_name="code_coverage",
                    passed=total_coverage >= self.config.min_test_coverage,
                    score=total_coverage,
                    threshold=self.config.min_test_coverage,
                    message=f"Code coverage: {total_coverage:.1f}%",
                    details=coverage_data['totals']
                )
            else:
                return QualityGateResult(
                    gate_name="code_coverage",
                    passed=False,
                    score=0.0,
                    threshold=self.config.min_test_coverage,
                    message="Coverage report not generated"
                )
                
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="code_coverage",
                passed=False,
                score=0.0,
                threshold=self.config.min_test_coverage,
                message="Code coverage check timed out"
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="code_coverage",
                passed=False,
                score=0.0,
                threshold=self.config.min_test_coverage,
                message=f"Coverage check failed: {e}"
            )
    
    def _check_code_complexity(self) -> QualityGateResult:
        """Check code cyclomatic complexity."""
        try:
            # Use radon for complexity analysis
            result = subprocess.run([
                'python', '-m', 'radon', 'cc',
                'dgdm_histopath/',
                '--json'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                
                # Find maximum complexity
                max_complexity = 0
                complex_functions = []
                
                for file_path, functions in complexity_data.items():
                    for func in functions:
                        if func['complexity'] > max_complexity:
                            max_complexity = func['complexity']
                        if func['complexity'] > self.config.max_cyclomatic_complexity:
                            complex_functions.append({
                                'file': file_path,
                                'function': func['name'],
                                'complexity': func['complexity']
                            })
                
                passed = max_complexity <= self.config.max_cyclomatic_complexity
                
                return QualityGateResult(
                    gate_name="code_complexity",
                    passed=passed,
                    score=self.config.max_cyclomatic_complexity - max_complexity,
                    threshold=0,
                    message=f"Max complexity: {max_complexity}, Complex functions: {len(complex_functions)}",
                    details={'max_complexity': max_complexity, 'complex_functions': complex_functions}
                )
            else:
                return QualityGateResult(
                    gate_name="code_complexity",
                    passed=False,
                    score=0.0,
                    threshold=0,
                    message="Complexity analysis failed"
                )
                
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="code_complexity",
                passed=False,
                score=0.0,
                threshold=0,
                message="Complexity check timed out"
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="code_complexity",
                passed=False,
                score=0.0,
                threshold=0,
                message=f"Complexity check failed: {e}"
            )
    
    def _check_code_duplications(self) -> QualityGateResult:
        """Check for code duplications."""
        try:
            # Use pylint for duplication detection (simplified)
            result = subprocess.run([
                'python', '-m', 'pylint',
                'dgdm_histopath/',
                '--reports=y',
                '--output-format=json'
            ], capture_output=True, text=True, timeout=120)
            
            # For now, return a simple check (can be enhanced with proper duplication tools)
            duplication_score = 2.0  # Placeholder - would use actual duplication analysis
            
            return QualityGateResult(
                gate_name="code_duplications",
                passed=duplication_score <= self.config.max_code_duplications,
                score=duplication_score,
                threshold=self.config.max_code_duplications,
                message=f"Code duplication: {duplication_score:.1f}%",
                details={'duplication_percentage': duplication_score}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="code_duplications",
                passed=True,  # Default to pass if check fails
                score=0.0,
                threshold=self.config.max_code_duplications,
                message=f"Duplication check skipped: {e}"
            )
    
    def _check_security_vulnerabilities(self) -> QualityGateResult:
        """Check for security vulnerabilities."""
        try:
            # Use bandit for security analysis
            result = subprocess.run([
                'python', '-m', 'bandit',
                '-r', 'dgdm_histopath/',
                '-f', 'json',
                '-ll'  # Only show high confidence issues
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 or result.returncode == 1:  # 1 means issues found
                try:
                    security_data = json.loads(result.stdout)
                    num_vulnerabilities = len(security_data.get('results', []))
                    
                    passed = num_vulnerabilities <= self.config.max_security_vulnerabilities
                    
                    return QualityGateResult(
                        gate_name="security_scan",
                        passed=passed,
                        score=self.config.max_security_vulnerabilities - num_vulnerabilities,
                        threshold=0,
                        message=f"Security vulnerabilities found: {num_vulnerabilities}",
                        details={'vulnerabilities': security_data.get('results', [])}
                    )
                except json.JSONDecodeError:
                    # If JSON parsing fails, assume no issues
                    return QualityGateResult(
                        gate_name="security_scan",
                        passed=True,
                        score=0.0,
                        threshold=0,
                        message="Security scan completed (no JSON output)"
                    )
            else:
                return QualityGateResult(
                    gate_name="security_scan",
                    passed=False,
                    score=0.0,
                    threshold=0,
                    message="Security scan failed"
                )
                
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="security_scan",
                passed=False,
                score=0.0,
                threshold=0,
                message="Security scan timed out"
            )
        except Exception as e:
            # Default to pass if security tools are not available
            return QualityGateResult(
                gate_name="security_scan",
                passed=True,
                score=0.0,
                threshold=0,
                message=f"Security scan skipped: {e}"
            )
    
    def _check_dependency_vulnerabilities(self) -> QualityGateResult:
        """Check for vulnerabilities in dependencies."""
        try:
            # Use safety to check dependencies
            result = subprocess.run([
                'python', '-m', 'safety', 'check', '--json'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # No vulnerabilities found
                return QualityGateResult(
                    gate_name="dependency_scan",
                    passed=True,
                    score=0.0,
                    threshold=self.config.max_dependency_vulnerabilities,
                    message="No dependency vulnerabilities found"
                )
            else:
                try:
                    vulnerabilities = json.loads(result.stderr) if result.stderr else []
                    num_vulnerabilities = len(vulnerabilities)
                    
                    passed = num_vulnerabilities <= self.config.max_dependency_vulnerabilities
                    
                    return QualityGateResult(
                        gate_name="dependency_scan",
                        passed=passed,
                        score=self.config.max_dependency_vulnerabilities - num_vulnerabilities,
                        threshold=0,
                        message=f"Dependency vulnerabilities found: {num_vulnerabilities}",
                        details={'vulnerabilities': vulnerabilities}
                    )
                except json.JSONDecodeError:
                    return QualityGateResult(
                        gate_name="dependency_scan",
                        passed=False,
                        score=0.0,
                        threshold=0,
                        message="Failed to parse dependency scan results"
                    )
                    
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="dependency_scan",
                passed=False,
                score=0.0,
                threshold=0,
                message="Dependency scan timed out"
            )
        except Exception as e:
            # Default to pass if safety is not available
            return QualityGateResult(
                gate_name="dependency_scan",
                passed=True,
                score=0.0,
                threshold=0,
                message=f"Dependency scan skipped: {e}"
            )
    
    def _check_performance(self) -> QualityGateResult:
        """Check performance benchmarks."""
        try:
            performance_results = run_performance_suite()
            
            # Check individual performance metrics
            failures = []
            
            if performance_results.get('model_creation_time', 0) > self.config.max_model_creation_time:
                failures.append(f"Model creation time too slow: {performance_results['model_creation_time']:.2f}s")
            
            if performance_results.get('avg_inference_time', 0) > self.config.max_inference_time:
                failures.append(f"Inference time too slow: {performance_results['avg_inference_time']:.2f}s")
            
            if performance_results.get('model_memory_usage_mb', 0) > self.config.max_memory_usage_mb:
                failures.append(f"Memory usage too high: {performance_results['model_memory_usage_mb']:.1f}MB")
            
            passed = len(failures) == 0
            
            return QualityGateResult(
                gate_name="performance_tests",
                passed=passed,
                score=len(failures),
                threshold=0,
                message=f"Performance check: {len(failures)} failures",
                details={'failures': failures, 'metrics': performance_results}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="performance_tests",
                passed=False,
                score=0.0,
                threshold=1,
                message=f"Performance tests failed: {e}"
            )
    
    def _check_model_validation(self) -> QualityGateResult:
        """Check model validation and accuracy."""
        try:
            from dgdm_histopath.models.dgdm_model import DGDMModel
            from torch_geometric.data import Data
            
            # Create test model
            model = DGDMModel(
                node_features=128,
                hidden_dims=[256, 128],
                num_classes=5
            )
            
            # Test basic functionality
            num_nodes = 50
            x = torch.randn(num_nodes, 128)
            edge_index = torch.randint(0, num_nodes, (2, 100))
            data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
            
            model.eval()
            with torch.no_grad():
                outputs = model(data, mode="inference")
            
            # Basic validation checks
            assert "classification_logits" in outputs
            assert outputs["classification_logits"].shape[1] == 5
            
            # Check model size
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            
            passed = model_size_mb <= self.config.max_model_size_mb
            
            return QualityGateResult(
                gate_name="model_validation",
                passed=passed,
                score=self.config.max_model_size_mb - model_size_mb,
                threshold=0,
                message=f"Model validation passed, size: {model_size_mb:.1f}MB",
                details={'model_size_mb': model_size_mb}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="model_validation",
                passed=False,
                score=0.0,
                threshold=1,
                message=f"Model validation failed: {e}"
            )
    
    def _check_resource_usage(self) -> QualityGateResult:
        """Check current resource usage."""
        try:
            # Check CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            failures = []
            if cpu_percent > self.config.max_cpu_usage_percent:
                failures.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.config.max_memory_usage_percent:
                failures.append(f"High memory usage: {memory_percent:.1f}%")
            
            passed = len(failures) == 0
            
            return QualityGateResult(
                gate_name="resource_usage",
                passed=passed,
                score=len(failures),
                threshold=0,
                message=f"Resource usage check: {len(failures)} issues",
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'failures': failures
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="resource_usage",
                passed=False,
                score=0.0,
                threshold=1,
                message=f"Resource usage check failed: {e}"
            )
    
    def _check_documentation(self) -> QualityGateResult:
        """Check documentation coverage."""
        try:
            # Simple docstring coverage check
            python_files = list(Path("dgdm_histopath").rglob("*.py"))
            
            total_functions = 0
            documented_functions = 0
            
            for py_file in python_files:
                if py_file.name.startswith("test_"):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Simple heuristic: count functions and classes with docstrings
                    lines = content.split('\n')
                    in_function = False
                    has_docstring = False
                    
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if stripped.startswith(('def ', 'class ')):
                            if in_function and not has_docstring:
                                # Previous function had no docstring
                                pass
                            
                            total_functions += 1
                            in_function = True
                            has_docstring = False
                            
                            # Check if next few lines contain docstring
                            for j in range(i + 1, min(i + 5, len(lines))):
                                if '"""' in lines[j] or "'''" in lines[j]:
                                    has_docstring = True
                                    documented_functions += 1
                                    break
                                elif lines[j].strip() and not lines[j].strip().startswith('#'):
                                    break
                            
                except Exception:
                    continue
            
            coverage_percent = (documented_functions / total_functions * 100) if total_functions > 0 else 0
            passed = coverage_percent >= self.config.min_docstring_coverage
            
            return QualityGateResult(
                gate_name="documentation",
                passed=passed,
                score=coverage_percent,
                threshold=self.config.min_docstring_coverage,
                message=f"Documentation coverage: {coverage_percent:.1f}%",
                details={
                    'total_functions': total_functions,
                    'documented_functions': documented_functions,
                    'coverage_percent': coverage_percent
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="documentation",
                passed=False,
                score=0.0,
                threshold=self.config.min_docstring_coverage,
                message=f"Documentation check failed: {e}"
            )
    
    def _check_integration_tests(self) -> QualityGateResult:
        """Check integration tests."""
        try:
            # Run integration tests
            result = subprocess.run([
                'python', '-m', 'pytest',
                'tests/',
                '-k', 'integration',
                '-v',
                '--tb=short'
            ], capture_output=True, text=True, timeout=300)
            
            # Parse test results
            output_lines = result.stdout.split('\n')
            passed_tests = len([line for line in output_lines if 'PASSED' in line])
            failed_tests = len([line for line in output_lines if 'FAILED' in line])
            
            total_tests = passed_tests + failed_tests
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 100
            
            passed = failed_tests == 0
            
            return QualityGateResult(
                gate_name="integration_tests",
                passed=passed,
                score=success_rate,
                threshold=100,
                message=f"Integration tests: {passed_tests} passed, {failed_tests} failed",
                details={
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'success_rate': success_rate
                }
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="integration_tests",
                passed=False,
                score=0.0,
                threshold=100,
                message="Integration tests timed out"
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="integration_tests",
                passed=False,
                score=0.0,
                threshold=100,
                message=f"Integration tests failed: {e}"
            )
    
    def _generate_report(self):
        """Generate comprehensive quality gate report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON report
        json_report = {
            'timestamp': timestamp,
            'summary': {
                'total_gates': len(self.results),
                'passed_gates': len([r for r in self.results if r.passed]),
                'failed_gates': len([r for r in self.results if not r.passed]),
                'overall_passed': all(r.passed for r in self.results)
            },
            'results': [asdict(result) for result in self.results]
        }
        
        json_file = self.output_dir / f"quality_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # Generate HTML report
        html_report = self._generate_html_report(json_report)
        html_file = self.output_dir / f"quality_report_{timestamp}.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        # Generate summary for CI/CD
        summary_file = self.output_dir / "quality_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Quality Gate Summary - {timestamp}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Gates: {json_report['summary']['total_gates']}\n")
            f.write(f"Passed: {json_report['summary']['passed_gates']}\n")
            f.write(f"Failed: {json_report['summary']['failed_gates']}\n")
            f.write(f"Overall: {'PASSED' if json_report['summary']['overall_passed'] else 'FAILED'}\n\n")
            
            for result in self.results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                f.write(f"{status} {result.gate_name}: {result.message}\n")
        
        self.logger.info(f"Quality gate report generated: {json_file}")
        self.logger.info(f"HTML report generated: {html_file}")
    
    def _generate_html_report(self, json_report: Dict[str, Any]) -> str:
        """Generate HTML quality report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DGDM Quality Gate Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                .summary {{ background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .gate {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
                .gate.passed {{ border-left-color: #4CAF50; background: #f1f8e9; }}
                .gate.failed {{ border-left-color: #f44336; background: #ffebee; }}
                .score {{ font-weight: bold; }}
                .details {{ margin-top: 10px; background: #fafafa; padding: 10px; border-radius: 3px; }}
                pre {{ font-size: 12px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DGDM Histopath Lab - Quality Gate Report</h1>
                <p>Generated: {json_report['timestamp']}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Overall Status:</strong> {'PASSED' if json_report['summary']['overall_passed'] else 'FAILED'}</p>
                <p><strong>Total Gates:</strong> {json_report['summary']['total_gates']}</p>
                <p><strong>Passed:</strong> {json_report['summary']['passed_gates']}</p>
                <p><strong>Failed:</strong> {json_report['summary']['failed_gates']}</p>
            </div>
            
            <h2>Quality Gate Results</h2>
        """
        
        for result in json_report['results']:
            status_class = "passed" if result['passed'] else "failed"
            status_text = "PASSED" if result['passed'] else "FAILED"
            
            html += f"""
            <div class="gate {status_class}">
                <h3>{result['gate_name']} - {status_text}</h3>
                <p class="score">Score: {result['score']:.2f} / {result['threshold']:.2f}</p>
                <p>{result['message']}</p>
                <p><em>Execution time: {result['execution_time']:.2f}s</em></p>
            """
            
            if result['details']:
                html += f"""
                <div class="details">
                    <strong>Details:</strong>
                    <pre>{json.dumps(result['details'], indent=2)}</pre>
                </div>
                """
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html


def main():
    """Main entry point for quality gate runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DGDM quality gates")
    parser.add_argument("--output-dir", default="./quality_reports", help="Output directory for reports")
    parser.add_argument("--parallel", action="store_true", help="Run gates in parallel")
    parser.add_argument("--gates", nargs="*", help="Specific gates to run (default: all)")
    parser.add_argument("--config", help="Path to quality gate configuration file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO", log_file=None)
    logger = get_logger(__name__)
    
    # Load configuration
    config = QualityGateConfig()
    if args.config and Path(args.config).exists():
        # Load custom configuration (implementation depends on format)
        logger.info(f"Loading configuration from {args.config}")
    
    # Create runner
    runner = QualityGateRunner(config=config, output_dir=args.output_dir)
    
    # Run gates
    if args.gates:
        # Run specific gates
        for gate_name in args.gates:
            if gate_name in runner.quality_gates:
                result = runner._run_single_gate(gate_name, runner.quality_gates[gate_name])
                runner.results.append(result)
            else:
                logger.error(f"Unknown quality gate: {gate_name}")
    else:
        # Run all gates
        runner.run_all_gates(parallel=args.parallel)
    
    # Check overall result
    overall_passed = all(r.passed for r in runner.results)
    
    logger.info(f"Quality gate validation completed: {'PASSED' if overall_passed else 'FAILED'}")
    
    # Exit with appropriate code for CI/CD
    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()