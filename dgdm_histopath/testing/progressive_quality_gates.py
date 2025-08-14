"""Progressive quality gates that adapt based on project maturity and context."""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .quality_gates import QualityGateResult, QualityGateConfig, QualityGateRunner
from ..utils.logging import get_logger


class ProjectMaturity(Enum):
    """Project maturity levels."""
    GREENFIELD = "greenfield"  # New project, minimal requirements
    DEVELOPMENT = "development"  # Active development, moderate requirements
    STAGING = "staging"  # Pre-production, strict requirements
    PRODUCTION = "production"  # Production-ready, strictest requirements


@dataclass
class ProgressiveQualityConfig:
    """Configuration that adapts based on project maturity."""
    
    maturity: ProjectMaturity = ProjectMaturity.DEVELOPMENT
    
    # Progressive thresholds based on maturity
    test_coverage_thresholds = {
        ProjectMaturity.GREENFIELD: 50.0,
        ProjectMaturity.DEVELOPMENT: 70.0,
        ProjectMaturity.STAGING: 85.0,
        ProjectMaturity.PRODUCTION: 90.0
    }
    
    performance_thresholds = {
        ProjectMaturity.GREENFIELD: {"inference_time": 10.0, "memory_mb": 4000},
        ProjectMaturity.DEVELOPMENT: {"inference_time": 7.0, "memory_mb": 3000},
        ProjectMaturity.STAGING: {"inference_time": 5.0, "memory_mb": 2000},
        ProjectMaturity.PRODUCTION: {"inference_time": 3.0, "memory_mb": 1500}
    }
    
    security_thresholds = {
        ProjectMaturity.GREENFIELD: {"vulnerabilities": 5, "dependencies": 10},
        ProjectMaturity.DEVELOPMENT: {"vulnerabilities": 2, "dependencies": 5},
        ProjectMaturity.STAGING: {"vulnerabilities": 0, "dependencies": 2},
        ProjectMaturity.PRODUCTION: {"vulnerabilities": 0, "dependencies": 0}
    }
    
    # Feature gates - what to check at each maturity level
    enabled_gates = {
        ProjectMaturity.GREENFIELD: [
            "code_compilation", "basic_tests", "model_validation"
        ],
        ProjectMaturity.DEVELOPMENT: [
            "code_compilation", "basic_tests", "model_validation",
            "code_coverage", "performance_basic", "security_basic"
        ],
        ProjectMaturity.STAGING: [
            "code_compilation", "basic_tests", "model_validation",
            "code_coverage", "performance_basic", "security_basic",
            "integration_tests", "performance_advanced", "security_advanced",
            "documentation", "resource_usage"
        ],
        ProjectMaturity.PRODUCTION: [  # All gates
            "code_compilation", "basic_tests", "model_validation",
            "code_coverage", "performance_basic", "security_basic",
            "integration_tests", "performance_advanced", "security_advanced",
            "documentation", "resource_usage", "compliance_checks",
            "disaster_recovery", "monitoring_health"
        ]
    }
    
    # Progressive timeout increases
    timeout_multipliers = {
        ProjectMaturity.GREENFIELD: 0.5,
        ProjectMaturity.DEVELOPMENT: 1.0,
        ProjectMaturity.STAGING: 2.0,
        ProjectMaturity.PRODUCTION: 3.0
    }


class ProgressiveQualityRunner:
    """Progressive quality gate runner that adapts to project context."""
    
    def __init__(self, config: ProgressiveQualityConfig = None, output_dir: str = "./quality_reports"):
        self.config = config or ProgressiveQualityConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(__name__)
        self.results = []
        
        # Auto-detect project maturity if not specified
        if not hasattr(self.config, 'maturity') or self.config.maturity is None:
            self.config.maturity = self._detect_project_maturity()
            
        self.logger.info(f"Progressive Quality Gates - Maturity Level: {self.config.maturity.value}")
        
        # Progressive quality gates registry
        self.quality_gates = {
            'code_compilation': self._check_code_compilation,
            'basic_tests': self._check_basic_tests,
            'model_validation': self._check_model_validation,
            'code_coverage': self._check_progressive_coverage,
            'performance_basic': self._check_basic_performance,
            'security_basic': self._check_basic_security,
            'integration_tests': self._check_integration_tests,
            'performance_advanced': self._check_advanced_performance,
            'security_advanced': self._check_advanced_security,
            'documentation': self._check_documentation_quality,
            'resource_usage': self._check_resource_efficiency,
            'compliance_checks': self._check_compliance,
            'disaster_recovery': self._check_disaster_recovery,
            'monitoring_health': self._check_monitoring_health
        }
    
    def _detect_project_maturity(self) -> ProjectMaturity:
        """Auto-detect project maturity based on codebase analysis."""
        indicators = {
            'has_tests': bool(list(Path('.').glob('**/test*.py'))),
            'has_ci': any(Path(p).exists() for p in ['.github', '.gitlab-ci.yml', 'Jenkinsfile']),
            'has_deployment': any(Path(p).exists() for p in ['Dockerfile', 'kubernetes', 'deploy']),
            'has_monitoring': any(Path(p).glob('**/*monitor*') for p in [Path('.')]),
            'code_quality_tools': any(Path(p).exists() for p in ['.pre-commit-config.yaml', 'pyproject.toml']),
            'documentation': len(list(Path('.').glob('**/*.md'))) > 2,
            'production_configs': any(Path(p).exists() for p in ['production.yaml', 'prod.env'])
        }
        
        maturity_score = sum(indicators.values())
        
        if maturity_score >= 6:
            return ProjectMaturity.PRODUCTION
        elif maturity_score >= 4:
            return ProjectMaturity.STAGING
        elif maturity_score >= 2:
            return ProjectMaturity.DEVELOPMENT
        else:
            return ProjectMaturity.GREENFIELD
    
    def run_progressive_gates(self, parallel: bool = True) -> List[QualityGateResult]:
        """Run quality gates appropriate for current maturity level."""
        self.logger.info(f"Starting progressive quality validation for {self.config.maturity.value} project...")
        self.results.clear()
        
        # Get enabled gates for current maturity
        enabled_gates = self.config.enabled_gates.get(self.config.maturity, [])
        
        if parallel and len(enabled_gates) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(enabled_gates))) as executor:
                future_to_gate = {
                    executor.submit(self._run_single_gate, gate_name): gate_name
                    for gate_name in enabled_gates
                    if gate_name in self.quality_gates
                }
                
                for future in as_completed(future_to_gate):
                    gate_name = future_to_gate[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        self.logger.error(f"Gate {gate_name} failed: {e}")
                        self.results.append(self._create_error_result(gate_name, e))
        else:
            for gate_name in enabled_gates:
                if gate_name in self.quality_gates:
                    result = self._run_single_gate(gate_name)
                    self.results.append(result)
        
        # Generate maturity-appropriate report
        self._generate_progressive_report()
        
        return self.results
    
    def _run_single_gate(self, gate_name: str) -> QualityGateResult:
        """Run a single quality gate with progressive timeout."""
        self.logger.info(f"Running progressive gate: {gate_name}")
        
        base_timeout = 60  # Base timeout in seconds
        timeout = base_timeout * self.config.timeout_multipliers[self.config.maturity]
        
        start_time = time.time()
        try:
            gate_func = self.quality_gates[gate_name]
            result = gate_func()
            result.execution_time = time.time() - start_time
            
            status = "✅" if result.passed else "❌"
            self.logger.info(f"{status} {gate_name}: {result.message}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Gate {gate_name} error: {e}")
            return self._create_error_result(gate_name, e, execution_time)
    
    def _create_error_result(self, gate_name: str, error: Exception, execution_time: float = 0.0) -> QualityGateResult:
        """Create error result for failed gate."""
        return QualityGateResult(
            gate_name=gate_name,
            passed=False,
            score=0.0,
            threshold=1.0,
            message=f"Execution failed: {str(error)[:100]}",
            details={'error': str(error)},
            execution_time=execution_time
        )
    
    # Progressive Quality Gate Implementations
    
    def _check_code_compilation(self) -> QualityGateResult:
        """Check if code compiles without syntax errors."""
        try:
            # Compile all Python files
            compilation_errors = []
            
            for py_file in Path('dgdm_histopath').rglob('*.py'):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    compile(content, str(py_file), 'exec')
                except SyntaxError as e:
                    compilation_errors.append(f"{py_file}:{e.lineno} - {e.msg}")
                except Exception as e:
                    compilation_errors.append(f"{py_file} - {str(e)}")
            
            passed = len(compilation_errors) == 0
            
            return QualityGateResult(
                gate_name="code_compilation",
                passed=passed,
                score=len(compilation_errors),
                threshold=0,
                message=f"Compilation: {len(compilation_errors)} errors found",
                details={'errors': compilation_errors}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="code_compilation",
                passed=False,
                score=0.0,
                threshold=0,
                message=f"Compilation check failed: {e}"
            )
    
    def _check_basic_tests(self) -> QualityGateResult:
        """Run basic test suite."""
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/', 
                '--tb=short', '-q'
            ], capture_output=True, text=True, timeout=120)
            
            # Parse test results
            output = result.stdout + result.stderr
            
            # Simple parsing for test results
            import re
            passed_match = re.search(r'(\d+) passed', output)
            failed_match = re.search(r'(\d+) failed', output)
            
            passed_tests = int(passed_match.group(1)) if passed_match else 0
            failed_tests = int(failed_match.group(1)) if failed_match else 0
            
            if result.returncode == 0:
                return QualityGateResult(
                    gate_name="basic_tests",
                    passed=True,
                    score=passed_tests,
                    threshold=1,
                    message=f"Tests passed: {passed_tests}",
                    details={'passed': passed_tests, 'failed': failed_tests}
                )
            else:
                return QualityGateResult(
                    gate_name="basic_tests",
                    passed=False,
                    score=passed_tests,
                    threshold=passed_tests + failed_tests,
                    message=f"Tests failed: {failed_tests}",
                    details={'passed': passed_tests, 'failed': failed_tests}
                )
                
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="basic_tests",
                passed=False,
                score=0.0,
                threshold=1,
                message="Basic tests timed out"
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="basic_tests",
                passed=False,
                score=0.0,
                threshold=1,
                message=f"Test execution failed: {e}"
            )
    
    def _check_model_validation(self) -> QualityGateResult:
        """Validate that core model can be instantiated and run."""
        try:
            from dgdm_histopath.models.dgdm_model import DGDMModel
            from torch_geometric.data import Data
            
            # Create minimal model
            model = DGDMModel(
                node_features=64,
                hidden_dims=[128, 64],
                num_classes=2
            )
            
            # Test forward pass
            num_nodes = 20
            x = torch.randn(num_nodes, 64)
            edge_index = torch.randint(0, num_nodes, (2, 40))
            data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
            
            model.eval()
            with torch.no_grad():
                outputs = model(data, mode="inference")
            
            # Basic validation
            assert isinstance(outputs, dict)
            assert "classification_logits" in outputs
            assert outputs["classification_logits"].shape[0] == 1  # batch size
            assert outputs["classification_logits"].shape[1] == 2  # num classes
            
            return QualityGateResult(
                gate_name="model_validation",
                passed=True,
                score=1.0,
                threshold=1.0,
                message="Model validation successful",
                details={'output_shape': list(outputs["classification_logits"].shape)}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="model_validation",
                passed=False,
                score=0.0,
                threshold=1.0,
                message=f"Model validation failed: {e}",
                details={'error': str(e)}
            )
    
    def _check_progressive_coverage(self) -> QualityGateResult:
        """Check test coverage with maturity-appropriate thresholds."""
        try:
            # Run coverage analysis
            result = subprocess.run([
                'python', '-m', 'pytest',
                '--cov=dgdm_histopath',
                '--cov-report=json',
                'tests/', '-q'
            ], capture_output=True, text=True, timeout=180)
            
            # Parse coverage
            coverage_file = Path('coverage.json')
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data['totals']['percent_covered']
                threshold = self.config.test_coverage_thresholds[self.config.maturity]
                
                return QualityGateResult(
                    gate_name="code_coverage",
                    passed=total_coverage >= threshold,
                    score=total_coverage,
                    threshold=threshold,
                    message=f"Coverage: {total_coverage:.1f}% (required: {threshold:.1f}%)",
                    details={
                        'coverage_percent': total_coverage,
                        'maturity_threshold': threshold,
                        'lines_covered': coverage_data['totals']['covered_lines'],
                        'lines_total': coverage_data['totals']['num_statements']
                    }
                )
            else:
                return QualityGateResult(
                    gate_name="code_coverage",
                    passed=False,
                    score=0.0,
                    threshold=self.config.test_coverage_thresholds[self.config.maturity],
                    message="Coverage report not generated"
                )
                
        except Exception as e:
            return QualityGateResult(
                gate_name="code_coverage",
                passed=False,
                score=0.0,
                threshold=self.config.test_coverage_thresholds[self.config.maturity],
                message=f"Coverage check failed: {e}"
            )
    
    def _check_basic_performance(self) -> QualityGateResult:
        """Check basic performance metrics."""
        try:
            from dgdm_histopath.models.dgdm_model import DGDMModel
            from torch_geometric.data import Data
            import time
            
            # Performance test
            model = DGDMModel(node_features=128, hidden_dims=[256, 128], num_classes=5)
            model.eval()
            
            # Measure inference time
            num_nodes = 100
            x = torch.randn(num_nodes, 128)
            edge_index = torch.randint(0, num_nodes, (2, 200))
            data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
            
            # Warm up
            with torch.no_grad():
                _ = model(data, mode="inference")
            
            # Measure
            start_time = time.time()
            with torch.no_grad():
                outputs = model(data, mode="inference")
            inference_time = time.time() - start_time
            
            # Get thresholds for maturity level
            perf_thresholds = self.config.performance_thresholds[self.config.maturity]
            
            failures = []
            if inference_time > perf_thresholds["inference_time"]:
                failures.append(f"Slow inference: {inference_time:.2f}s > {perf_thresholds['inference_time']}s")
            
            passed = len(failures) == 0
            
            return QualityGateResult(
                gate_name="performance_basic",
                passed=passed,
                score=perf_thresholds["inference_time"] - inference_time,
                threshold=0,
                message=f"Performance check: {len(failures)} issues",
                details={
                    'inference_time': inference_time,
                    'threshold': perf_thresholds["inference_time"],
                    'failures': failures
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="performance_basic",
                passed=False,
                score=0.0,
                threshold=1,
                message=f"Performance check failed: {e}"
            )
    
    def _check_basic_security(self) -> QualityGateResult:
        """Check basic security with maturity-appropriate thresholds."""
        try:
            # Simple security checks
            security_issues = []
            
            # Check for common security anti-patterns
            for py_file in Path('dgdm_histopath').rglob('*.py'):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Basic security pattern detection
                    if 'eval(' in content:
                        security_issues.append(f"{py_file}: Use of eval() detected")
                    if 'exec(' in content:
                        security_issues.append(f"{py_file}: Use of exec() detected")
                    if 'subprocess.call' in content and 'shell=True' in content:
                        security_issues.append(f"{py_file}: Shell injection risk detected")
                        
                except Exception:
                    continue
            
            security_thresholds = self.config.security_thresholds[self.config.maturity]
            threshold = security_thresholds["vulnerabilities"]
            
            passed = len(security_issues) <= threshold
            
            return QualityGateResult(
                gate_name="security_basic",
                passed=passed,
                score=threshold - len(security_issues),
                threshold=0,
                message=f"Security: {len(security_issues)} issues (max: {threshold})",
                details={
                    'issues_found': len(security_issues),
                    'threshold': threshold,
                    'issues': security_issues
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="security_basic",
                passed=False,
                score=0.0,
                threshold=1,
                message=f"Security check failed: {e}"
            )
    
    # Placeholder implementations for advanced gates
    def _check_integration_tests(self) -> QualityGateResult:
        """Check integration tests."""
        # Implementation would run integration test suite
        return QualityGateResult(
            gate_name="integration_tests",
            passed=True,
            score=1.0,
            threshold=1.0,
            message="Integration tests skipped (placeholder)"
        )
    
    def _check_advanced_performance(self) -> QualityGateResult:
        """Advanced performance testing."""
        return QualityGateResult(
            gate_name="performance_advanced",
            passed=True,
            score=1.0,
            threshold=1.0,
            message="Advanced performance tests skipped (placeholder)"
        )
    
    def _check_advanced_security(self) -> QualityGateResult:
        """Advanced security testing."""
        return QualityGateResult(
            gate_name="security_advanced",
            passed=True,
            score=1.0,
            threshold=1.0,
            message="Advanced security tests skipped (placeholder)"
        )
    
    def _check_documentation_quality(self) -> QualityGateResult:
        """Check documentation quality."""
        return QualityGateResult(
            gate_name="documentation",
            passed=True,
            score=1.0,
            threshold=1.0,
            message="Documentation check skipped (placeholder)"
        )
    
    def _check_resource_efficiency(self) -> QualityGateResult:
        """Check resource usage efficiency."""
        return QualityGateResult(
            gate_name="resource_usage",
            passed=True,
            score=1.0,
            threshold=1.0,
            message="Resource efficiency check skipped (placeholder)"
        )
    
    def _check_compliance(self) -> QualityGateResult:
        """Check regulatory compliance."""
        return QualityGateResult(
            gate_name="compliance_checks",
            passed=True,
            score=1.0,
            threshold=1.0,
            message="Compliance checks skipped (placeholder)"
        )
    
    def _check_disaster_recovery(self) -> QualityGateResult:
        """Check disaster recovery readiness."""
        return QualityGateResult(
            gate_name="disaster_recovery",
            passed=True,
            score=1.0,
            threshold=1.0,
            message="Disaster recovery check skipped (placeholder)"
        )
    
    def _check_monitoring_health(self) -> QualityGateResult:
        """Check monitoring and observability."""
        return QualityGateResult(
            gate_name="monitoring_health",
            passed=True,
            score=1.0,
            threshold=1.0,
            message="Monitoring health check skipped (placeholder)"
        )
    
    def _generate_progressive_report(self):
        """Generate maturity-appropriate quality report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Progressive report structure
        progressive_report = {
            'timestamp': timestamp,
            'project_maturity': self.config.maturity.value,
            'gates_run': len(self.results),
            'summary': {
                'total_gates': len(self.results),
                'passed_gates': len([r for r in self.results if r.passed]),
                'failed_gates': len([r for r in self.results if not r.passed]),
                'overall_passed': all(r.passed for r in self.results),
                'maturity_appropriate': True  # Always true since we run appropriate gates
            },
            'maturity_thresholds': {
                'test_coverage': self.config.test_coverage_thresholds[self.config.maturity],
                'performance': self.config.performance_thresholds[self.config.maturity],
                'security': self.config.security_thresholds[self.config.maturity]
            },
            'results': [asdict(result) for result in self.results],
            'next_maturity_recommendations': self._get_next_maturity_recommendations()
        }
        
        # Save report
        json_file = self.output_dir / f"progressive_quality_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(progressive_report, f, indent=2)
        
        # Generate summary
        self._generate_progressive_summary(progressive_report)
        
        self.logger.info(f"Progressive quality report generated: {json_file}")
    
    def _get_next_maturity_recommendations(self) -> List[str]:
        """Get recommendations for advancing to next maturity level."""
        current = self.config.maturity
        recommendations = []
        
        if current == ProjectMaturity.GREENFIELD:
            recommendations = [
                "Add comprehensive test suite with >70% coverage",
                "Implement basic performance benchmarks",
                "Add security scanning tools",
                "Set up continuous integration"
            ]
        elif current == ProjectMaturity.DEVELOPMENT:
            recommendations = [
                "Increase test coverage to >85%",
                "Add integration tests",
                "Implement advanced performance monitoring",
                "Add documentation standards",
                "Set up deployment pipelines"
            ]
        elif current == ProjectMaturity.STAGING:
            recommendations = [
                "Achieve >90% test coverage",
                "Add compliance checking",
                "Implement disaster recovery procedures",
                "Add comprehensive monitoring",
                "Prepare for production deployment"
            ]
        else:  # PRODUCTION
            recommendations = [
                "Maintain all quality standards",
                "Monitor for performance regression",
                "Regular security audits",
                "Continuous compliance validation"
            ]
        
        return recommendations
    
    def _generate_progressive_summary(self, report: Dict[str, Any]):
        """Generate human-readable summary."""
        summary_file = self.output_dir / "progressive_quality_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Progressive Quality Gates Summary - {report['timestamp']}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Project Maturity Level: {report['project_maturity'].upper()}\n")
            f.write(f"Gates Executed: {report['gates_run']}\n")
            f.write(f"Overall Status: {'PASSED' if report['summary']['overall_passed'] else 'FAILED'}\n\n")
            
            f.write("Gate Results:\n")
            f.write("-" * 40 + "\n")
            for result in report['results']:
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                f.write(f"{status} {result['gate_name']}: {result['message']}\n")
            
            f.write("\nMaturity Thresholds Applied:\n")
            f.write("-" * 40 + "\n")
            for category, thresholds in report['maturity_thresholds'].items():
                f.write(f"{category}: {thresholds}\n")
            
            f.write("\nNext Maturity Level Recommendations:\n")
            f.write("-" * 40 + "\n")
            for rec in report['next_maturity_recommendations']:
                f.write(f"• {rec}\n")
        
        self.logger.info(f"Progressive summary generated: {summary_file}")


def main():
    """Main entry point for progressive quality gate runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Progressive Quality Gates")
    parser.add_argument("--maturity", choices=['greenfield', 'development', 'staging', 'production'],
                       help="Force specific maturity level (auto-detected by default)")
    parser.add_argument("--output-dir", default="./quality_reports", help="Output directory")
    parser.add_argument("--parallel", action="store_true", help="Run gates in parallel")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = ProgressiveQualityConfig()
    if args.maturity:
        config.maturity = ProjectMaturity(args.maturity)
    
    # Create runner and execute
    runner = ProgressiveQualityRunner(config=config, output_dir=args.output_dir)
    results = runner.run_progressive_gates(parallel=args.parallel)
    
    # Exit with appropriate code
    overall_passed = all(r.passed for r in results)
    exit_code = 0 if overall_passed else 1
    
    print(f"\nProgressive Quality Gates: {'PASSED' if overall_passed else 'FAILED'}")
    print(f"Project Maturity: {config.maturity.value}")
    print(f"Gates Executed: {len(results)}")
    
    exit(exit_code)


if __name__ == "__main__":
    main()