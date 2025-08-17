"""
Comprehensive test framework for DGDM Histopath Lab.

This module provides automated testing capabilities including unit tests,
integration tests, performance benchmarks, and security validation.
"""

import unittest
import sys
import time
import json
import logging
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import traceback
import warnings


@dataclass
class TestResult:
    """Comprehensive test result information."""
    test_name: str
    category: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class TestCategory:
    """Test categories for organization."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SMOKE = "smoke"
    REGRESSION = "regression"


class ComprehensiveTestFramework:
    """Advanced test framework with multiple test categories."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.results: List[TestResult] = []
        self.test_functions: Dict[str, List[Callable]] = {
            category: [] for category in [
                TestCategory.UNIT, TestCategory.INTEGRATION, 
                TestCategory.PERFORMANCE, TestCategory.SECURITY,
                TestCategory.SMOKE, TestCategory.REGRESSION
            ]
        }
        self.setup_logging()
    
    def setup_logging(self):
        """Setup test logging."""
        self.logger = logging.getLogger("dgdm_test_framework")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def register_test(self, category: str, name: str = None):
        """Decorator to register test functions."""
        def decorator(func: Callable) -> Callable:
            test_name = name or func.__name__
            self.test_functions[category].append((test_name, func))
            return func
        return decorator
    
    def run_category(self, category: str, parallel: bool = True) -> List[TestResult]:
        """Run all tests in a specific category."""
        category_tests = self.test_functions.get(category, [])
        if not category_tests:
            self.logger.warning(f"No tests found for category: {category}")
            return []
        
        self.logger.info(f"Running {len(category_tests)} tests in category: {category}")
        
        if parallel and len(category_tests) > 1:
            return self._run_tests_parallel(category_tests, category)
        else:
            return self._run_tests_sequential(category_tests, category)
    
    def _run_tests_sequential(self, tests: List[Tuple[str, Callable]], 
                             category: str) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        
        for test_name, test_func in tests:
            result = self._execute_test(test_name, test_func, category)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def _run_tests_parallel(self, tests: List[Tuple[str, Callable]], 
                           category: str) -> List[TestResult]:
        """Run tests in parallel."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._execute_test, test_name, test_func, category): test_name
                for test_name, test_func in tests
            }
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                    self.results.append(result)
                except Exception as e:
                    test_name = futures[future]
                    error_result = TestResult(
                        test_name=test_name,
                        category=category,
                        passed=False,
                        execution_time=0.0,
                        error_message=f"Test execution failed: {str(e)}",
                        traceback=traceback.format_exc()
                    )
                    results.append(error_result)
                    self.results.append(error_result)
        
        return results
    
    def _execute_test(self, test_name: str, test_func: Callable, 
                     category: str) -> TestResult:
        """Execute a single test with comprehensive error handling."""
        start_time = time.time()
        
        try:
            # Suppress warnings during testing unless critical
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                # Execute test function
                test_result = test_func()
                
                execution_time = time.time() - start_time
                
                # Handle different return types
                if isinstance(test_result, bool):
                    passed = test_result
                    metadata = None
                elif isinstance(test_result, dict):
                    passed = test_result.get("passed", True)
                    metadata = test_result
                else:
                    passed = True  # Assume passed if no exception
                    metadata = {"result": test_result}
                
                return TestResult(
                    test_name=test_name,
                    category=category,
                    passed=passed,
                    execution_time=execution_time,
                    metadata=metadata
                )
        
        except AssertionError as e:
            return TestResult(
                test_name=test_name,
                category=category,
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category=category,
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all registered tests across all categories."""
        self.logger.info("Starting comprehensive test suite...")
        start_time = time.time()
        
        category_results = {}
        
        for category in self.test_functions.keys():
            if self.test_functions[category]:
                category_results[category] = self.run_category(category)
        
        total_time = time.time() - start_time
        
        return self._generate_test_report(category_results, total_time)
    
    def _generate_test_report(self, category_results: Dict[str, List[TestResult]], 
                             total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = sum(len(results) for results in category_results.values())
        total_passed = sum(sum(1 for r in results if r.passed) for results in category_results.values())
        
        category_summary = {}
        for category, results in category_results.items():
            if results:
                passed_count = sum(1 for r in results if r.passed)
                category_summary[category] = {
                    "total": len(results),
                    "passed": passed_count,
                    "failed": len(results) - passed_count,
                    "pass_rate": (passed_count / len(results)) * 100 if results else 0,
                    "avg_execution_time": sum(r.execution_time for r in results) / len(results)
                }
        
        # Identify failing tests
        failing_tests = [r for r in self.results if not r.passed]
        
        # Performance analysis
        slowest_tests = sorted(self.results, key=lambda x: x.execution_time, reverse=True)[:5]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_tests - total_passed,
                "pass_rate": (total_passed / total_tests) * 100 if total_tests > 0 else 0
            },
            "category_breakdown": category_summary,
            "failing_tests": [
                {
                    "name": t.test_name,
                    "category": t.category,
                    "error": t.error_message,
                    "execution_time": t.execution_time
                }
                for t in failing_tests
            ],
            "performance_analysis": {
                "slowest_tests": [
                    {
                        "name": t.test_name,
                        "category": t.category,
                        "execution_time": t.execution_time
                    }
                    for t in slowest_tests
                ],
                "avg_test_time": sum(r.execution_time for r in self.results) / len(self.results) if self.results else 0
            },
            "recommendations": self._generate_test_recommendations(category_summary, failing_tests)
        }
    
    def _generate_test_recommendations(self, category_summary: Dict[str, Any], 
                                     failing_tests: List[TestResult]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        overall_pass_rate = sum(cat["pass_rate"] for cat in category_summary.values()) / len(category_summary) if category_summary else 0
        
        if overall_pass_rate < 80:
            recommendations.append("Overall test pass rate is below 80%. Review failing tests.")
        
        for category, stats in category_summary.items():
            if stats["pass_rate"] < 70:
                recommendations.append(f"{category} tests have low pass rate ({stats['pass_rate']:.1f}%). Needs attention.")
        
        if len(failing_tests) > 5:
            recommendations.append(f"{len(failing_tests)} tests are failing. Consider batch fixes.")
        
        # Performance recommendations
        slow_tests = [t for t in self.results if t.execution_time > 10.0]
        if slow_tests:
            recommendations.append(f"{len(slow_tests)} tests are slow (>10s). Consider optimization.")
        
        return recommendations


# Create global test framework instance
test_framework = ComprehensiveTestFramework(Path.cwd())


# Register specific test functions
@test_framework.register_test(TestCategory.SMOKE, "import_test")
def test_basic_imports():
    """Test that core modules can be imported."""
    try:
        import dgdm_histopath
        return {"passed": True, "message": "Core imports successful"}
    except ImportError as e:
        return {"passed": False, "error": str(e)}


@test_framework.register_test(TestCategory.SMOKE, "build_info_test")
def test_build_info():
    """Test build information retrieval."""
    try:
        import dgdm_histopath
        build_info = dgdm_histopath.get_build_info()
        assert "version" in build_info
        assert "features" in build_info
        return {"passed": True, "build_info": build_info}
    except Exception as e:
        return {"passed": False, "error": str(e)}


@test_framework.register_test(TestCategory.UNIT, "dependency_check_test")
def test_dependency_checker():
    """Test dependency checking functionality."""
    try:
        from dgdm_histopath.utils.dependency_check import DependencyChecker
        
        checker = DependencyChecker()
        
        # Test Python version check
        python_ok = checker.check_python_version()
        
        # Test basic dependency check
        numpy_available, numpy_version = checker.check_dependency("sys")  # sys is always available
        
        assert isinstance(python_ok, bool)
        assert numpy_available is True
        
        return {
            "passed": True,
            "python_version_ok": python_ok,
            "test_dependency_available": numpy_available
        }
    except Exception as e:
        return {"passed": False, "error": str(e)}


@test_framework.register_test(TestCategory.UNIT, "error_handling_test")
def test_error_handling():
    """Test error handling utilities."""
    try:
        from dgdm_histopath.utils.enhanced_error_handling import (
            resilient, safe_execution, ErrorSeverity
        )
        
        # Test safe_execution
        def failing_function():
            raise ValueError("Test error")
        
        result = safe_execution(failing_function, default_return="fallback")
        assert result == "fallback"
        
        # Test resilient decorator
        @resilient(max_retries=1, circuit_breaker=False)
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
        
        return {"passed": True, "safe_execution_works": True, "resilient_decorator_works": True}
    except Exception as e:
        return {"passed": False, "error": str(e)}


@test_framework.register_test(TestCategory.SECURITY, "security_basic_test")
def test_basic_security():
    """Test basic security measures."""
    try:
        # Test that no hardcoded secrets are in obvious places
        project_root = Path.cwd()
        
        dangerous_patterns = [
            "password = ",
            "api_key = ",
            "secret = "
        ]
        
        issues_found = []
        
        for py_file in project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern in dangerous_patterns:
                        if pattern in content.lower():
                            issues_found.append(f"{py_file}: {pattern}")
            except:
                continue
        
        return {
            "passed": len(issues_found) == 0,
            "security_issues": issues_found,
            "files_checked": len(list(project_root.rglob("*.py")))
        }
    except Exception as e:
        return {"passed": False, "error": str(e)}


@test_framework.register_test(TestCategory.PERFORMANCE, "import_performance_test")
def test_import_performance():
    """Test import performance."""
    try:
        start_time = time.time()
        import dgdm_histopath
        import_time = time.time() - start_time
        
        # Import should be fast (< 2 seconds)
        performance_ok = import_time < 2.0
        
        return {
            "passed": performance_ok,
            "import_time": import_time,
            "threshold": 2.0,
            "performance_ok": performance_ok
        }
    except Exception as e:
        return {"passed": False, "error": str(e)}


@test_framework.register_test(TestCategory.INTEGRATION, "quality_gates_integration_test")
def test_quality_gates_integration():
    """Test integration with quality gates."""
    try:
        from dgdm_histopath.testing.autonomous_quality_gates import run_autonomous_quality_gates
        
        # Run a minimal quality gates check
        results = run_autonomous_quality_gates()
        
        assert "overall_score" in results
        assert "gates_passed" in results
        assert isinstance(results["overall_score"], (int, float))
        
        return {
            "passed": True,
            "quality_gates_functional": True,
            "overall_score": results["overall_score"]
        }
    except Exception as e:
        return {"passed": False, "error": str(e)}


def run_comprehensive_tests() -> Dict[str, Any]:
    """Run the comprehensive test suite."""
    return test_framework.run_all_tests()


def run_smoke_tests() -> Dict[str, Any]:
    """Run only smoke tests for quick validation."""
    results = test_framework.run_category(TestCategory.SMOKE)
    return {
        "smoke_tests": results,
        "all_passed": all(r.passed for r in results),
        "total_tests": len(results),
        "passed_tests": sum(1 for r in results if r.passed)
    }


if __name__ == "__main__":
    # Run comprehensive tests
    print("=" * 80)
    print("DGDM COMPREHENSIVE TEST FRAMEWORK")
    print("=" * 80)
    
    results = run_comprehensive_tests()
    
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Pass Rate: {results['summary']['pass_rate']:.1f}%")
    print(f"Execution Time: {results['total_execution_time']:.2f}s")
    
    print("\nCategory Breakdown:")
    for category, stats in results['category_breakdown'].items():
        print(f"  {category}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1f}%)")
    
    if results['failing_tests']:
        print(f"\nFailing Tests ({len(results['failing_tests'])}):")
        for test in results['failing_tests'][:5]:  # Show first 5
            print(f"  ✗ {test['name']} ({test['category']}): {test['error']}")
    
    if results['recommendations']:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  • {rec}")
    
    print("=" * 80)