"""
Autonomous Quality Gates for DGDM Histopath Lab.

This module implements comprehensive quality gates that run autonomously
and provide detailed feedback on code quality, security, and functionality.
"""

import sys
import subprocess
import logging
import json
import ast
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: Dict[str, Any]
    execution_time: float
    warnings: List[str]
    errors: List[str]


class AutonomousQualityGates:
    """Comprehensive autonomous quality gate system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.results = []
        self.overall_score = 0.0
        self.start_time = time.time()
        
        # Quality gate weights
        self.gate_weights = {
            "syntax_check": 0.20,
            "import_check": 0.15,
            "code_structure": 0.15,
            "security_scan": 0.20,
            "performance_analysis": 0.10,
            "documentation_check": 0.10,
            "test_coverage": 0.10
        }
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates in parallel where possible."""
        logger.info("Starting autonomous quality gates...")
        
        # Define quality gates
        gates = [
            ("syntax_check", self._syntax_check),
            ("import_check", self._import_check),
            ("code_structure", self._code_structure_analysis),
            ("security_scan", self._security_scan),
            ("performance_analysis", self._performance_analysis),
            ("documentation_check", self._documentation_check),
            ("test_coverage", self._test_coverage_analysis)
        ]
        
        # Run gates concurrently where safe
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(gate_func): gate_name 
                for gate_name, gate_func in gates
            }
            
            for future in concurrent.futures.as_completed(futures):
                gate_name = futures[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    self.results.append(result)
                    logger.info(f"✓ {gate_name}: {result.score:.2f}")
                except Exception as e:
                    logger.error(f"✗ {gate_name} failed: {e}")
                    self.results.append(QualityGateResult(
                        name=gate_name,
                        passed=False,
                        score=0.0,
                        details={"error": str(e)},
                        execution_time=0.0,
                        warnings=[],
                        errors=[str(e)]
                    ))
        
        # Calculate overall score
        self._calculate_overall_score()
        
        # Generate summary
        return self._generate_summary()
    
    def _syntax_check(self) -> QualityGateResult:
        """Check Python syntax across all files."""
        start_time = time.time()
        
        python_files = list(self.project_root.rglob("*.py"))
        syntax_errors = []
        checked_files = 0
        
        for py_file in python_files:
            # Skip __pycache__ and .git directories
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    ast.parse(source)
                checked_files += 1
            except SyntaxError as e:
                syntax_errors.append({
                    "file": str(py_file),
                    "line": e.lineno,
                    "error": str(e)
                })
            except Exception as e:
                syntax_errors.append({
                    "file": str(py_file),
                    "error": f"Failed to read: {e}"
                })
        
        execution_time = time.time() - start_time
        
        # Calculate score
        if checked_files == 0:
            score = 0.0
        else:
            score = max(0.0, 1.0 - (len(syntax_errors) / checked_files))
        
        return QualityGateResult(
            name="syntax_check",
            passed=len(syntax_errors) == 0,
            score=score,
            details={
                "checked_files": checked_files,
                "syntax_errors": len(syntax_errors),
                "error_details": syntax_errors[:10]  # First 10 errors
            },
            execution_time=execution_time,
            warnings=[],
            errors=[f"Syntax error in {err['file']}" for err in syntax_errors[:5]]
        )
    
    def _import_check(self) -> QualityGateResult:
        """Check import statements and dependencies."""
        start_time = time.time()
        
        python_files = list(self.project_root.rglob("*.py"))
        import_issues = []
        imports_checked = 0
        
        for py_file in python_files:
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                # Parse imports
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        imports_checked += 1
                        
                        # Check for common import issues
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name.startswith('.'):
                                    import_issues.append({
                                        "file": str(py_file),
                                        "line": node.lineno,
                                        "issue": f"Relative import in absolute context: {alias.name}"
                                    })
                        
                        elif isinstance(node, ast.ImportFrom):
                            if node.level > 0 and not node.module:
                                import_issues.append({
                                    "file": str(py_file),
                                    "line": node.lineno,
                                    "issue": "Relative import without module"
                                })
            
            except Exception as e:
                import_issues.append({
                    "file": str(py_file),
                    "error": f"Failed to analyze imports: {e}"
                })
        
        execution_time = time.time() - start_time
        
        # Calculate score
        if imports_checked == 0:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (len(import_issues) / max(imports_checked, 1)))
        
        return QualityGateResult(
            name="import_check",
            passed=len(import_issues) == 0,
            score=score,
            details={
                "imports_checked": imports_checked,
                "import_issues": len(import_issues),
                "issue_details": import_issues[:10]
            },
            execution_time=execution_time,
            warnings=[f"Import issue in {issue['file']}" for issue in import_issues[:3]],
            errors=[]
        )
    
    def _code_structure_analysis(self) -> QualityGateResult:
        """Analyze code structure and organization."""
        start_time = time.time()
        
        python_files = list(self.project_root.rglob("*.py"))
        structure_metrics = {
            "total_files": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "avg_function_length": 0,
            "avg_class_length": 0,
            "complex_functions": 0,
            "undocumented_functions": 0
        }
        
        function_lengths = []
        class_lengths = []
        
        for py_file in python_files:
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    lines = source.split('\n')
                
                structure_metrics["total_files"] += 1
                structure_metrics["total_lines"] += len(lines)
                
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        structure_metrics["total_functions"] += 1
                        
                        # Calculate function length
                        func_length = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 10
                        function_lengths.append(func_length)
                        
                        # Check complexity (simple heuristic)
                        if func_length > 50:
                            structure_metrics["complex_functions"] += 1
                        
                        # Check documentation
                        if not ast.get_docstring(node):
                            structure_metrics["undocumented_functions"] += 1
                    
                    elif isinstance(node, ast.ClassDef):
                        structure_metrics["total_classes"] += 1
                        class_length = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 20
                        class_lengths.append(class_length)
            
            except Exception:
                continue
        
        # Calculate averages
        if function_lengths:
            structure_metrics["avg_function_length"] = sum(function_lengths) / len(function_lengths)
        if class_lengths:
            structure_metrics["avg_class_length"] = sum(class_lengths) / len(class_lengths)
        
        execution_time = time.time() - start_time
        
        # Calculate score based on good practices
        score = 1.0
        
        # Penalize very complex functions
        if structure_metrics["total_functions"] > 0:
            complex_ratio = structure_metrics["complex_functions"] / structure_metrics["total_functions"]
            score -= complex_ratio * 0.3
        
        # Penalize undocumented functions
        if structure_metrics["total_functions"] > 0:
            undoc_ratio = structure_metrics["undocumented_functions"] / structure_metrics["total_functions"]
            score -= undoc_ratio * 0.2
        
        score = max(0.0, score)
        
        return QualityGateResult(
            name="code_structure",
            passed=score > 0.7,
            score=score,
            details=structure_metrics,
            execution_time=execution_time,
            warnings=[],
            errors=[]
        )
    
    def _security_scan(self) -> QualityGateResult:
        """Basic security analysis."""
        start_time = time.time()
        
        python_files = list(self.project_root.rglob("*.py"))
        security_issues = []
        
        # Security patterns to check
        security_patterns = {
            "hardcoded_secrets": [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ],
            "dangerous_functions": [
                r'eval\s*\(',
                r'exec\s*\(',
                r'subprocess\.call\s*\(',
                r'os\.system\s*\('
            ],
            "sql_injection_risk": [
                r'\.execute\s*\(\s*["\'][^"\']*%[^"\']*["\']',
                r'\.execute\s*\(\s*f["\'][^"\']*\{[^}]*\}[^"\']*["\']'
            ]
        }
        
        for py_file in python_files:
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                for category, patterns in security_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, source, re.IGNORECASE | re.MULTILINE)
                        for match in matches:
                            line_num = source[:match.start()].count('\n') + 1
                            security_issues.append({
                                "file": str(py_file),
                                "line": line_num,
                                "category": category,
                                "pattern": pattern,
                                "match": match.group()
                            })
            
            except Exception:
                continue
        
        execution_time = time.time() - start_time
        
        # Calculate score
        score = max(0.0, 1.0 - (len(security_issues) * 0.1))
        
        return QualityGateResult(
            name="security_scan",
            passed=len(security_issues) == 0,
            score=score,
            details={
                "security_issues": len(security_issues),
                "issue_details": security_issues[:10]
            },
            execution_time=execution_time,
            warnings=[f"Security issue in {issue['file']}" for issue in security_issues[:3]],
            errors=[]
        )
    
    def _performance_analysis(self) -> QualityGateResult:
        """Analyze potential performance issues."""
        start_time = time.time()
        
        python_files = list(self.project_root.rglob("*.py"))
        performance_issues = []
        
        # Performance anti-patterns
        performance_patterns = [
            (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', "Use enumerate() instead of range(len())"),
            (r'\.append\s*\([^)]*\)\s*(?:\n\s*)*\.append', "Multiple appends - consider extend()"),
            (r'open\s*\([^)]*\)\s*\.read\s*\(\)', "Use context manager for file operations"),
            (r'time\.sleep\s*\(\s*[0-9.]+\s*\)', "Sleep in main thread - consider async"),
        ]
        
        for py_file in python_files:
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                for pattern, suggestion in performance_patterns:
                    matches = re.finditer(pattern, source, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        line_num = source[:match.start()].count('\n') + 1
                        performance_issues.append({
                            "file": str(py_file),
                            "line": line_num,
                            "issue": suggestion,
                            "pattern": match.group()
                        })
            
            except Exception:
                continue
        
        execution_time = time.time() - start_time
        
        # Calculate score
        score = max(0.0, 1.0 - (len(performance_issues) * 0.05))
        
        return QualityGateResult(
            name="performance_analysis",
            passed=len(performance_issues) < 5,
            score=score,
            details={
                "performance_issues": len(performance_issues),
                "issue_details": performance_issues[:10]
            },
            execution_time=execution_time,
            warnings=[f"Performance issue in {issue['file']}" for issue in performance_issues[:3]],
            errors=[]
        )
    
    def _documentation_check(self) -> QualityGateResult:
        """Check documentation coverage and quality."""
        start_time = time.time()
        
        python_files = list(self.project_root.rglob("*.py"))
        doc_metrics = {
            "total_modules": 0,
            "documented_modules": 0,
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0
        }
        
        for py_file in python_files:
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                doc_metrics["total_modules"] += 1
                
                # Check module docstring
                if ast.get_docstring(tree):
                    doc_metrics["documented_modules"] += 1
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        doc_metrics["total_functions"] += 1
                        if ast.get_docstring(node):
                            doc_metrics["documented_functions"] += 1
                    
                    elif isinstance(node, ast.ClassDef):
                        doc_metrics["total_classes"] += 1
                        if ast.get_docstring(node):
                            doc_metrics["documented_classes"] += 1
            
            except Exception:
                continue
        
        execution_time = time.time() - start_time
        
        # Calculate documentation coverage
        doc_scores = []
        
        if doc_metrics["total_modules"] > 0:
            module_coverage = doc_metrics["documented_modules"] / doc_metrics["total_modules"]
            doc_scores.append(module_coverage)
        
        if doc_metrics["total_functions"] > 0:
            function_coverage = doc_metrics["documented_functions"] / doc_metrics["total_functions"]
            doc_scores.append(function_coverage)
        
        if doc_metrics["total_classes"] > 0:
            class_coverage = doc_metrics["documented_classes"] / doc_metrics["total_classes"]
            doc_scores.append(class_coverage)
        
        score = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
        
        return QualityGateResult(
            name="documentation_check",
            passed=score > 0.6,
            score=score,
            details=doc_metrics,
            execution_time=execution_time,
            warnings=[],
            errors=[]
        )
    
    def _test_coverage_analysis(self) -> QualityGateResult:
        """Analyze test coverage and test quality."""
        start_time = time.time()
        
        test_files = list(self.project_root.rglob("test_*.py")) + list(self.project_root.rglob("*_test.py"))
        tests_dir = self.project_root / "tests"
        
        if tests_dir.exists():
            test_files.extend(tests_dir.rglob("*.py"))
        
        test_metrics = {
            "test_files": len(test_files),
            "total_tests": 0,
            "test_functions": 0,
            "test_classes": 0
        }
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name.startswith('test_'):
                            test_metrics["test_functions"] += 1
                            test_metrics["total_tests"] += 1
                    
                    elif isinstance(node, ast.ClassDef):
                        if node.name.startswith('Test') or 'test' in node.name.lower():
                            test_metrics["test_classes"] += 1
            
            except Exception:
                continue
        
        execution_time = time.time() - start_time
        
        # Calculate score based on test presence and organization
        score = 0.0
        
        if test_metrics["test_files"] > 0:
            score += 0.4
        
        if test_metrics["total_tests"] > 10:
            score += 0.4
        elif test_metrics["total_tests"] > 0:
            score += 0.2
        
        if test_metrics["test_classes"] > 0:
            score += 0.2
        
        return QualityGateResult(
            name="test_coverage",
            passed=score > 0.5,
            score=score,
            details=test_metrics,
            execution_time=execution_time,
            warnings=[],
            errors=[]
        )
    
    def _calculate_overall_score(self):
        """Calculate weighted overall score."""
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in self.results:
            weight = self.gate_weights.get(result.name, 0.1)
            weighted_score += result.score * weight
            total_weight += weight
        
        self.overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        total_time = time.time() - self.start_time
        
        passed_gates = sum(1 for result in self.results if result.passed)
        total_gates = len(self.results)
        
        # Collect all warnings and errors
        all_warnings = []
        all_errors = []
        
        for result in self.results:
            all_warnings.extend(result.warnings)
            all_errors.extend(result.errors)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "execution_time": total_time,
            "overall_score": self.overall_score,
            "overall_grade": self._get_grade(self.overall_score),
            "gates_passed": f"{passed_gates}/{total_gates}",
            "gates_passed_percentage": round(100 * passed_gates / total_gates, 1) if total_gates > 0 else 0,
            "individual_results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "score": result.score,
                    "grade": self._get_grade(result.score),
                    "execution_time": result.execution_time,
                    "details": result.details
                }
                for result in self.results
            ],
            "warnings": all_warnings,
            "errors": all_errors,
            "recommendations": recommendations,
            "ready_for_production": self.overall_score > 0.8 and len(all_errors) == 0
        }
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        for result in self.results:
            if result.score < 0.7:
                if result.name == "syntax_check":
                    recommendations.append("Fix syntax errors to improve code compilation")
                elif result.name == "security_scan":
                    recommendations.append("Address security vulnerabilities before production")
                elif result.name == "documentation_check":
                    recommendations.append("Add docstrings to improve code maintainability")
                elif result.name == "test_coverage":
                    recommendations.append("Increase test coverage for better reliability")
        
        if self.overall_score < 0.8:
            recommendations.append("Overall quality score below production threshold")
        
        return recommendations


def run_autonomous_quality_gates(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Run autonomous quality gates and return results."""
    if project_root is None:
        project_root = Path.cwd()
    
    gates = AutonomousQualityGates(project_root)
    return gates.run_all_gates()


if __name__ == "__main__":
    # Run quality gates
    results = run_autonomous_quality_gates()
    
    # Print summary
    print("=" * 80)
    print("AUTONOMOUS QUALITY GATES REPORT")
    print("=" * 80)
    print(f"Overall Score: {results['overall_score']:.2f} ({results['overall_grade']})")
    print(f"Gates Passed: {results['gates_passed']} ({results['gates_passed_percentage']}%)")
    print(f"Execution Time: {results['execution_time']:.2f}s")
    print(f"Production Ready: {'Yes' if results['ready_for_production'] else 'No'}")
    
    print("\nIndividual Gate Results:")
    for result in results['individual_results']:
        status = "✓" if result['passed'] else "✗"
        print(f"  {status} {result['name']}: {result['score']:.2f} ({result['grade']})")
    
    if results['recommendations']:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  • {rec}")
    
    print("=" * 80)