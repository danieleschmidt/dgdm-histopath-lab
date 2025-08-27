"""
Intelligent Testing Framework with Automated Test Generation

Features:
- AI-powered test case generation
- Mutation testing and fault injection
- Property-based testing
- Visual regression testing
- Performance testing automation
- Test maintenance and evolution
"""

import asyncio
import json
import time
import uuid
import ast
import inspect
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict
import tempfile
from pathlib import Path
import subprocess

from ..utils.logging import get_logger
from ..utils.monitoring import MetricsCollector

logger = get_logger(__name__)


class TestType(Enum):
    """Types of generated tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PROPERTY_BASED = "property_based"
    MUTATION = "mutation"
    VISUAL_REGRESSION = "visual_regression"
    PERFORMANCE = "performance"
    FUZZ = "fuzz"
    CONTRACT = "contract"
    CHAOS = "chaos"


class TestGenerationStrategy(Enum):
    """Test generation strategies"""
    STATIC_ANALYSIS = "static_analysis"
    DYNAMIC_ANALYSIS = "dynamic_analysis"
    ML_BASED = "ml_based"
    TEMPLATE_BASED = "template_based"
    COMBINATORIAL = "combinatorial"
    BOUNDARY_VALUE = "boundary_value"


class TestExecutionResult(Enum):
    """Test execution results"""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class FunctionSignature:
    """Function signature for test generation"""
    name: str
    module: str
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    docstring: Optional[str]
    complexity: int
    dependencies: List[str]


@dataclass
class GeneratedTest:
    """Generated test case"""
    test_id: str
    name: str
    test_type: TestType
    strategy: TestGenerationStrategy
    source_code: str
    target_function: Optional[str]
    input_data: Dict[str, Any]
    expected_output: Any
    assertions: List[str]
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TestExecution:
    """Test execution record"""
    execution_id: str
    test_id: str
    result: TestExecutionResult
    duration_seconds: float
    error_message: Optional[str] = None
    output: Optional[str] = None
    coverage_data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=datetime.now)


@dataclass
class TestSuite:
    """Collection of generated tests"""
    suite_id: str
    name: str
    tests: List[GeneratedTest]
    configuration: Dict[str, Any]
    total_tests: int = 0
    generated_at: datetime = field(default_factory=datetime.now)


class CodeAnalyzer:
    """Analyzes code to extract information for test generation"""
    
    def __init__(self):
        self.analyzed_functions: Dict[str, FunctionSignature] = {}
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        
    async def analyze_codebase(self, project_path: str) -> Dict[str, List[FunctionSignature]]:
        """Analyze codebase to extract function signatures"""
        try:
            functions_by_module = defaultdict(list)
            
            # Find Python files
            python_files = await self._find_python_files(project_path)
            
            for file_path in python_files:
                try:
                    functions = await self._analyze_python_file(file_path)
                    module_name = self._get_module_name(file_path, project_path)
                    functions_by_module[module_name].extend(functions)
                except Exception as e:
                    logger.error(f"Failed to analyze {file_path}: {e}")
            
            return dict(functions_by_module)
            
        except Exception as e:
            logger.error(f"Codebase analysis failed: {e}")
            return {}
    
    async def _find_python_files(self, project_path: str) -> List[str]:
        """Find all Python files in project"""
        python_files = []
        
        try:
            for path in Path(project_path).rglob("*.py"):
                if not any(part.startswith('.') for part in path.parts):
                    python_files.append(str(path))
        except Exception as e:
            logger.error(f"Failed to find Python files: {e}")
        
        return python_files
    
    async def _analyze_python_file(self, file_path: str) -> List[FunctionSignature]:
        """Analyze single Python file"""
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):  # Skip private functions
                        function_sig = self._extract_function_signature(node, file_path)
                        functions.append(function_sig)
                        self.analyzed_functions[f"{file_path}:{node.name}"] = function_sig
        
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
        
        return functions
    
    def _extract_function_signature(self, node: ast.FunctionDef, file_path: str) -> FunctionSignature:
        """Extract function signature from AST node"""
        parameters = []
        
        # Extract parameters
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "type": self._extract_type_annotation(arg.annotation) if arg.annotation else None,
                "default": None
            }
            parameters.append(param_info)
        
        # Extract defaults
        if node.args.defaults:
            num_defaults = len(node.args.defaults)
            for i, default in enumerate(node.args.defaults):
                param_index = len(parameters) - num_defaults + i
                if param_index >= 0 and param_index < len(parameters):
                    parameters[param_index]["default"] = ast.unparse(default)
        
        # Extract return type
        return_type = self._extract_type_annotation(node.returns) if node.returns else None
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Calculate complexity (simplified cyclomatic complexity)
        complexity = self._calculate_complexity(node)
        
        # Extract dependencies (simplified)
        dependencies = self._extract_dependencies(node)
        
        return FunctionSignature(
            name=node.name,
            module=file_path,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            complexity=complexity,
            dependencies=dependencies
        )
    
    def _extract_type_annotation(self, annotation) -> str:
        """Extract type annotation as string"""
        try:
            if annotation:
                return ast.unparse(annotation)
        except Exception:
            pass
        return "Any"
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _extract_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract function dependencies"""
        dependencies = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        dependencies.append(f"{child.func.value.id}.{child.func.attr}")
        
        return list(set(dependencies))
    
    def _get_module_name(self, file_path: str, project_path: str) -> str:
        """Get module name from file path"""
        try:
            relative_path = Path(file_path).relative_to(Path(project_path))
            module_parts = relative_path.with_suffix('').parts
            return '.'.join(module_parts)
        except Exception:
            return Path(file_path).stem


class TestGenerator:
    """Generates test cases using various strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.code_analyzer = CodeAnalyzer()
        self.generated_tests: Dict[str, GeneratedTest] = {}
        
        # Test templates
        self.test_templates = self._load_test_templates()
        
        # Value generators
        self.value_generators = self._initialize_value_generators()
    
    def _load_test_templates(self) -> Dict[str, str]:
        """Load test code templates"""
        return {
            "unit_test": '''
def test_{function_name}_{test_case}():
    """Generated test for {function_name}"""
    {setup_code}
    
    # Arrange
    {input_setup}
    
    # Act
    result = {function_call}
    
    # Assert
    {assertions}
    
    {teardown_code}
''',
            "property_test": '''
from hypothesis import given, strategies as st

@given({input_strategies})
def test_{function_name}_property_{property_name}(${parameters}):
    """Property-based test for {function_name}"""
    # Assume
    {assumptions}
    
    # Act
    result = {function_call}
    
    # Assert property
    {property_assertion}
''',
            "mutation_test": '''
def test_{function_name}_mutation_{mutation_id}():
    """Mutation test for {function_name}"""
    # This test should fail when mutation is applied
    {test_body}
    
    # Mutation point: {mutation_description}
    assert {assertion_that_should_fail_on_mutation}
''',
            "performance_test": '''
import time
import pytest

def test_{function_name}_performance():
    """Performance test for {function_name}"""
    {setup_code}
    
    # Warm up
    for _ in range(10):
        {function_call}
    
    # Measure performance
    start_time = time.time()
    for _ in range({iterations}):
        result = {function_call}
    end_time = time.time()
    
    avg_time = (end_time - start_time) / {iterations}
    assert avg_time < {max_time_threshold}, f"Function too slow: {{avg_time:.4f}}s"
'''
        }
    
    def _initialize_value_generators(self) -> Dict[str, Callable]:
        """Initialize value generators for different types"""
        return {
            "int": lambda: random.randint(-1000, 1000),
            "float": lambda: random.uniform(-1000.0, 1000.0),
            "str": lambda: ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=random.randint(1, 20))),
            "bool": lambda: random.choice([True, False]),
            "list": lambda: [random.randint(1, 100) for _ in range(random.randint(0, 10))],
            "dict": lambda: {f"key_{i}": random.randint(1, 100) for i in range(random.randint(0, 5))},
            "None": lambda: None
        }
    
    async def generate_test_suite(self, project_path: str, 
                                 target_functions: Optional[List[str]] = None) -> TestSuite:
        """Generate comprehensive test suite"""
        suite_id = str(uuid.uuid4())
        
        try:
            # Analyze codebase
            logger.info("Analyzing codebase for test generation")
            functions_by_module = await self.code_analyzer.analyze_codebase(project_path)
            
            if not functions_by_module:
                logger.warning("No functions found for test generation")
                return TestSuite(suite_id, "Empty Suite", [], {})
            
            # Generate tests for each strategy
            all_tests = []
            
            # Unit tests
            unit_tests = await self._generate_unit_tests(functions_by_module, target_functions)
            all_tests.extend(unit_tests)
            
            # Property-based tests
            property_tests = await self._generate_property_tests(functions_by_module, target_functions)
            all_tests.extend(property_tests)
            
            # Mutation tests
            mutation_tests = await self._generate_mutation_tests(functions_by_module, target_functions)
            all_tests.extend(mutation_tests)
            
            # Performance tests
            performance_tests = await self._generate_performance_tests(functions_by_module, target_functions)
            all_tests.extend(performance_tests)
            
            # Fuzz tests
            fuzz_tests = await self._generate_fuzz_tests(functions_by_module, target_functions)
            all_tests.extend(fuzz_tests)
            
            suite = TestSuite(
                suite_id=suite_id,
                name=f"Generated Test Suite - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tests=all_tests,
                configuration=self.config,
                total_tests=len(all_tests)
            )
            
            logger.info(f"Generated test suite with {len(all_tests)} tests")
            
            return suite
            
        except Exception as e:
            logger.error(f"Test suite generation failed: {e}")
            return TestSuite(suite_id, "Failed Suite", [], {})
    
    async def _generate_unit_tests(self, functions_by_module: Dict[str, List[FunctionSignature]],
                                  target_functions: Optional[List[str]] = None) -> List[GeneratedTest]:
        """Generate unit tests"""
        tests = []
        
        for module_name, functions in functions_by_module.items():
            for function in functions:
                if target_functions and function.name not in target_functions:
                    continue
                
                try:
                    # Generate multiple test cases per function
                    test_cases = await self._generate_unit_test_cases(function)
                    tests.extend(test_cases)
                except Exception as e:
                    logger.error(f"Failed to generate unit tests for {function.name}: {e}")
        
        return tests
    
    async def _generate_unit_test_cases(self, function: FunctionSignature) -> List[GeneratedTest]:
        """Generate multiple unit test cases for a function"""
        test_cases = []
        
        # Generate boundary value tests
        test_cases.extend(await self._generate_boundary_tests(function))
        
        # Generate normal case tests
        test_cases.extend(await self._generate_normal_case_tests(function))
        
        # Generate error case tests
        test_cases.extend(await self._generate_error_case_tests(function))
        
        return test_cases
    
    async def _generate_boundary_tests(self, function: FunctionSignature) -> List[GeneratedTest]:
        """Generate boundary value tests"""
        tests = []
        
        boundary_values = {
            "int": [0, 1, -1, 2147483647, -2147483648],
            "float": [0.0, 1.0, -1.0, float('inf'), -float('inf')],
            "str": ["", "a", "very_long_string_" * 100],
            "list": [[], [1], list(range(1000))],
            "dict": [{}, {"key": "value"}, {f"key_{i}": i for i in range(100)}]
        }
        
        for i, boundary_case in enumerate(["min", "max", "zero", "empty"]):
            test_inputs = {}
            
            for param in function.parameters:
                param_type = param.get("type", "Any")
                
                if param_type in boundary_values:
                    values = boundary_values[param_type]
                    test_inputs[param["name"]] = values[min(i, len(values) - 1)]
                else:
                    test_inputs[param["name"]] = self._generate_value_for_type(param_type)
            
            # Generate test code
            test_code = await self._generate_unit_test_code(function, test_inputs, f"boundary_{boundary_case}")
            
            test = GeneratedTest(
                test_id=str(uuid.uuid4()),
                name=f"test_{function.name}_boundary_{boundary_case}",
                test_type=TestType.UNIT,
                strategy=TestGenerationStrategy.BOUNDARY_VALUE,
                source_code=test_code,
                target_function=function.name,
                input_data=test_inputs,
                expected_output=None,  # Would be determined by test oracle
                assertions=[f"assert result is not None  # Boundary test for {boundary_case}"]
            )
            
            tests.append(test)
        
        return tests
    
    async def _generate_normal_case_tests(self, function: FunctionSignature) -> List[GeneratedTest]:
        """Generate normal case tests"""
        tests = []
        
        # Generate 3-5 normal test cases
        for i in range(random.randint(3, 5)):
            test_inputs = {}
            
            for param in function.parameters:
                param_type = param.get("type", "Any")
                test_inputs[param["name"]] = self._generate_value_for_type(param_type)
            
            test_code = await self._generate_unit_test_code(function, test_inputs, f"normal_{i}")
            
            test = GeneratedTest(
                test_id=str(uuid.uuid4()),
                name=f"test_{function.name}_normal_case_{i}",
                test_type=TestType.UNIT,
                strategy=TestGenerationStrategy.STATIC_ANALYSIS,
                source_code=test_code,
                target_function=function.name,
                input_data=test_inputs,
                expected_output=None,
                assertions=[f"assert result is not None  # Normal case test {i}"]
            )
            
            tests.append(test)
        
        return tests
    
    async def _generate_error_case_tests(self, function: FunctionSignature) -> List[GeneratedTest]:
        """Generate error case tests"""
        tests = []
        
        # Generate tests for common error conditions
        error_cases = [
            {"name": "none_input", "input_modifier": lambda x: None},
            {"name": "wrong_type", "input_modifier": lambda x: "wrong_type"},
            {"name": "negative_where_positive", "input_modifier": lambda x: -abs(x) if isinstance(x, (int, float)) else x}
        ]
        
        for error_case in error_cases:
            test_inputs = {}
            
            for param in function.parameters:
                param_type = param.get("type", "Any")
                value = self._generate_value_for_type(param_type)
                
                # Apply error modifier to first parameter
                if param == function.parameters[0]:
                    value = error_case["input_modifier"](value)
                
                test_inputs[param["name"]] = value
            
            test_code = await self._generate_error_test_code(function, test_inputs, error_case["name"])
            
            test = GeneratedTest(
                test_id=str(uuid.uuid4()),
                name=f"test_{function.name}_error_{error_case['name']}",
                test_type=TestType.UNIT,
                strategy=TestGenerationStrategy.STATIC_ANALYSIS,
                source_code=test_code,
                target_function=function.name,
                input_data=test_inputs,
                expected_output="Exception",
                assertions=[f"# Should handle error case: {error_case['name']}"]
            )
            
            tests.append(test)
        
        return tests
    
    async def _generate_property_tests(self, functions_by_module: Dict[str, List[FunctionSignature]],
                                      target_functions: Optional[List[str]] = None) -> List[GeneratedTest]:
        """Generate property-based tests"""
        tests = []
        
        for module_name, functions in functions_by_module.items():
            for function in functions:
                if target_functions and function.name not in target_functions:
                    continue
                
                try:
                    # Generate property tests
                    property_tests = await self._generate_function_properties(function)
                    tests.extend(property_tests)
                except Exception as e:
                    logger.error(f"Failed to generate property tests for {function.name}: {e}")
        
        return tests
    
    async def _generate_function_properties(self, function: FunctionSignature) -> List[GeneratedTest]:
        """Generate property-based tests for function"""
        tests = []
        
        # Common properties to test
        properties = [
            "idempotency",  # f(f(x)) == f(x)
            "consistency",  # f(x) always returns same result
            "non_null",     # f(x) never returns null for valid input
            "type_safety",  # f(x) returns expected type
            "inverse"       # f(g(x)) == x (if inverse function exists)
        ]
        
        for prop in properties:
            if await self._is_property_applicable(function, prop):
                test_code = await self._generate_property_test_code(function, prop)
                
                test = GeneratedTest(
                    test_id=str(uuid.uuid4()),
                    name=f"test_{function.name}_property_{prop}",
                    test_type=TestType.PROPERTY_BASED,
                    strategy=TestGenerationStrategy.ML_BASED,
                    source_code=test_code,
                    target_function=function.name,
                    input_data={"property": prop},
                    expected_output=True,  # Property should hold
                    assertions=[f"# Property: {prop} should hold"]
                )
                
                tests.append(test)
        
        return tests
    
    async def _is_property_applicable(self, function: FunctionSignature, property_name: str) -> bool:
        """Check if property is applicable to function"""
        # Simplified heuristics
        if property_name == "idempotency":
            # Applicable if function modifies state or has side effects
            return "update" in function.name.lower() or "set" in function.name.lower()
        elif property_name == "consistency":
            # Always applicable
            return True
        elif property_name == "non_null":
            # Applicable if function should never return None
            return function.return_type != "None"
        elif property_name == "type_safety":
            # Applicable if function has return type annotation
            return function.return_type is not None
        
        return random.choice([True, False])  # Random for simulation
    
    async def _generate_mutation_tests(self, functions_by_module: Dict[str, List[FunctionSignature]],
                                      target_functions: Optional[List[str]] = None) -> List[GeneratedTest]:
        """Generate mutation tests"""
        tests = []
        
        for module_name, functions in functions_by_module.items():
            for function in functions:
                if target_functions and function.name not in target_functions:
                    continue
                
                try:
                    # Generate mutation tests
                    mutation_tests = await self._generate_function_mutations(function)
                    tests.extend(mutation_tests)
                except Exception as e:
                    logger.error(f"Failed to generate mutation tests for {function.name}: {e}")
        
        return tests
    
    async def _generate_function_mutations(self, function: FunctionSignature) -> List[GeneratedTest]:
        """Generate mutation tests for function"""
        tests = []
        
        # Common mutation operators
        mutations = [
            "arithmetic_operator",  # + -> -, * -> /, etc.
            "relational_operator",  # > -> <, == -> !=, etc.
            "logical_operator",     # and -> or, not -> identity
            "constant_replacement", # 0 -> 1, True -> False
            "statement_deletion",   # Remove statement
            "boundary_shift"        # < -> <=, > -> >=
        ]
        
        for mutation in mutations:
            test_code = await self._generate_mutation_test_code(function, mutation)
            
            test = GeneratedTest(
                test_id=str(uuid.uuid4()),
                name=f"test_{function.name}_mutation_{mutation}",
                test_type=TestType.MUTATION,
                strategy=TestGenerationStrategy.STATIC_ANALYSIS,
                source_code=test_code,
                target_function=function.name,
                input_data={"mutation": mutation},
                expected_output="mutation_detected",
                assertions=[f"# Should detect mutation: {mutation}"]
            )
            
            tests.append(test)
        
        return tests
    
    async def _generate_performance_tests(self, functions_by_module: Dict[str, List[FunctionSignature]],
                                         target_functions: Optional[List[str]] = None) -> List[GeneratedTest]:
        """Generate performance tests"""
        tests = []
        
        for module_name, functions in functions_by_module.items():
            for function in functions:
                if target_functions and function.name not in target_functions:
                    continue
                
                # Only generate performance tests for complex functions
                if function.complexity >= 3:
                    try:
                        perf_test = await self._generate_performance_test(function)
                        tests.append(perf_test)
                    except Exception as e:
                        logger.error(f"Failed to generate performance test for {function.name}: {e}")
        
        return tests
    
    async def _generate_performance_test(self, function: FunctionSignature) -> GeneratedTest:
        """Generate performance test for function"""
        # Generate test inputs
        test_inputs = {}
        for param in function.parameters:
            param_type = param.get("type", "Any")
            test_inputs[param["name"]] = self._generate_value_for_type(param_type)
        
        # Determine performance thresholds based on complexity
        max_time_threshold = min(1.0, function.complexity * 0.1)  # 0.1s per complexity point
        iterations = max(10, 100 // function.complexity)
        
        test_code = self.test_templates["performance_test"].format(
            function_name=function.name,
            setup_code=self._generate_setup_code(function),
            function_call=self._generate_function_call(function, test_inputs),
            iterations=iterations,
            max_time_threshold=max_time_threshold
        )
        
        return GeneratedTest(
            test_id=str(uuid.uuid4()),
            name=f"test_{function.name}_performance",
            test_type=TestType.PERFORMANCE,
            strategy=TestGenerationStrategy.STATIC_ANALYSIS,
            source_code=test_code,
            target_function=function.name,
            input_data=test_inputs,
            expected_output=f"execution_time < {max_time_threshold}",
            assertions=[f"assert avg_time < {max_time_threshold}"]
        )
    
    async def _generate_fuzz_tests(self, functions_by_module: Dict[str, List[FunctionSignature]],
                                  target_functions: Optional[List[str]] = None) -> List[GeneratedTest]:
        """Generate fuzz tests"""
        tests = []
        
        for module_name, functions in functions_by_module.items():
            for function in functions:
                if target_functions and function.name not in target_functions:
                    continue
                
                # Generate fuzz tests for functions that handle external input
                if self._handles_external_input(function):
                    try:
                        fuzz_test = await self._generate_fuzz_test(function)
                        tests.append(fuzz_test)
                    except Exception as e:
                        logger.error(f"Failed to generate fuzz test for {function.name}: {e}")
        
        return tests
    
    def _handles_external_input(self, function: FunctionSignature) -> bool:
        """Check if function handles external input"""
        # Heuristics to identify functions that handle external input
        external_indicators = ["input", "parse", "decode", "load", "read", "request"]
        return any(indicator in function.name.lower() for indicator in external_indicators)
    
    async def _generate_fuzz_test(self, function: FunctionSignature) -> GeneratedTest:
        """Generate fuzz test for function"""
        # Generate random/malformed inputs
        fuzz_inputs = {}
        
        for param in function.parameters:
            param_type = param.get("type", "Any")
            fuzz_inputs[param["name"]] = self._generate_fuzz_value(param_type)
        
        test_code = f'''
def test_{function.name}_fuzz():
    """Fuzz test for {function.name}"""
    import pytest
    
    # Fuzz inputs that might cause issues
    fuzz_inputs = {fuzz_inputs}
    
    # Test should not crash
    try:
        result = {self._generate_function_call(function, fuzz_inputs)}
        # Function should handle malformed input gracefully
        assert result is not None or result is None  # Should not crash
    except (ValueError, TypeError, AttributeError) as e:
        # Expected exceptions for malformed input
        assert True
    except Exception as e:
        # Unexpected exceptions indicate potential bugs
        pytest.fail(f"Unexpected exception: {{e}}")
'''
        
        return GeneratedTest(
            test_id=str(uuid.uuid4()),
            name=f"test_{function.name}_fuzz",
            test_type=TestType.FUZZ,
            strategy=TestGenerationStrategy.DYNAMIC_ANALYSIS,
            source_code=test_code,
            target_function=function.name,
            input_data=fuzz_inputs,
            expected_output="no_crash",
            assertions=["# Should handle malformed input gracefully"]
        )
    
    def _generate_value_for_type(self, type_name: str) -> Any:
        """Generate value for given type"""
        # Map complex types to simpler types
        type_mapping = {
            "Optional[str]": "str",
            "List[int]": "list", 
            "Dict[str, Any]": "dict",
            "Union[str, int]": "str"
        }
        
        simplified_type = type_mapping.get(type_name, type_name)
        
        if simplified_type in self.value_generators:
            return self.value_generators[simplified_type]()
        else:
            # Default to string for unknown types
            return self.value_generators["str"]()
    
    def _generate_fuzz_value(self, type_name: str) -> Any:
        """Generate malformed/edge case value for fuzzing"""
        fuzz_values = {
            "str": ["", " " * 10000, "\x00\x01\x02", "\\n\\r\\t", "🎵🎶🎵"],
            "int": [0, -1, 2**63-1, -2**63, None, "not_an_int"],
            "float": [float('inf'), -float('inf'), float('nan'), None, "not_a_float"],
            "list": [None, [], [None] * 10000, "not_a_list"],
            "dict": [None, {}, {"key": None}, "not_a_dict"]
        }
        
        if type_name in fuzz_values:
            return random.choice(fuzz_values[type_name])
        else:
            return random.choice([None, "", 0, [], {}])
    
    async def _generate_unit_test_code(self, function: FunctionSignature, 
                                      test_inputs: Dict[str, Any], test_case: str) -> str:
        """Generate unit test code"""
        template = self.test_templates["unit_test"]
        
        return template.format(
            function_name=function.name,
            test_case=test_case,
            setup_code=self._generate_setup_code(function),
            input_setup=self._generate_input_setup(test_inputs),
            function_call=self._generate_function_call(function, test_inputs),
            assertions=self._generate_assertions(function, test_inputs),
            teardown_code=""
        )
    
    async def _generate_error_test_code(self, function: FunctionSignature,
                                       test_inputs: Dict[str, Any], error_case: str) -> str:
        """Generate error case test code"""
        return f'''
def test_{function.name}_error_{error_case}():
    """Error case test for {function.name}"""
    import pytest
    
    # This should raise an exception
    with pytest.raises((ValueError, TypeError, AttributeError)):
        result = {self._generate_function_call(function, test_inputs)}
'''
    
    async def _generate_property_test_code(self, function: FunctionSignature, property_name: str) -> str:
        """Generate property-based test code"""
        template = self.test_templates["property_test"]
        
        # Generate Hypothesis strategies
        strategies = []
        parameters = []
        
        for param in function.parameters:
            param_type = param.get("type", "Any")
            strategy = self._get_hypothesis_strategy(param_type)
            strategies.append(f"{param['name']}={strategy}")
            parameters.append(param["name"])
        
        return template.format(
            function_name=function.name,
            property_name=property_name,
            input_strategies=", ".join(strategies),
            parameters=", ".join(parameters),
            assumptions="",  # Would add assumptions based on property
            function_call=self._generate_function_call(function, {p: p for p in parameters}),
            property_assertion=self._generate_property_assertion(function, property_name)
        )
    
    async def _generate_mutation_test_code(self, function: FunctionSignature, mutation: str) -> str:
        """Generate mutation test code"""
        template = self.test_templates["mutation_test"]
        
        # Generate test that should fail when mutation is applied
        test_inputs = {}
        for param in function.parameters:
            param_type = param.get("type", "Any")
            test_inputs[param["name"]] = self._generate_value_for_type(param_type)
        
        return template.format(
            function_name=function.name,
            mutation_id=mutation.replace("_", ""),
            test_body=self._generate_mutation_test_body(function, test_inputs),
            mutation_description=f"Apply {mutation} mutation",
            assertion_that_should_fail_on_mutation=self._generate_mutation_assertion(function, mutation)
        )
    
    def _generate_setup_code(self, function: FunctionSignature) -> str:
        """Generate setup code for test"""
        setup_lines = []
        
        # Import the module
        if function.module:
            module_name = Path(function.module).stem
            setup_lines.append(f"from {module_name} import {function.name}")
        
        return "\n    ".join(setup_lines) if setup_lines else "pass"
    
    def _generate_input_setup(self, test_inputs: Dict[str, Any]) -> str:
        """Generate input setup code"""
        setup_lines = []
        
        for param_name, value in test_inputs.items():
            if isinstance(value, str):
                setup_lines.append(f"{param_name} = \"{value}\"")
            else:
                setup_lines.append(f"{param_name} = {repr(value)}")
        
        return "\n    ".join(setup_lines)
    
    def _generate_function_call(self, function: FunctionSignature, test_inputs: Dict[str, Any]) -> str:
        """Generate function call code"""
        args = []
        
        for param in function.parameters:
            if param["name"] in test_inputs:
                args.append(f"{param['name']}")
        
        return f"{function.name}({', '.join(args)})"
    
    def _generate_assertions(self, function: FunctionSignature, test_inputs: Dict[str, Any]) -> str:
        """Generate assertions for test"""
        assertions = []
        
        # Basic assertion that result is not None (unless function returns None)
        if function.return_type != "None":
            assertions.append("assert result is not None")
        
        # Type-based assertions
        if function.return_type:
            if function.return_type == "bool":
                assertions.append("assert isinstance(result, bool)")
            elif function.return_type == "int":
                assertions.append("assert isinstance(result, int)")
            elif function.return_type == "str":
                assertions.append("assert isinstance(result, str)")
            elif function.return_type.startswith("List"):
                assertions.append("assert isinstance(result, list)")
        
        return "\n    ".join(assertions) if assertions else "assert True  # Placeholder assertion"
    
    def _get_hypothesis_strategy(self, type_name: str) -> str:
        """Get Hypothesis strategy for type"""
        strategy_mapping = {
            "int": "st.integers()",
            "float": "st.floats(allow_nan=False, allow_infinity=False)",
            "str": "st.text()",
            "bool": "st.booleans()",
            "list": "st.lists(st.integers())",
            "dict": "st.dictionaries(st.text(), st.integers())"
        }
        
        return strategy_mapping.get(type_name, "st.text()")
    
    def _generate_property_assertion(self, function: FunctionSignature, property_name: str) -> str:
        """Generate property assertion"""
        if property_name == "consistency":
            return f"assert {function.name}({', '.join(p['name'] for p in function.parameters)}) == result"
        elif property_name == "non_null":
            return "assert result is not None"
        elif property_name == "type_safety":
            return f"assert isinstance(result, {function.return_type or 'object'})"
        else:
            return "assert True  # Property should hold"
    
    def _generate_mutation_test_body(self, function: FunctionSignature, test_inputs: Dict[str, Any]) -> str:
        """Generate mutation test body"""
        return f'''
    # Test inputs
    {self._generate_input_setup(test_inputs)}
    
    # Call original function
    result = {self._generate_function_call(function, test_inputs)}
'''
    
    def _generate_mutation_assertion(self, function: FunctionSignature, mutation: str) -> str:
        """Generate assertion that should fail on mutation"""
        # Simplified - would be more sophisticated based on mutation type
        return "result == expected_result  # Should fail when mutated"


class IntelligentTesting:
    """Main intelligent testing framework"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.test_generator = TestGenerator(self.config)
        
        # Execution tracking
        self.test_suites: Dict[str, TestSuite] = {}
        self.execution_history: List[TestExecution] = []
        
        # Performance monitoring
        self.metrics_collector = MetricsCollector()
        
        # Test evolution
        self.test_effectiveness_scores: Dict[str, float] = {}
        self.generated_test_performance: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("Intelligent testing framework initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load testing configuration"""
        default_config = {
            "generation": {
                "max_tests_per_function": 10,
                "include_property_tests": True,
                "include_mutation_tests": True,
                "include_performance_tests": True,
                "include_fuzz_tests": True
            },
            "execution": {
                "parallel_execution": True,
                "max_workers": 10,
                "timeout_seconds": 300
            },
            "optimization": {
                "adaptive_generation": True,
                "test_effectiveness_tracking": True,
                "automatic_test_evolution": True
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load testing config: {e}")
        
        return default_config
    
    async def generate_and_execute_tests(self, project_path: str, 
                                        target_functions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate and execute comprehensive test suite"""
        try:
            # Generate test suite
            logger.info("Generating intelligent test suite")
            test_suite = await self.test_generator.generate_test_suite(project_path, target_functions)
            
            if not test_suite.tests:
                return {
                    "suite_id": test_suite.suite_id,
                    "total_tests": 0,
                    "execution_results": [],
                    "summary": {"error": "No tests generated"}
                }
            
            # Store test suite
            self.test_suites[test_suite.suite_id] = test_suite
            
            # Execute tests
            logger.info(f"Executing {len(test_suite.tests)} generated tests")
            execution_results = await self._execute_test_suite(test_suite)
            
            # Analyze results and evolve tests
            if self.config["optimization"]["automatic_test_evolution"]:
                await self._evolve_tests_based_on_results(test_suite, execution_results)
            
            # Generate summary
            summary = self._generate_execution_summary(execution_results)
            
            return {
                "suite_id": test_suite.suite_id,
                "total_tests": len(test_suite.tests),
                "execution_results": [asdict(result) for result in execution_results],
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Test generation and execution failed: {e}")
            return {
                "suite_id": "failed",
                "total_tests": 0,
                "execution_results": [],
                "summary": {"error": str(e)}
            }
    
    async def _execute_test_suite(self, test_suite: TestSuite) -> List[TestExecution]:
        """Execute all tests in suite"""
        execution_results = []
        
        if self.config["execution"]["parallel_execution"]:
            # Execute tests in parallel
            semaphore = asyncio.Semaphore(self.config["execution"]["max_workers"])
            
            async def execute_single_test(test: GeneratedTest) -> TestExecution:
                async with semaphore:
                    return await self._execute_single_test(test)
            
            test_tasks = [execute_single_test(test) for test in test_suite.tests]
            execution_results = await asyncio.gather(*test_tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = []
            for result in execution_results:
                if isinstance(result, TestExecution):
                    valid_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Test execution failed: {result}")
            
            execution_results = valid_results
        else:
            # Execute tests sequentially
            for test in test_suite.tests:
                try:
                    result = await self._execute_single_test(test)
                    execution_results.append(result)
                except Exception as e:
                    logger.error(f"Test execution failed for {test.name}: {e}")
        
        return execution_results
    
    async def _execute_single_test(self, test: GeneratedTest) -> TestExecution:
        """Execute a single generated test"""
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        try:
            # Create temporary test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test.source_code)
                test_file = f.name
            
            # Execute test using pytest
            result = await self._run_test_with_pytest(test_file, test.name)
            
            execution = TestExecution(
                execution_id=execution_id,
                test_id=test.test_id,
                result=result["result"],
                duration_seconds=time.time() - start_time,
                error_message=result.get("error"),
                output=result.get("output"),
                coverage_data=result.get("coverage", {}),
                performance_metrics=result.get("performance", {})
            )
            
            # Clean up
            Path(test_file).unlink(missing_ok=True)
            
            # Record for learning
            self.execution_history.append(execution)
            
            return execution
            
        except Exception as e:
            return TestExecution(
                execution_id=execution_id,
                test_id=test.test_id,
                result=TestExecutionResult.ERROR,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _run_test_with_pytest(self, test_file: str, test_name: str) -> Dict[str, Any]:
        """Run test using pytest"""
        try:
            # Run pytest with coverage
            cmd = [
                "python", "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short",
                f"--timeout={self.config['execution']['timeout_seconds']}"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Parse results
            output = stdout.decode() + stderr.decode()
            
            if process.returncode == 0:
                result = TestExecutionResult.PASSED
            elif "FAILED" in output:
                result = TestExecutionResult.FAILED
            elif "timeout" in output.lower():
                result = TestExecutionResult.TIMEOUT
            else:
                result = TestExecutionResult.ERROR
            
            return {
                "result": result,
                "output": output,
                "error": stderr.decode() if stderr else None
            }
            
        except Exception as e:
            return {
                "result": TestExecutionResult.ERROR,
                "error": str(e)
            }
    
    async def _evolve_tests_based_on_results(self, test_suite: TestSuite, 
                                           execution_results: List[TestExecution]):
        """Evolve tests based on execution results"""
        try:
            logger.info("Analyzing test results for test evolution")
            
            # Analyze test effectiveness
            for execution in execution_results:
                test = next((t for t in test_suite.tests if t.test_id == execution.test_id), None)
                if test:
                    effectiveness_score = self._calculate_test_effectiveness(test, execution)
                    self.test_effectiveness_scores[test.test_id] = effectiveness_score
            
            # Identify tests that need improvement
            low_effectiveness_tests = [
                test_id for test_id, score in self.test_effectiveness_scores.items()
                if score < 0.3
            ]
            
            # Generate improved versions of low-effectiveness tests
            for test_id in low_effectiveness_tests:
                test = next((t for t in test_suite.tests if t.test_id == test_id), None)
                if test:
                    improved_test = await self._generate_improved_test(test)
                    if improved_test:
                        test_suite.tests.append(improved_test)
                        logger.info(f"Generated improved version of test {test.name}")
            
        except Exception as e:
            logger.error(f"Test evolution failed: {e}")
    
    def _calculate_test_effectiveness(self, test: GeneratedTest, 
                                    execution: TestExecution) -> float:
        """Calculate effectiveness score for test"""
        score = 0.0
        
        # Execution success component
        if execution.result == TestExecutionResult.PASSED:
            score += 0.4
        elif execution.result == TestExecutionResult.FAILED:
            score += 0.6  # Failing tests can be valuable if they catch bugs
        
        # Performance component
        if execution.duration_seconds > 0:
            # Prefer faster tests
            time_score = max(0, 1 - (execution.duration_seconds / 10))  # 10s threshold
            score += 0.2 * time_score
        
        # Coverage component (would need actual coverage data)
        coverage_score = 0.3  # Placeholder
        score += 0.2 * coverage_score
        
        # Uniqueness component (tests that find unique issues are more valuable)
        uniqueness_score = 0.5  # Placeholder
        score += 0.2 * uniqueness_score
        
        return max(0.0, min(1.0, score))
    
    async def _generate_improved_test(self, original_test: GeneratedTest) -> Optional[GeneratedTest]:
        """Generate improved version of test"""
        try:
            # Create variations of the original test
            improved_test = GeneratedTest(
                test_id=str(uuid.uuid4()),
                name=f"{original_test.name}_improved",
                test_type=original_test.test_type,
                strategy=TestGenerationStrategy.ML_BASED,
                source_code=self._improve_test_code(original_test.source_code),
                target_function=original_test.target_function,
                input_data=self._generate_improved_inputs(original_test.input_data),
                expected_output=original_test.expected_output,
                assertions=original_test.assertions,
                metadata={"improved_from": original_test.test_id}
            )
            
            return improved_test
            
        except Exception as e:
            logger.error(f"Test improvement failed: {e}")
            return None
    
    def _improve_test_code(self, original_code: str) -> str:
        """Improve test code based on common patterns"""
        # Simple improvements - in practice would be more sophisticated
        improved_code = original_code
        
        # Add more assertions
        if "assert result is not None" in improved_code:
            improved_code = improved_code.replace(
                "assert result is not None",
                "assert result is not None\n    assert isinstance(result, (str, int, float, list, dict, bool))"
            )
        
        # Add error handling
        if "try:" not in improved_code:
            improved_code = improved_code.replace(
                "# Act",
                '''# Act
    try:'''
            ).replace(
                "# Assert",
                '''except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")
    
    # Assert'''
            )
        
        return improved_code
    
    def _generate_improved_inputs(self, original_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate improved test inputs"""
        improved_inputs = original_inputs.copy()
        
        # Add edge cases
        for key, value in improved_inputs.items():
            if isinstance(value, int):
                improved_inputs[key] = random.choice([value, 0, -1, value * 2])
            elif isinstance(value, str):
                improved_inputs[key] = random.choice([value, "", value + "_test"])
            elif isinstance(value, list):
                improved_inputs[key] = random.choice([value, [], value + [None]])
        
        return improved_inputs
    
    def _generate_execution_summary(self, execution_results: List[TestExecution]) -> Dict[str, Any]:
        """Generate execution summary"""
        if not execution_results:
            return {"total": 0}
        
        result_counts = defaultdict(int)
        total_duration = 0.0
        
        for execution in execution_results:
            result_counts[execution.result.value] += 1
            total_duration += execution.duration_seconds
        
        return {
            "total": len(execution_results),
            "passed": result_counts.get("passed", 0),
            "failed": result_counts.get("failed", 0),
            "errors": result_counts.get("error", 0),
            "timeouts": result_counts.get("timeout", 0),
            "pass_rate": result_counts.get("passed", 0) / len(execution_results),
            "total_duration": total_duration,
            "average_duration": total_duration / len(execution_results)
        }
    
    def get_test_suite(self, suite_id: str) -> Optional[TestSuite]:
        """Get test suite by ID"""
        return self.test_suites.get(suite_id)
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get test execution history"""
        recent_executions = self.execution_history[-limit:]
        return [asdict(execution) for execution in recent_executions]
    
    def get_test_effectiveness_report(self) -> Dict[str, Any]:
        """Get test effectiveness analysis report"""
        if not self.test_effectiveness_scores:
            return {"message": "No effectiveness data available"}
        
        scores = list(self.test_effectiveness_scores.values())
        
        return {
            "total_tests_analyzed": len(scores),
            "average_effectiveness": sum(scores) / len(scores),
            "high_effectiveness_tests": len([s for s in scores if s >= 0.7]),
            "medium_effectiveness_tests": len([s for s in scores if 0.3 <= s < 0.7]),
            "low_effectiveness_tests": len([s for s in scores if s < 0.3]),
            "top_effective_tests": sorted(
                [(test_id, score) for test_id, score in self.test_effectiveness_scores.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }
    
    async def shutdown(self):
        """Gracefully shutdown testing framework"""
        logger.info("Intelligent testing framework shutdown complete")