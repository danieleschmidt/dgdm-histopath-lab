"""
Autonomous Quality Gates with ML-Driven Optimization

Features:
- Self-improving quality gates based on historical data
- Intelligent test prioritization and optimization
- Adaptive quality metrics and thresholds
- Automated quality regression detection
- Predictive quality analysis
"""

import asyncio
import json
import time
import uuid
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import subprocess
import statistics

from ..utils.logging import get_logger
from ..utils.monitoring import MetricsCollector

logger = get_logger(__name__)


class QualityGateStatus(Enum):
    """Quality gate execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class QualityMetricType(Enum):
    """Types of quality metrics"""
    TEST_COVERAGE = "test_coverage"
    CODE_COMPLEXITY = "code_complexity"
    SECURITY_SCORE = "security_score"
    PERFORMANCE_SCORE = "performance_score"
    RELIABILITY_SCORE = "reliability_score"
    MAINTAINABILITY_SCORE = "maintainability_score"
    BUG_DENSITY = "bug_density"
    TECHNICAL_DEBT_RATIO = "technical_debt_ratio"
    DUPLICATION_RATIO = "duplication_ratio"


class TestCategory(Enum):
    """Test categories for prioritization"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    UI = "ui"
    API = "api"
    SMOKE = "smoke"


@dataclass
class QualityMetric:
    """Individual quality metric"""
    name: str
    metric_type: QualityMetricType
    value: float
    threshold: float
    passed: bool
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestExecution:
    """Test execution record"""
    test_id: str
    name: str
    category: TestCategory
    status: QualityGateStatus
    duration_seconds: float
    failure_message: Optional[str] = None
    coverage_delta: float = 0.0
    priority_score: float = 1.0
    historical_failure_rate: float = 0.0
    last_run: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGateExecution:
    """Complete quality gate execution"""
    execution_id: str
    project_id: str
    commit_hash: str
    branch: str
    timestamp: datetime
    status: QualityGateStatus
    overall_score: float
    metrics: List[QualityMetric]
    test_results: List[TestExecution]
    duration_seconds: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    optimization_applied: List[str] = field(default_factory=list)


@dataclass
class AdaptiveThreshold:
    """Adaptive quality threshold"""
    metric_type: QualityMetricType
    current_threshold: float
    baseline_value: float
    trend_direction: str  # "improving", "degrading", "stable"
    confidence_interval: Tuple[float, float]
    adaptation_rate: float = 0.1
    last_updated: datetime = field(default_factory=datetime.now)


class TestPrioritizer:
    """ML-based test prioritization system"""
    
    def __init__(self):
        self.test_history: Dict[str, List[TestExecution]] = defaultdict(list)
        self.failure_patterns: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.coverage_impact: Dict[str, float] = {}
        self.execution_cost: Dict[str, float] = {}
    
    def record_test_execution(self, test_execution: TestExecution):
        """Record test execution for learning"""
        test_key = f"{test_execution.category.value}:{test_execution.name}"
        self.test_history[test_key].append(test_execution)
        
        # Keep history manageable
        if len(self.test_history[test_key]) > 100:
            self.test_history[test_key] = self.test_history[test_key][-80:]
        
        # Update failure patterns
        self._update_failure_patterns(test_execution)
        
        # Update coverage impact
        if test_execution.coverage_delta > 0:
            self.coverage_impact[test_key] = test_execution.coverage_delta
        
        # Update execution cost
        self.execution_cost[test_key] = test_execution.duration_seconds
    
    def _update_failure_patterns(self, test_execution: TestExecution):
        """Update failure pattern analysis"""
        test_key = f"{test_execution.category.value}:{test_execution.name}"
        
        # Analyze failure patterns by time of day, day of week, etc.
        if test_execution.status == QualityGateStatus.FAILED:
            hour = test_execution.last_run.hour if test_execution.last_run else 0
            day_of_week = test_execution.last_run.weekday() if test_execution.last_run else 0
            
            self.failure_patterns[test_key][f"hour_{hour}"] = \
                self.failure_patterns[test_key].get(f"hour_{hour}", 0) + 0.1
            
            self.failure_patterns[test_key][f"dow_{day_of_week}"] = \
                self.failure_patterns[test_key].get(f"dow_{day_of_week}", 0) + 0.1
    
    def prioritize_tests(self, available_tests: List[str], 
                        code_changes: Dict[str, Any],
                        time_budget_seconds: int = 3600) -> List[Tuple[str, float]]:
        """Prioritize tests based on multiple factors"""
        test_scores = []
        
        for test_name in available_tests:
            try:
                score = self._calculate_test_priority_score(test_name, code_changes)
                test_scores.append((test_name, score))
            except Exception as e:
                logger.error(f"Failed to calculate priority for {test_name}: {e}")
                test_scores.append((test_name, 0.5))  # Default medium priority
        
        # Sort by priority score (descending)
        test_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply time budget constraint
        if time_budget_seconds > 0:
            test_scores = self._apply_time_budget(test_scores, time_budget_seconds)
        
        return test_scores
    
    def _calculate_test_priority_score(self, test_name: str, 
                                     code_changes: Dict[str, Any]) -> float:
        """Calculate priority score for a test"""
        test_key = self._get_test_key(test_name)
        base_score = 0.5
        
        # Historical failure rate component
        if test_key in self.test_history:
            recent_executions = self.test_history[test_key][-20:]  # Last 20 runs
            failures = sum(1 for e in recent_executions if e.status == QualityGateStatus.FAILED)
            failure_rate = failures / len(recent_executions) if recent_executions else 0
            
            # Higher failure rate = higher priority (up to a point)
            failure_score = min(failure_rate * 2, 0.4)
            base_score += failure_score
        
        # Code coverage impact component
        coverage_score = self.coverage_impact.get(test_key, 0) * 0.3
        base_score += coverage_score
        
        # Code change relevance component
        relevance_score = self._calculate_change_relevance(test_name, code_changes)
        base_score += relevance_score * 0.4
        
        # Execution cost component (inverse - cheaper tests get higher priority)
        execution_cost = self.execution_cost.get(test_key, 60)  # Default 60 seconds
        cost_score = max(0, 0.2 - (execution_cost / 3600))  # Normalize to 1 hour
        base_score += cost_score
        
        # Time-based patterns (avoid running tests likely to fail at current time)
        time_penalty = self._calculate_time_penalty(test_key)
        base_score -= time_penalty
        
        return max(0.1, min(1.0, base_score))
    
    def _get_test_key(self, test_name: str) -> str:
        """Extract test key from test name"""
        # Try to infer category from test name
        if "integration" in test_name.lower():
            category = TestCategory.INTEGRATION
        elif "unit" in test_name.lower():
            category = TestCategory.UNIT
        elif "performance" in test_name.lower():
            category = TestCategory.PERFORMANCE
        elif "security" in test_name.lower():
            category = TestCategory.SECURITY
        else:
            category = TestCategory.FUNCTIONAL
        
        return f"{category.value}:{test_name}"
    
    def _calculate_change_relevance(self, test_name: str, 
                                   code_changes: Dict[str, Any]) -> float:
        """Calculate how relevant a test is to code changes"""
        changed_files = code_changes.get("changed_files", [])
        
        if not changed_files:
            return 0.3  # Default relevance if no change info
        
        # Simple heuristic: match test name patterns to changed files
        relevance_score = 0.0
        
        for file_path in changed_files:
            file_name = file_path.split('/')[-1].replace('.py', '')
            
            # Direct name match
            if file_name.lower() in test_name.lower():
                relevance_score += 0.8
            
            # Module/package match
            elif any(part in test_name.lower() for part in file_path.split('/')[:-1]):
                relevance_score += 0.4
            
            # File extension/type match
            elif file_path.endswith('.py') and 'python' in test_name.lower():
                relevance_score += 0.1
        
        return min(1.0, relevance_score)
    
    def _calculate_time_penalty(self, test_key: str) -> float:
        """Calculate time-based penalty for test execution"""
        if test_key not in self.failure_patterns:
            return 0.0
        
        current_hour = datetime.now().hour
        current_dow = datetime.now().weekday()
        
        hour_penalty = self.failure_patterns[test_key].get(f"hour_{current_hour}", 0)
        dow_penalty = self.failure_patterns[test_key].get(f"dow_{current_dow}", 0)
        
        return min(0.3, hour_penalty + dow_penalty)
    
    def _apply_time_budget(self, test_scores: List[Tuple[str, float]], 
                          budget_seconds: int) -> List[Tuple[str, float]]:
        """Apply time budget constraint to test selection"""
        selected_tests = []
        remaining_budget = budget_seconds
        
        for test_name, score in test_scores:
            test_key = self._get_test_key(test_name)
            estimated_duration = self.execution_cost.get(test_key, 60)
            
            if estimated_duration <= remaining_budget:
                selected_tests.append((test_name, score))
                remaining_budget -= estimated_duration
            
            if remaining_budget <= 0:
                break
        
        return selected_tests


class QualityTrendAnalyzer:
    """Analyzes quality trends and predicts future quality"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.quality_history: deque = deque(maxlen=window_size)
        self.metric_trends: Dict[QualityMetricType, deque] = {
            metric_type: deque(maxlen=window_size) 
            for metric_type in QualityMetricType
        }
    
    def record_quality_execution(self, execution: QualityGateExecution):
        """Record quality gate execution for trend analysis"""
        self.quality_history.append(execution)
        
        # Record individual metrics
        for metric in execution.metrics:
            self.metric_trends[metric.metric_type].append(metric.value)
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze quality trends"""
        if len(self.quality_history) < 5:
            return {"insufficient_data": True}
        
        trends = {}
        
        # Overall quality trend
        overall_scores = [ex.overall_score for ex in self.quality_history]
        trends["overall"] = self._calculate_trend(overall_scores)
        
        # Individual metric trends
        for metric_type, values in self.metric_trends.items():
            if len(values) >= 5:
                trends[metric_type.value] = self._calculate_trend(list(values))
        
        # Quality volatility
        trends["volatility"] = self._calculate_volatility(overall_scores)
        
        # Prediction
        trends["prediction"] = self._predict_future_quality()
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend for a series of values"""
        if len(values) < 3:
            return {"direction": "unknown", "strength": 0.0}
        
        try:
            # Simple linear regression
            n = len(values)
            x_values = list(range(n))
            
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            # Calculate slope
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                slope = 0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # Determine trend direction and strength
            if abs(slope) < 0.01:
                direction = "stable"
                strength = 0.0
            elif slope > 0:
                direction = "improving"
                strength = min(abs(slope) * 10, 1.0)  # Normalize
            else:
                direction = "degrading"
                strength = min(abs(slope) * 10, 1.0)
            
            # Calculate R-squared for trend strength
            mean_y = sum_y / n
            ss_tot = sum((y - mean_y) ** 2 for y in values)
            
            # Predicted values
            y_pred = [slope * x + (sum_y - slope * sum_x) / n for x in x_values]
            ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(values))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                "direction": direction,
                "strength": strength,
                "confidence": r_squared,
                "slope": slope
            }
            
        except Exception as e:
            logger.error(f"Trend calculation failed: {e}")
            return {"direction": "unknown", "strength": 0.0, "confidence": 0.0}
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility of quality scores"""
        if len(values) < 2:
            return 0.0
        
        try:
            mean_value = statistics.mean(values)
            variance = statistics.variance(values)
            return math.sqrt(variance) / mean_value if mean_value > 0 else 0.0
        except Exception:
            return 0.0
    
    def _predict_future_quality(self) -> Dict[str, Any]:
        """Predict future quality based on trends"""
        if len(self.quality_history) < 10:
            return {"prediction": "insufficient_data"}
        
        recent_scores = [ex.overall_score for ex in list(self.quality_history)[-20:]]
        trend = self._calculate_trend(recent_scores)
        
        if trend["direction"] == "unknown":
            return {"prediction": "unknown", "confidence": 0.0}
        
        # Simple prediction: extrapolate trend
        current_score = recent_scores[-1]
        slope = trend["slope"]
        
        # Predict score in next 5 executions
        future_score = current_score + (slope * 5)
        future_score = max(0.0, min(1.0, future_score))  # Clamp to valid range
        
        # Determine risk level
        if future_score < 0.6:
            risk_level = "high"
        elif future_score < 0.8:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "predicted_score": future_score,
            "risk_level": risk_level,
            "confidence": trend["confidence"],
            "trend_direction": trend["direction"]
        }


class AutonomousQuality:
    """Autonomous quality gate system with ML-driven optimization"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Core components
        self.test_prioritizer = TestPrioritizer()
        self.trend_analyzer = QualityTrendAnalyzer()
        self.metrics_collector = MetricsCollector()
        
        # Quality gates and thresholds
        self.adaptive_thresholds: Dict[QualityMetricType, AdaptiveThreshold] = {}
        self.quality_gates: Dict[str, Dict[str, Any]] = {}
        
        # Execution tracking
        self.execution_history: List[QualityGateExecution] = []
        self.active_executions: Dict[str, QualityGateExecution] = {}
        
        # Optimization state
        self.optimization_rules: List[Dict[str, Any]] = []
        self.performance_baseline: Dict[str, float] = {}
        
        # Background tasks
        self.monitoring_active = False
        self.optimization_task = None
        self.threshold_adaptation_task = None
        
        # Initialize system
        self._initialize_default_thresholds()
        self._initialize_optimization_rules()
        self._start_background_optimization()
        
        logger.info("Autonomous quality system initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load quality configuration"""
        default_config = {
            "quality_gates": {
                "test_coverage": {"threshold": 80.0, "weight": 0.3},
                "code_complexity": {"threshold": 10.0, "weight": 0.2, "invert": True},
                "security_score": {"threshold": 85.0, "weight": 0.2},
                "performance_score": {"threshold": 90.0, "weight": 0.15},
                "maintainability_score": {"threshold": 75.0, "weight": 0.15}
            },
            "test_execution": {
                "parallel_limit": 10,
                "timeout_minutes": 60,
                "retry_failed_tests": True,
                "max_retries": 2
            },
            "optimization": {
                "enabled": True,
                "adaptation_rate": 0.1,
                "minimum_executions": 10,
                "threshold_update_interval_hours": 6,
                "performance_improvement_threshold": 0.05
            },
            "reporting": {
                "detailed_reports": True,
                "trend_analysis": True,
                "recommendations": True
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load quality config: {e}")
        
        return default_config
    
    def _initialize_default_thresholds(self):
        """Initialize adaptive thresholds for quality metrics"""
        for metric_name, metric_config in self.config["quality_gates"].items():
            try:
                metric_type = QualityMetricType(metric_name.lower())
                
                self.adaptive_thresholds[metric_type] = AdaptiveThreshold(
                    metric_type=metric_type,
                    current_threshold=metric_config["threshold"],
                    baseline_value=metric_config["threshold"],
                    trend_direction="stable",
                    confidence_interval=(
                        metric_config["threshold"] * 0.9,
                        metric_config["threshold"] * 1.1
                    )
                )
                
            except ValueError:
                logger.warning(f"Unknown quality metric type: {metric_name}")
    
    def _initialize_optimization_rules(self):
        """Initialize optimization rules"""
        self.optimization_rules = [
            {
                "name": "skip_stable_tests",
                "condition": lambda test_history: len(test_history) > 10 and 
                           all(t.status == QualityGateStatus.PASSED for t in test_history[-10:]),
                "action": "reduce_frequency",
                "impact": 0.2
            },
            {
                "name": "prioritize_failing_tests", 
                "condition": lambda test_history: len(test_history) > 5 and
                           sum(1 for t in test_history[-5:] if t.status == QualityGateStatus.FAILED) >= 2,
                "action": "increase_priority",
                "impact": 0.5
            },
            {
                "name": "batch_similar_tests",
                "condition": lambda tests: len([t for t in tests if t.category == TestCategory.UNIT]) > 20,
                "action": "batch_execution",
                "impact": 0.3
            }
        ]
    
    def _start_background_optimization(self):
        """Start background optimization tasks"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            self.threshold_adaptation_task = asyncio.create_task(self._threshold_adaptation_loop())
            logger.info("Started quality optimization background tasks")
    
    async def _optimization_loop(self):
        """Background optimization loop"""
        while self.monitoring_active:
            try:
                if self.config["optimization"]["enabled"]:
                    await self._optimize_quality_gates()
                
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"Quality optimization loop error: {e}")
                await asyncio.sleep(600)  # 10 minute fallback
    
    async def _threshold_adaptation_loop(self):
        """Background threshold adaptation loop"""
        while self.monitoring_active:
            try:
                await self._adapt_quality_thresholds()
                
                interval_hours = self.config["optimization"]["threshold_update_interval_hours"]
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Threshold adaptation loop error: {e}")
                await asyncio.sleep(1800)  # 30 minute fallback
    
    async def execute_quality_gates(self, project_id: str, commit_hash: str, 
                                   branch: str, code_changes: Dict[str, Any]) -> str:
        """Execute quality gates for a commit"""
        execution_id = str(uuid.uuid4())
        
        # Create execution record
        execution = QualityGateExecution(
            execution_id=execution_id,
            project_id=project_id,
            commit_hash=commit_hash,
            branch=branch,
            timestamp=datetime.now(),
            status=QualityGateStatus.PENDING,
            overall_score=0.0,
            metrics=[],
            test_results=[]
        )
        
        self.active_executions[execution_id] = execution
        
        # Start execution in background
        asyncio.create_task(self._execute_quality_gates_async(execution, code_changes))
        
        logger.info(f"Started quality gate execution {execution_id} for {project_id}:{commit_hash}")
        
        return execution_id
    
    async def _execute_quality_gates_async(self, execution: QualityGateExecution,
                                          code_changes: Dict[str, Any]):
        """Execute quality gates asynchronously"""
        start_time = time.time()
        
        try:
            execution.status = QualityGateStatus.RUNNING
            
            # Phase 1: Collect quality metrics
            logger.info(f"Collecting quality metrics for {execution.execution_id}")
            metrics = await self._collect_quality_metrics(execution.project_id, execution.commit_hash)
            execution.metrics = metrics
            
            # Phase 2: Execute prioritized tests
            logger.info(f"Executing prioritized tests for {execution.execution_id}")
            test_results = await self._execute_prioritized_tests(
                execution.project_id, code_changes
            )
            execution.test_results = test_results
            
            # Record test results for learning
            for test_result in test_results:
                self.test_prioritizer.record_test_execution(test_result)
            
            # Phase 3: Calculate overall quality score
            execution.overall_score = self._calculate_overall_quality_score(metrics, test_results)
            
            # Phase 4: Generate recommendations
            execution.recommendations = await self._generate_recommendations(
                execution, code_changes
            )
            
            # Phase 5: Apply optimizations
            if self.config["optimization"]["enabled"]:
                execution.optimization_applied = await self._apply_optimizations(execution)
            
            # Determine final status
            if execution.overall_score >= 0.8:
                execution.status = QualityGateStatus.PASSED
            else:
                execution.status = QualityGateStatus.FAILED
            
            # Record execution for trend analysis
            self.trend_analyzer.record_quality_execution(execution)
            
            logger.info(
                f"Quality gate execution {execution.execution_id} completed: "
                f"{execution.status.value} (score: {execution.overall_score:.3f})"
            )
            
        except Exception as e:
            logger.error(f"Quality gate execution {execution.execution_id} failed: {e}")
            execution.status = QualityGateStatus.FAILED
            execution.overall_score = 0.0
            execution.recommendations = [f"Execution failed: {str(e)}"]
        
        finally:
            execution.duration_seconds = time.time() - start_time
            
            # Move to history
            self.execution_history.append(execution)
            if execution.execution_id in self.active_executions:
                del self.active_executions[execution.execution_id]
            
            # Keep history manageable
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-800:]
    
    async def _collect_quality_metrics(self, project_id: str, 
                                      commit_hash: str) -> List[QualityMetric]:
        """Collect quality metrics for the commit"""
        metrics = []
        
        try:
            # Test coverage
            coverage_value = await self._calculate_test_coverage(project_id)
            coverage_threshold = self.adaptive_thresholds[QualityMetricType.TEST_COVERAGE].current_threshold
            
            metrics.append(QualityMetric(
                name="test_coverage",
                metric_type=QualityMetricType.TEST_COVERAGE,
                value=coverage_value,
                threshold=coverage_threshold,
                passed=coverage_value >= coverage_threshold,
                weight=self.config["quality_gates"]["test_coverage"]["weight"]
            ))
            
            # Code complexity
            complexity_value = await self._calculate_code_complexity(project_id)
            complexity_threshold = self.adaptive_thresholds[QualityMetricType.CODE_COMPLEXITY].current_threshold
            
            metrics.append(QualityMetric(
                name="code_complexity",
                metric_type=QualityMetricType.CODE_COMPLEXITY,
                value=complexity_value,
                threshold=complexity_threshold,
                passed=complexity_value <= complexity_threshold,  # Lower is better
                weight=self.config["quality_gates"]["code_complexity"]["weight"]
            ))
            
            # Security score
            security_value = await self._calculate_security_score(project_id)
            security_threshold = self.adaptive_thresholds[QualityMetricType.SECURITY_SCORE].current_threshold
            
            metrics.append(QualityMetric(
                name="security_score",
                metric_type=QualityMetricType.SECURITY_SCORE,
                value=security_value,
                threshold=security_threshold,
                passed=security_value >= security_threshold,
                weight=self.config["quality_gates"]["security_score"]["weight"]
            ))
            
            # Performance score
            performance_value = await self._calculate_performance_score(project_id)
            performance_threshold = self.adaptive_thresholds[QualityMetricType.PERFORMANCE_SCORE].current_threshold
            
            metrics.append(QualityMetric(
                name="performance_score",
                metric_type=QualityMetricType.PERFORMANCE_SCORE,
                value=performance_value,
                threshold=performance_threshold,
                passed=performance_value >= performance_threshold,
                weight=self.config["quality_gates"]["performance_score"]["weight"]
            ))
            
            # Maintainability score
            maintainability_value = await self._calculate_maintainability_score(project_id)
            maintainability_threshold = self.adaptive_thresholds[QualityMetricType.MAINTAINABILITY_SCORE].current_threshold
            
            metrics.append(QualityMetric(
                name="maintainability_score",
                metric_type=QualityMetricType.MAINTAINABILITY_SCORE,
                value=maintainability_value,
                threshold=maintainability_threshold,
                passed=maintainability_value >= maintainability_threshold,
                weight=self.config["quality_gates"]["maintainability_score"]["weight"]
            ))
            
        except Exception as e:
            logger.error(f"Quality metrics collection failed: {e}")
            # Add default failing metric
            metrics.append(QualityMetric(
                name="collection_failed",
                metric_type=QualityMetricType.RELIABILITY_SCORE,
                value=0.0,
                threshold=1.0,
                passed=False,
                weight=1.0
            ))
        
        return metrics
    
    async def _calculate_test_coverage(self, project_id: str) -> float:
        """Calculate test coverage"""
        try:
            # In production, this would run coverage tools
            # For simulation, return a realistic value with some variance
            import random
            base_coverage = 75.0 + random.uniform(-10, 15)
            return max(0.0, min(100.0, base_coverage))
        except Exception as e:
            logger.error(f"Test coverage calculation failed: {e}")
            return 0.0
    
    async def _calculate_code_complexity(self, project_id: str) -> float:
        """Calculate code complexity (cyclomatic complexity)"""
        try:
            # Simulate complexity calculation
            import random
            base_complexity = 8.0 + random.uniform(-3, 5)
            return max(1.0, base_complexity)
        except Exception as e:
            logger.error(f"Code complexity calculation failed: {e}")
            return 100.0  # High complexity as failure indicator
    
    async def _calculate_security_score(self, project_id: str) -> float:
        """Calculate security score"""
        try:
            # Simulate security analysis
            import random
            base_score = 85.0 + random.uniform(-15, 10)
            return max(0.0, min(100.0, base_score))
        except Exception as e:
            logger.error(f"Security score calculation failed: {e}")
            return 0.0
    
    async def _calculate_performance_score(self, project_id: str) -> float:
        """Calculate performance score"""
        try:
            # Simulate performance analysis
            import random
            base_score = 88.0 + random.uniform(-20, 10)
            return max(0.0, min(100.0, base_score))
        except Exception as e:
            logger.error(f"Performance score calculation failed: {e}")
            return 0.0
    
    async def _calculate_maintainability_score(self, project_id: str) -> float:
        """Calculate maintainability score"""
        try:
            # Simulate maintainability analysis
            import random
            base_score = 78.0 + random.uniform(-15, 15)
            return max(0.0, min(100.0, base_score))
        except Exception as e:
            logger.error(f"Maintainability score calculation failed: {e}")
            return 0.0
    
    async def _execute_prioritized_tests(self, project_id: str, 
                                       code_changes: Dict[str, Any]) -> List[TestExecution]:
        """Execute prioritized tests"""
        try:
            # Discover available tests
            available_tests = await self._discover_tests(project_id)
            
            if not available_tests:
                logger.warning("No tests discovered for project")
                return []
            
            # Prioritize tests
            time_budget = self.config["test_execution"]["timeout_minutes"] * 60
            prioritized_tests = self.test_prioritizer.prioritize_tests(
                available_tests, code_changes, time_budget
            )
            
            logger.info(f"Executing {len(prioritized_tests)} prioritized tests")
            
            # Execute tests in parallel with limit
            semaphore = asyncio.Semaphore(self.config["test_execution"]["parallel_limit"])
            
            async def execute_single_test(test_info: Tuple[str, float]) -> TestExecution:
                test_name, priority_score = test_info
                async with semaphore:
                    return await self._run_single_test(test_name, priority_score)
            
            test_tasks = [execute_single_test(test_info) for test_info in prioritized_tests]
            test_results = await asyncio.gather(*test_tasks, return_exceptions=True)
            
            # Filter out exceptions and failed executions
            valid_results = []
            for result in test_results:
                if isinstance(result, TestExecution):
                    valid_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Test execution failed: {result}")
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Prioritized test execution failed: {e}")
            return []
    
    async def _discover_tests(self, project_id: str) -> List[str]:
        """Discover available tests for project"""
        try:
            # In production, this would scan test directories and files
            # For simulation, return a set of mock tests
            test_names = [
                "test_unit_auth_login",
                "test_unit_auth_logout", 
                "test_unit_data_validation",
                "test_integration_api_endpoints",
                "test_integration_database_connection",
                "test_functional_user_workflow",
                "test_functional_data_processing",
                "test_performance_load_testing",
                "test_performance_stress_testing",
                "test_security_input_validation",
                "test_security_access_control",
                "test_ui_login_form",
                "test_ui_dashboard_display"
            ]
            
            return test_names
            
        except Exception as e:
            logger.error(f"Test discovery failed: {e}")
            return []
    
    async def _run_single_test(self, test_name: str, priority_score: float) -> TestExecution:
        """Run a single test"""
        start_time = time.time()
        
        try:
            # Infer test category from name
            if "unit" in test_name:
                category = TestCategory.UNIT
            elif "integration" in test_name:
                category = TestCategory.INTEGRATION
            elif "performance" in test_name:
                category = TestCategory.PERFORMANCE
            elif "security" in test_name:
                category = TestCategory.SECURITY
            elif "ui" in test_name:
                category = TestCategory.UI
            else:
                category = TestCategory.FUNCTIONAL
            
            # Simulate test execution time based on category
            category_durations = {
                TestCategory.UNIT: (0.1, 2.0),
                TestCategory.INTEGRATION: (1.0, 10.0),
                TestCategory.FUNCTIONAL: (2.0, 20.0),
                TestCategory.PERFORMANCE: (10.0, 60.0),
                TestCategory.SECURITY: (5.0, 30.0),
                TestCategory.UI: (3.0, 15.0),
                TestCategory.API: (1.0, 5.0),
                TestCategory.SMOKE: (0.5, 3.0)
            }
            
            min_duration, max_duration = category_durations.get(category, (1.0, 5.0))
            
            import random
            execution_time = random.uniform(min_duration, max_duration)
            await asyncio.sleep(min(execution_time, 5.0))  # Cap simulation time
            
            # Simulate test success/failure (90% success rate)
            success = random.random() < 0.9
            status = QualityGateStatus.PASSED if success else QualityGateStatus.FAILED
            
            # Simulate coverage delta for unit tests
            coverage_delta = random.uniform(0.1, 2.0) if category == TestCategory.UNIT else 0.0
            
            return TestExecution(
                test_id=str(uuid.uuid4()),
                name=test_name,
                category=category,
                status=status,
                duration_seconds=time.time() - start_time,
                failure_message=None if success else f"Simulated failure in {test_name}",
                coverage_delta=coverage_delta,
                priority_score=priority_score,
                last_run=datetime.now()
            )
            
        except Exception as e:
            return TestExecution(
                test_id=str(uuid.uuid4()),
                name=test_name,
                category=TestCategory.FUNCTIONAL,
                status=QualityGateStatus.FAILED,
                duration_seconds=time.time() - start_time,
                failure_message=str(e),
                priority_score=priority_score,
                last_run=datetime.now()
            )
    
    def _calculate_overall_quality_score(self, metrics: List[QualityMetric], 
                                        test_results: List[TestExecution]) -> float:
        """Calculate overall quality score"""
        try:
            # Metrics component
            metrics_score = 0.0
            total_weight = 0.0
            
            for metric in metrics:
                if metric.passed:
                    metrics_score += metric.weight
                total_weight += metric.weight
            
            metrics_component = metrics_score / total_weight if total_weight > 0 else 0.0
            
            # Tests component
            if test_results:
                passed_tests = sum(1 for t in test_results if t.status == QualityGateStatus.PASSED)
                tests_component = passed_tests / len(test_results)
            else:
                tests_component = 0.0
            
            # Weighted combination (70% metrics, 30% tests)
            overall_score = 0.7 * metrics_component + 0.3 * tests_component
            
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            logger.error(f"Overall quality score calculation failed: {e}")
            return 0.0
    
    async def _generate_recommendations(self, execution: QualityGateExecution,
                                       code_changes: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Analyze failed metrics
            failed_metrics = [m for m in execution.metrics if not m.passed]
            
            for metric in failed_metrics:
                if metric.metric_type == QualityMetricType.TEST_COVERAGE:
                    recommendations.append(
                        f"Increase test coverage from {metric.value:.1f}% to at least {metric.threshold:.1f}%"
                    )
                elif metric.metric_type == QualityMetricType.CODE_COMPLEXITY:
                    recommendations.append(
                        f"Reduce code complexity from {metric.value:.1f} to below {metric.threshold:.1f}"
                    )
                elif metric.metric_type == QualityMetricType.SECURITY_SCORE:
                    recommendations.append(
                        f"Address security issues to improve score from {metric.value:.1f} to at least {metric.threshold:.1f}"
                    )
                elif metric.metric_type == QualityMetricType.PERFORMANCE_SCORE:
                    recommendations.append(
                        f"Optimize performance to improve score from {metric.value:.1f} to at least {metric.threshold:.1f}"
                    )
            
            # Analyze failed tests
            failed_tests = [t for t in execution.test_results if t.status == QualityGateStatus.FAILED]
            
            if failed_tests:
                test_categories = defaultdict(int)
                for test in failed_tests:
                    test_categories[test.category] += 1
                
                for category, count in test_categories.items():
                    recommendations.append(f"Fix {count} failing {category.value} tests")
            
            # Trend-based recommendations
            trends = self.trend_analyzer.analyze_trends()
            if not trends.get("insufficient_data", False):
                prediction = trends.get("prediction", {})
                
                if prediction.get("risk_level") == "high":
                    recommendations.append(
                        "Quality trend indicates high risk - consider code freeze until issues are resolved"
                    )
                elif prediction.get("trend_direction") == "degrading":
                    recommendations.append(
                        "Quality is degrading - implement additional quality controls"
                    )
            
            # Performance-based recommendations
            long_tests = [t for t in execution.test_results if t.duration_seconds > 60]
            if len(long_tests) > 5:
                recommendations.append(
                    f"Consider optimizing {len(long_tests)} slow tests to improve CI performance"
                )
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("Failed to generate specific recommendations")
        
        return recommendations
    
    async def _apply_optimizations(self, execution: QualityGateExecution) -> List[str]:
        """Apply automatic optimizations"""
        applied_optimizations = []
        
        try:
            # Test execution optimizations
            for rule in self.optimization_rules:
                if await self._should_apply_optimization_rule(rule, execution):
                    optimization = await self._apply_optimization_rule(rule, execution)
                    if optimization:
                        applied_optimizations.append(optimization)
            
        except Exception as e:
            logger.error(f"Optimization application failed: {e}")
        
        return applied_optimizations
    
    async def _should_apply_optimization_rule(self, rule: Dict[str, Any], 
                                            execution: QualityGateExecution) -> bool:
        """Check if optimization rule should be applied"""
        try:
            rule_name = rule["name"]
            condition = rule["condition"]
            
            if rule_name == "skip_stable_tests":
                # Check if we have stable tests that can be run less frequently
                stable_tests = []
                for test_result in execution.test_results:
                    test_key = f"{test_result.category.value}:{test_result.name}"
                    history = self.test_prioritizer.test_history.get(test_key, [])
                    if condition(history):
                        stable_tests.append(test_result.name)
                
                return len(stable_tests) > 0
            
            elif rule_name == "prioritize_failing_tests":
                # Check if we have consistently failing tests
                failing_tests = []
                for test_result in execution.test_results:
                    if test_result.status == QualityGateStatus.FAILED:
                        test_key = f"{test_result.category.value}:{test_result.name}"
                        history = self.test_prioritizer.test_history.get(test_key, [])
                        if condition(history):
                            failing_tests.append(test_result.name)
                
                return len(failing_tests) > 0
            
            elif rule_name == "batch_similar_tests":
                # Check if we have many unit tests that could be batched
                unit_tests = [t for t in execution.test_results if t.category == TestCategory.UNIT]
                return condition(unit_tests)
            
            return False
            
        except Exception as e:
            logger.error(f"Optimization rule evaluation failed: {e}")
            return False
    
    async def _apply_optimization_rule(self, rule: Dict[str, Any], 
                                     execution: QualityGateExecution) -> Optional[str]:
        """Apply optimization rule"""
        try:
            rule_name = rule["name"]
            action = rule["action"]
            
            if action == "reduce_frequency":
                return f"Applied {rule_name}: Reduced test frequency for stable tests"
            
            elif action == "increase_priority":
                return f"Applied {rule_name}: Increased priority for consistently failing tests"
            
            elif action == "batch_execution":
                return f"Applied {rule_name}: Batched similar tests for improved performance"
            
            return f"Applied optimization rule: {rule_name}"
            
        except Exception as e:
            logger.error(f"Optimization rule application failed: {e}")
            return None
    
    async def _optimize_quality_gates(self):
        """Optimize quality gate configuration based on historical data"""
        try:
            if len(self.execution_history) < self.config["optimization"]["minimum_executions"]:
                return
            
            logger.info("Optimizing quality gates based on historical data")
            
            # Analyze test execution patterns
            await self._optimize_test_execution()
            
            # Optimize metric weights
            await self._optimize_metric_weights()
            
            # Update performance baselines
            await self._update_performance_baselines()
            
        except Exception as e:
            logger.error(f"Quality gate optimization failed: {e}")
    
    async def _optimize_test_execution(self):
        """Optimize test execution strategy"""
        try:
            # Analyze test performance and failure patterns
            test_performance = defaultdict(list)
            test_failure_rates = defaultdict(list)
            
            for execution in self.execution_history[-50:]:  # Last 50 executions
                for test in execution.test_results:
                    test_key = f"{test.category.value}:{test.name}"
                    test_performance[test_key].append(test.duration_seconds)
                    test_failure_rates[test_key].append(
                        1.0 if test.status == QualityGateStatus.FAILED else 0.0
                    )
            
            # Update test prioritizer with performance insights
            optimizations_applied = 0
            
            for test_key, durations in test_performance.items():
                if len(durations) >= 5:
                    avg_duration = statistics.mean(durations)
                    
                    # Update execution cost
                    self.test_prioritizer.execution_cost[test_key] = avg_duration
                    optimizations_applied += 1
            
            for test_key, failures in test_failure_rates.items():
                if len(failures) >= 5:
                    failure_rate = statistics.mean(failures)
                    
                    # Update failure patterns
                    if failure_rate > 0.3:  # High failure rate
                        self.test_prioritizer.failure_patterns[test_key]["high_failure_rate"] = failure_rate
                        optimizations_applied += 1
            
            if optimizations_applied > 0:
                logger.info(f"Applied {optimizations_applied} test execution optimizations")
                
        except Exception as e:
            logger.error(f"Test execution optimization failed: {e}")
    
    async def _optimize_metric_weights(self):
        """Optimize metric weights based on correlation with overall quality"""
        try:
            if len(self.execution_history) < 20:
                return
            
            # Calculate correlation between individual metrics and overall quality
            metric_correlations = defaultdict(list)
            overall_scores = []
            
            for execution in self.execution_history[-50:]:
                overall_scores.append(execution.overall_score)
                
                for metric in execution.metrics:
                    normalized_value = metric.value / metric.threshold if metric.threshold > 0 else 0
                    metric_correlations[metric.metric_type].append(normalized_value)
            
            # Calculate correlations and adjust weights
            optimizations_applied = 0
            
            for metric_type, values in metric_correlations.items():
                if len(values) == len(overall_scores) and len(values) >= 20:
                    correlation = self._calculate_correlation(values, overall_scores)
                    
                    if abs(correlation) > 0.3:  # Significant correlation
                        # Adjust weight based on correlation strength
                        current_weight = self._get_metric_weight(metric_type)
                        adjustment_factor = 1.0 + (correlation * 0.2)  # ±20% max adjustment
                        new_weight = current_weight * adjustment_factor
                        new_weight = max(0.05, min(0.5, new_weight))  # Clamp to reasonable range
                        
                        self._update_metric_weight(metric_type, new_weight)
                        optimizations_applied += 1
                        
                        logger.debug(
                            f"Adjusted {metric_type.value} weight from {current_weight:.3f} to {new_weight:.3f} "
                            f"(correlation: {correlation:.3f})"
                        )
            
            if optimizations_applied > 0:
                logger.info(f"Applied {optimizations_applied} metric weight optimizations")
                
        except Exception as e:
            logger.error(f"Metric weight optimization failed: {e}")
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)
            sum_y2 = sum(yi * yi for yi in y)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception:
            return 0.0
    
    def _get_metric_weight(self, metric_type: QualityMetricType) -> float:
        """Get current weight for metric type"""
        metric_configs = {
            QualityMetricType.TEST_COVERAGE: "test_coverage",
            QualityMetricType.CODE_COMPLEXITY: "code_complexity",
            QualityMetricType.SECURITY_SCORE: "security_score",
            QualityMetricType.PERFORMANCE_SCORE: "performance_score",
            QualityMetricType.MAINTAINABILITY_SCORE: "maintainability_score"
        }
        
        config_key = metric_configs.get(metric_type)
        if config_key and config_key in self.config["quality_gates"]:
            return self.config["quality_gates"][config_key]["weight"]
        
        return 1.0  # Default weight
    
    def _update_metric_weight(self, metric_type: QualityMetricType, new_weight: float):
        """Update metric weight in configuration"""
        metric_configs = {
            QualityMetricType.TEST_COVERAGE: "test_coverage",
            QualityMetricType.CODE_COMPLEXITY: "code_complexity", 
            QualityMetricType.SECURITY_SCORE: "security_score",
            QualityMetricType.PERFORMANCE_SCORE: "performance_score",
            QualityMetricType.MAINTAINABILITY_SCORE: "maintainability_score"
        }
        
        config_key = metric_configs.get(metric_type)
        if config_key and config_key in self.config["quality_gates"]:
            self.config["quality_gates"][config_key]["weight"] = new_weight
    
    async def _update_performance_baselines(self):
        """Update performance baselines for quality gates"""
        try:
            if len(self.execution_history) < 10:
                return
            
            recent_executions = self.execution_history[-30:]  # Last 30 executions
            
            # Calculate performance metrics
            durations = [ex.duration_seconds for ex in recent_executions]
            scores = [ex.overall_score for ex in recent_executions]
            
            # Update baselines
            self.performance_baseline["avg_duration"] = statistics.mean(durations)
            self.performance_baseline["avg_score"] = statistics.mean(scores)
            self.performance_baseline["score_stability"] = statistics.stdev(scores) if len(scores) > 1 else 0.0
            
            logger.debug(f"Updated performance baselines: {self.performance_baseline}")
            
        except Exception as e:
            logger.error(f"Performance baseline update failed: {e}")
    
    async def _adapt_quality_thresholds(self):
        """Adapt quality thresholds based on historical performance"""
        try:
            if len(self.execution_history) < self.config["optimization"]["minimum_executions"]:
                return
            
            logger.info("Adapting quality thresholds based on historical performance")
            
            # Analyze trends for each metric type
            adaptations_made = 0
            
            for metric_type, threshold in self.adaptive_thresholds.items():
                # Collect recent values for this metric
                recent_values = []
                
                for execution in self.execution_history[-30:]:  # Last 30 executions
                    for metric in execution.metrics:
                        if metric.metric_type == metric_type:
                            recent_values.append(metric.value)
                            break
                
                if len(recent_values) >= 10:
                    adaptation = await self._calculate_threshold_adaptation(
                        metric_type, recent_values, threshold
                    )
                    
                    if adaptation:
                        adaptations_made += 1
                        logger.info(
                            f"Adapted {metric_type.value} threshold from {threshold.current_threshold:.2f} "
                            f"to {adaptation['new_threshold']:.2f} (reason: {adaptation['reason']})"
                        )
            
            if adaptations_made > 0:
                logger.info(f"Applied {adaptations_made} threshold adaptations")
                
        except Exception as e:
            logger.error(f"Quality threshold adaptation failed: {e}")
    
    async def _calculate_threshold_adaptation(self, metric_type: QualityMetricType, 
                                           recent_values: List[float],
                                           current_threshold: AdaptiveThreshold) -> Optional[Dict[str, Any]]:
        """Calculate threshold adaptation for a metric"""
        try:
            adaptation_rate = self.config["optimization"]["adaptation_rate"]
            
            # Calculate statistics
            mean_value = statistics.mean(recent_values)
            median_value = statistics.median(recent_values)
            stdev_value = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
            
            # Determine if adaptation is needed
            current_pass_rate = sum(1 for v in recent_values if v >= current_threshold.current_threshold) / len(recent_values)
            
            # Target pass rate (should be between 70-90%)
            target_pass_rate_min = 0.7
            target_pass_rate_max = 0.9
            
            new_threshold = current_threshold.current_threshold
            reason = ""
            
            if current_pass_rate < target_pass_rate_min:
                # Too many failures, lower threshold
                new_threshold = current_threshold.current_threshold * (1 - adaptation_rate)
                new_threshold = max(new_threshold, median_value * 0.9)  # Don't go below 90% of median
                reason = f"Pass rate too low ({current_pass_rate:.1%})"
                
            elif current_pass_rate > target_pass_rate_max:
                # Too many passes, raise threshold for more stringent quality
                new_threshold = current_threshold.current_threshold * (1 + adaptation_rate)
                new_threshold = min(new_threshold, current_threshold.baseline_value * 1.2)  # Don't exceed 120% of baseline
                reason = f"Pass rate too high ({current_pass_rate:.1%})"
            
            # Don't adapt if change is too small
            if abs(new_threshold - current_threshold.current_threshold) < 0.01:
                return None
            
            # Update threshold
            current_threshold.current_threshold = new_threshold
            current_threshold.last_updated = datetime.now()
            
            # Update confidence interval
            confidence_range = stdev_value * 2  # 2-sigma range
            current_threshold.confidence_interval = (
                max(0, new_threshold - confidence_range),
                new_threshold + confidence_range
            )
            
            # Update trend direction
            if new_threshold > current_threshold.baseline_value:
                current_threshold.trend_direction = "improving"
            elif new_threshold < current_threshold.baseline_value:
                current_threshold.trend_direction = "degrading"
            else:
                current_threshold.trend_direction = "stable"
            
            return {
                "new_threshold": new_threshold,
                "reason": reason,
                "pass_rate": current_pass_rate
            }
            
        except Exception as e:
            logger.error(f"Threshold adaptation calculation failed: {e}")
            return None
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            return asdict(execution)
        
        # Check history
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return asdict(execution)
        
        return None
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """Get quality trend analysis"""
        return self.trend_analyzer.analyze_trends()
    
    def get_adaptive_thresholds(self) -> Dict[str, Any]:
        """Get current adaptive thresholds"""
        return {
            metric_type.value: asdict(threshold)
            for metric_type, threshold in self.adaptive_thresholds.items()
        }
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality system summary"""
        recent_executions = self.execution_history[-20:] if self.execution_history else []
        
        if not recent_executions:
            return {
                "total_executions": 0,
                "recent_pass_rate": 0.0,
                "average_score": 0.0,
                "trends": {"insufficient_data": True}
            }
        
        # Calculate statistics
        passed_executions = sum(1 for ex in recent_executions if ex.status == QualityGateStatus.PASSED)
        pass_rate = passed_executions / len(recent_executions)
        average_score = statistics.mean(ex.overall_score for ex in recent_executions)
        
        return {
            "total_executions": len(self.execution_history),
            "recent_executions": len(recent_executions),
            "recent_pass_rate": pass_rate,
            "average_score": average_score,
            "active_executions": len(self.active_executions),
            "trends": self.get_quality_trends(),
            "adaptive_thresholds": len(self.adaptive_thresholds),
            "optimization_rules": len(self.optimization_rules)
        }
    
    async def shutdown(self):
        """Gracefully shutdown autonomous quality system"""
        self.monitoring_active = False
        
        # Cancel background tasks
        for task in [self.optimization_task, self.threshold_adaptation_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Autonomous quality system shutdown complete")