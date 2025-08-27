"""
Intelligent CI/CD System with ML-Driven Optimization

Features:
- Predictive build failure detection
- Intelligent test selection and prioritization
- Adaptive deployment strategies
- Code quality trend analysis
- Automated performance regression detection
"""

import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import subprocess
import pickle

from ..utils.logging import get_logger
from ..utils.monitoring import MetricsCollector

logger = get_logger(__name__)


@dataclass
class BuildMetrics:
    """Build execution metrics"""
    duration: float
    test_count: int
    failure_count: int
    coverage: float
    complexity: float
    timestamp: datetime
    commit_hash: str
    branch: str


@dataclass
class PredictionResult:
    """ML prediction result"""
    probability: float
    confidence: float
    reasons: List[str]
    recommendations: List[str]


class IntelligentCICD:
    """ML-powered CI/CD system with predictive capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.metrics_collector = MetricsCollector()
        self.build_history: List[BuildMetrics] = []
        self.model_cache = {}
        
        # ML Models for prediction
        self._initialize_models()
        
        logger.info("Intelligent CI/CD system initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with intelligent defaults"""
        default_config = {
            "prediction_threshold": 0.7,
            "history_window_days": 30,
            "min_samples_for_prediction": 10,
            "test_selection_strategy": "intelligent",
            "deployment_strategy": "adaptive",
            "rollback_threshold": 0.95,
            "performance_regression_threshold": 0.15
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def _initialize_models(self):
        """Initialize ML models for various predictions"""
        try:
            # Simple models using sklearn-style interface
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            self.failure_predictor = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            self.performance_predictor = LogisticRegression(random_state=42)
            self.test_prioritizer = RandomForestClassifier(
                n_estimators=50, random_state=42
            )
            
        except ImportError:
            logger.warning("ML dependencies not available, using heuristic models")
            self.failure_predictor = None
            self.performance_predictor = None
            self.test_prioritizer = None
    
    async def predict_build_failure(self, code_changes: Dict[str, Any]) -> PredictionResult:
        """Predict likelihood of build failure based on code changes"""
        try:
            # Extract features from code changes
            features = self._extract_change_features(code_changes)
            
            if self.failure_predictor and len(self.build_history) > 10:
                # Use ML model if available and trained
                prob = self.failure_predictor.predict_proba([features])[0][1]
                confidence = max(self.failure_predictor.predict_proba([features])[0])
            else:
                # Use heuristic model
                prob, confidence = self._heuristic_failure_prediction(features)
            
            # Generate reasons and recommendations
            reasons = self._analyze_failure_reasons(features, code_changes)
            recommendations = self._generate_recommendations(reasons, prob)
            
            return PredictionResult(
                probability=prob,
                confidence=confidence,
                reasons=reasons,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Build failure prediction failed: {e}")
            return PredictionResult(0.5, 0.0, ["Prediction unavailable"], [])
    
    def _extract_change_features(self, code_changes: Dict[str, Any]) -> List[float]:
        """Extract numerical features from code changes"""
        features = []
        
        # File change metrics
        features.append(code_changes.get('files_changed', 0))
        features.append(code_changes.get('lines_added', 0))
        features.append(code_changes.get('lines_deleted', 0))
        features.append(code_changes.get('complexity_delta', 0))
        
        # Change type indicators
        features.append(1.0 if code_changes.get('has_test_changes', False) else 0.0)
        features.append(1.0 if code_changes.get('has_config_changes', False) else 0.0)
        features.append(1.0 if code_changes.get('has_dependency_changes', False) else 0.0)
        features.append(1.0 if code_changes.get('has_core_changes', False) else 0.0)
        
        # Temporal features
        features.append(code_changes.get('hour_of_day', 12) / 24.0)
        features.append(code_changes.get('day_of_week', 3) / 7.0)
        
        # Developer experience (if available)
        features.append(min(code_changes.get('developer_experience_months', 12), 60) / 60.0)
        
        return features
    
    def _heuristic_failure_prediction(self, features: List[float]) -> Tuple[float, float]:
        """Heuristic-based failure prediction when ML model unavailable"""
        risk_score = 0.0
        
        # High risk factors
        if features[0] > 20:  # Many files changed
            risk_score += 0.3
        if features[1] > 1000:  # Many lines added
            risk_score += 0.2
        if features[6] > 0.5:  # Dependency changes
            risk_score += 0.3
        if features[7] > 0.5:  # Core changes
            risk_score += 0.2
        
        # Protective factors
        if features[4] > 0.5:  # Has test changes
            risk_score -= 0.1
        
        # Temporal risk (late night/weekend commits)
        if features[8] < 0.25 or features[8] > 0.75:  # Late night
            risk_score += 0.1
        if features[9] < 0.15 or features[9] > 0.85:  # Weekend
            risk_score += 0.1
        
        probability = min(max(risk_score, 0.0), 1.0)
        confidence = 0.6  # Moderate confidence for heuristics
        
        return probability, confidence
    
    def _analyze_failure_reasons(self, features: List[float], changes: Dict[str, Any]) -> List[str]:
        """Analyze potential reasons for build failure"""
        reasons = []
        
        if features[0] > 15:
            reasons.append("Large number of files modified")
        if features[6] > 0.5:
            reasons.append("Dependency changes detected")
        if features[7] > 0.5:
            reasons.append("Core system modifications")
        if features[4] == 0.0:
            reasons.append("No test changes accompanying code changes")
        if changes.get('has_breaking_changes', False):
            reasons.append("Potential breaking changes detected")
        
        return reasons if reasons else ["No specific risk factors identified"]
    
    def _generate_recommendations(self, reasons: List[str], probability: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if probability > 0.7:
            recommendations.append("Consider breaking changes into smaller commits")
            recommendations.append("Run extended test suite before merge")
        
        if "No test changes" in str(reasons):
            recommendations.append("Add comprehensive tests for modified functionality")
        
        if "Dependency changes" in str(reasons):
            recommendations.append("Verify compatibility matrix and run integration tests")
        
        if "Core system" in str(reasons):
            recommendations.append("Schedule deployment during maintenance window")
            recommendations.append("Prepare rollback plan before deployment")
        
        return recommendations if recommendations else ["Proceed with standard validation"]
    
    async def intelligent_test_selection(self, code_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Select and prioritize tests based on code changes"""
        try:
            # Analyze which tests are most likely to catch issues
            test_mapping = self._analyze_code_test_mapping(code_changes)
            
            # Prioritize tests based on:
            # 1. Direct coverage of changed code
            # 2. Historical failure correlation
            # 3. Test execution cost vs. value
            
            priority_tests = self._prioritize_tests(test_mapping, code_changes)
            
            return {
                "high_priority": priority_tests["high"],
                "medium_priority": priority_tests["medium"],
                "low_priority": priority_tests["low"],
                "skip_safe": priority_tests["skip"],
                "estimated_time_savings": priority_tests["time_savings"]
            }
            
        except Exception as e:
            logger.error(f"Test selection failed: {e}")
            return {"high_priority": [], "medium_priority": [], "low_priority": []}
    
    def _analyze_code_test_mapping(self, code_changes: Dict[str, Any]) -> Dict[str, List[str]]:
        """Map code changes to relevant tests"""
        mapping = {"direct": [], "indirect": [], "integration": []}
        
        changed_files = code_changes.get('changed_files', [])
        
        for file_path in changed_files:
            # Direct test mapping (e.g., module.py -> test_module.py)
            test_file = file_path.replace('.py', '_test.py').replace('src/', 'tests/')
            mapping["direct"].append(test_file)
            
            # Indirect mapping based on imports and dependencies
            if 'core/' in file_path:
                mapping["indirect"].extend(self._find_dependent_tests(file_path))
            
            # Integration tests for API/interface changes
            if any(keyword in file_path for keyword in ['api', 'interface', 'public']):
                mapping["integration"].extend(self._find_integration_tests())
        
        return mapping
    
    def _prioritize_tests(self, test_mapping: Dict[str, List[str]], 
                         changes: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize tests based on multiple factors"""
        all_tests = []
        all_tests.extend(test_mapping.get("direct", []))
        all_tests.extend(test_mapping.get("indirect", []))
        all_tests.extend(test_mapping.get("integration", []))
        
        # Score each test
        test_scores = {}
        for test in set(all_tests):
            score = self._calculate_test_score(test, test_mapping, changes)
            test_scores[test] = score
        
        # Sort by score and categorize
        sorted_tests = sorted(test_scores.items(), key=lambda x: x[1], reverse=True)
        
        total_tests = len(sorted_tests)
        high_cutoff = int(total_tests * 0.3)
        medium_cutoff = int(total_tests * 0.6)
        
        return {
            "high": [test for test, _ in sorted_tests[:high_cutoff]],
            "medium": [test for test, _ in sorted_tests[high_cutoff:medium_cutoff]],
            "low": [test for test, _ in sorted_tests[medium_cutoff:]],
            "skip": [],  # Tests safe to skip based on analysis
            "time_savings": self._estimate_time_savings(sorted_tests)
        }
    
    def _calculate_test_score(self, test: str, mapping: Dict[str, List[str]], 
                             changes: Dict[str, Any]) -> float:
        """Calculate priority score for a test"""
        score = 0.0
        
        # Direct coverage bonus
        if test in mapping.get("direct", []):
            score += 10.0
        
        # Historical failure correlation
        score += self._get_historical_failure_score(test)
        
        # Change magnitude bonus
        if changes.get('lines_changed', 0) > 100:
            score += 2.0
        
        # Core component bonus
        if any('core' in f for f in changes.get('changed_files', [])):
            score += 3.0
        
        return score
    
    def _get_historical_failure_score(self, test: str) -> float:
        """Get historical failure rate for test"""
        # Simplified - would use actual test history in production
        return np.random.uniform(0, 5)  # Placeholder
    
    def _estimate_time_savings(self, sorted_tests: List[Tuple[str, float]]) -> Dict[str, float]:
        """Estimate time savings from intelligent test selection"""
        total_time = len(sorted_tests) * 30  # Assume 30s per test average
        selected_time = len(sorted_tests[:int(len(sorted_tests) * 0.6)]) * 30
        
        return {
            "total_time_minutes": total_time / 60,
            "selected_time_minutes": selected_time / 60,
            "time_saved_minutes": (total_time - selected_time) / 60,
            "efficiency_gain_percent": ((total_time - selected_time) / total_time) * 100
        }
    
    def _find_dependent_tests(self, file_path: str) -> List[str]:
        """Find tests that depend on a given file"""
        # Simplified dependency analysis
        return [f"test_{Path(file_path).stem}_integration.py"]
    
    def _find_integration_tests(self) -> List[str]:
        """Find relevant integration tests"""
        return ["test_api_integration.py", "test_system_integration.py"]
    
    async def adaptive_deployment_strategy(self, deployment_context: Dict[str, Any]) -> Dict[str, Any]:
        """Choose optimal deployment strategy based on context"""
        try:
            risk_assessment = await self._assess_deployment_risk(deployment_context)
            
            if risk_assessment["risk_level"] == "high":
                strategy = self._high_risk_deployment_strategy()
            elif risk_assessment["risk_level"] == "medium":
                strategy = self._medium_risk_deployment_strategy()
            else:
                strategy = self._low_risk_deployment_strategy()
            
            strategy.update({
                "risk_assessment": risk_assessment,
                "monitoring_config": self._generate_monitoring_config(risk_assessment),
                "rollback_plan": self._generate_rollback_plan(risk_assessment)
            })
            
            return strategy
            
        except Exception as e:
            logger.error(f"Deployment strategy selection failed: {e}")
            return self._safe_default_strategy()
    
    async def _assess_deployment_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level for deployment"""
        risk_factors = []
        risk_score = 0.0
        
        # Change magnitude
        if context.get('lines_changed', 0) > 1000:
            risk_factors.append("Large code changes")
            risk_score += 0.3
        
        # System component changes
        if context.get('affects_core', False):
            risk_factors.append("Core system changes")
            risk_score += 0.4
        
        # Database migrations
        if context.get('has_migrations', False):
            risk_factors.append("Database schema changes")
            risk_score += 0.3
        
        # Time sensitivity
        if context.get('is_hotfix', False):
            risk_factors.append("Emergency hotfix deployment")
            risk_score += 0.2
        elif context.get('is_weekend', False):
            risk_factors.append("Weekend deployment")
            risk_score += 0.1
        
        # Historical performance
        recent_failures = context.get('recent_deployment_failures', 0)
        if recent_failures > 2:
            risk_factors.append(f"Recent deployment failures: {recent_failures}")
            risk_score += 0.2
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "assessment_time": datetime.now().isoformat()
        }
    
    def _high_risk_deployment_strategy(self) -> Dict[str, Any]:
        """High-risk deployment with maximum safety"""
        return {
            "strategy": "blue_green_with_canary",
            "canary_percentage": 1,
            "rollout_stages": [1, 5, 25, 50, 100],
            "stage_duration_minutes": 30,
            "automatic_rollback": True,
            "approval_required": True,
            "health_check_frequency": "30s",
            "success_threshold": 99.9
        }
    
    def _medium_risk_deployment_strategy(self) -> Dict[str, Any]:
        """Medium-risk deployment with balanced approach"""
        return {
            "strategy": "rolling_with_canary",
            "canary_percentage": 5,
            "rollout_stages": [5, 25, 75, 100],
            "stage_duration_minutes": 15,
            "automatic_rollback": True,
            "approval_required": False,
            "health_check_frequency": "60s",
            "success_threshold": 99.5
        }
    
    def _low_risk_deployment_strategy(self) -> Dict[str, Any]:
        """Low-risk deployment with fast rollout"""
        return {
            "strategy": "rolling_update",
            "canary_percentage": 10,
            "rollout_stages": [25, 75, 100],
            "stage_duration_minutes": 5,
            "automatic_rollback": True,
            "approval_required": False,
            "health_check_frequency": "120s",
            "success_threshold": 99.0
        }
    
    def _safe_default_strategy(self) -> Dict[str, Any]:
        """Safe default when risk assessment fails"""
        return {
            "strategy": "blue_green",
            "rollout_stages": [100],
            "automatic_rollback": True,
            "approval_required": True,
            "health_check_frequency": "30s",
            "success_threshold": 99.9
        }
    
    def _generate_monitoring_config(self, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate monitoring configuration based on risk"""
        base_metrics = ["error_rate", "response_time", "throughput", "cpu_usage", "memory_usage"]
        
        if risk_assessment["risk_level"] == "high":
            return {
                "metrics": base_metrics + ["database_connections", "queue_depth", "cache_hit_rate"],
                "alert_thresholds": {"error_rate": 0.001, "response_time": 500},
                "sampling_rate": "high"
            }
        else:
            return {
                "metrics": base_metrics,
                "alert_thresholds": {"error_rate": 0.01, "response_time": 1000},
                "sampling_rate": "normal"
            }
    
    def _generate_rollback_plan(self, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rollback plan based on risk assessment"""
        return {
            "automatic_triggers": {
                "error_rate_threshold": 0.05 if risk_assessment["risk_level"] == "high" else 0.1,
                "response_time_threshold": 2000,
                "success_rate_threshold": 95.0
            },
            "rollback_time_limit_minutes": 5,
            "validation_steps": [
                "Check service health",
                "Verify database consistency", 
                "Confirm traffic routing",
                "Validate critical user journeys"
            ]
        }
    
    async def continuous_optimization(self):
        """Continuously optimize CI/CD based on accumulated data"""
        try:
            # Analyze build history for patterns
            patterns = self._analyze_build_patterns()
            
            # Update prediction models
            if len(self.build_history) > 50:
                await self._retrain_models()
            
            # Optimize test selection
            test_optimization = self._optimize_test_selection()
            
            # Update deployment strategies
            deployment_optimization = self._optimize_deployment_strategies()
            
            logger.info(
                f"CI/CD optimization completed: "
                f"patterns={len(patterns)}, "
                f"test_optimization={test_optimization['improvement']:.2%}, "
                f"deployment_optimization={deployment_optimization['improvement']:.2%}"
            )
            
        except Exception as e:
            logger.error(f"Continuous optimization failed: {e}")
    
    def _analyze_build_patterns(self) -> List[Dict[str, Any]]:
        """Analyze historical build patterns"""
        patterns = []
        
        if len(self.build_history) < 10:
            return patterns
        
        # Time-based patterns
        failure_by_hour = {}
        for build in self.build_history:
            hour = build.timestamp.hour
            if hour not in failure_by_hour:
                failure_by_hour[hour] = {"total": 0, "failures": 0}
            failure_by_hour[hour]["total"] += 1
            if build.failure_count > 0:
                failure_by_hour[hour]["failures"] += 1
        
        for hour, data in failure_by_hour.items():
            if data["total"] > 5:  # Sufficient data
                failure_rate = data["failures"] / data["total"]
                if failure_rate > 0.3:  # High failure rate
                    patterns.append({
                        "type": "temporal",
                        "pattern": f"High failure rate at hour {hour}",
                        "failure_rate": failure_rate,
                        "recommendation": f"Extra validation for commits at hour {hour}"
                    })
        
        return patterns
    
    async def _retrain_models(self):
        """Retrain ML models with accumulated data"""
        if not self.failure_predictor or len(self.build_history) < 20:
            return
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            # Retrain failure predictor
            self.failure_predictor.fit(X, y)
            
            logger.info(f"Models retrained with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from build history"""
        X, y = [], []
        
        for build in self.build_history[-100:]:  # Last 100 builds
            # Extract features (would need to store more metadata in practice)
            features = [
                build.test_count,
                build.duration,
                build.coverage,
                build.complexity,
                build.timestamp.hour / 24.0,
                build.timestamp.weekday() / 7.0
            ]
            
            X.append(features)
            y.append(1 if build.failure_count > 0 else 0)
        
        return np.array(X), np.array(y)
    
    def _optimize_test_selection(self) -> Dict[str, Any]:
        """Optimize test selection strategies"""
        # Analyze test effectiveness vs. cost
        return {"improvement": 0.15}  # Placeholder
    
    def _optimize_deployment_strategies(self) -> Dict[str, Any]:
        """Optimize deployment strategies based on outcomes"""
        # Analyze deployment success rates and rollback frequencies
        return {"improvement": 0.10}  # Placeholder
    
    def record_build_result(self, build_result: Dict[str, Any]):
        """Record build result for learning"""
        try:
            metrics = BuildMetrics(
                duration=build_result.get('duration', 0),
                test_count=build_result.get('test_count', 0),
                failure_count=build_result.get('failure_count', 0),
                coverage=build_result.get('coverage', 0.0),
                complexity=build_result.get('complexity', 0.0),
                timestamp=datetime.now(),
                commit_hash=build_result.get('commit_hash', ''),
                branch=build_result.get('branch', 'main')
            )
            
            self.build_history.append(metrics)
            
            # Keep history manageable
            if len(self.build_history) > 1000:
                self.build_history = self.build_history[-800:]
            
            logger.debug(f"Build result recorded: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to record build result: {e}")