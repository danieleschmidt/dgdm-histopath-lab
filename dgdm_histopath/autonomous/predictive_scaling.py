"""
Predictive Scaling with ML-Driven Resource Allocation

Features:
- Time-series forecasting for resource demand
- Intelligent auto-scaling based on multiple signals
- Cost-optimized resource allocation
- Predictive capacity planning
- Multi-dimensional scaling (CPU, memory, GPU, storage)
"""

import asyncio
import json
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading

from ..utils.logging import get_logger
from ..utils.monitoring import MetricsCollector

logger = get_logger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"


class ResourceType(Enum):
    """Types of resources to scale"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    INSTANCES = "instances"
    BANDWIDTH = "bandwidth"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    resource_type: ResourceType
    current_usage: float
    current_capacity: float
    utilization_percent: float
    timestamp: datetime
    cost_per_unit: float = 0.0
    
    @property
    def usage_ratio(self) -> float:
        return self.current_usage / max(self.current_capacity, 1.0)


@dataclass
class ScalingPrediction:
    """Scaling prediction result"""
    resource_type: ResourceType
    predicted_usage: float
    predicted_time: datetime
    confidence: float
    recommended_action: ScalingAction
    recommended_capacity: float
    cost_impact: float
    reasoning: List[str]


@dataclass
class ScalingEvent:
    """Record of scaling event"""
    event_id: str
    timestamp: datetime
    resource_type: ResourceType
    action: ScalingAction
    from_capacity: float
    to_capacity: float
    trigger_reason: str
    success: bool
    duration_seconds: float
    cost_impact: float


class TimeSeriesPredictor:
    """Simple time series predictor for resource usage"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_points: deque = deque(maxlen=window_size)
        self.seasonal_patterns: Dict[str, List[float]] = {}
        
    def add_data_point(self, value: float, timestamp: datetime):
        """Add new data point"""
        self.data_points.append((timestamp, value))
        self._update_seasonal_patterns(value, timestamp)
    
    def _update_seasonal_patterns(self, value: float, timestamp: datetime):
        """Update seasonal patterns"""
        # Daily pattern (24 hours)
        hour = timestamp.hour
        if "daily" not in self.seasonal_patterns:
            self.seasonal_patterns["daily"] = [0.0] * 24
        
        # Exponential moving average for seasonal patterns
        alpha = 0.1
        self.seasonal_patterns["daily"][hour] = (
            alpha * value + (1 - alpha) * self.seasonal_patterns["daily"][hour]
        )
        
        # Weekly pattern (7 days)
        day_of_week = timestamp.weekday()
        if "weekly" not in self.seasonal_patterns:
            self.seasonal_patterns["weekly"] = [0.0] * 7
        
        self.seasonal_patterns["weekly"][day_of_week] = (
            alpha * value + (1 - alpha) * self.seasonal_patterns["weekly"][day_of_week]
        )
    
    def predict(self, forecast_minutes: int = 30) -> Tuple[List[float], float]:
        """Predict future values"""
        if len(self.data_points) < 10:
            # Not enough data, return current value
            current_value = self.data_points[-1][1] if self.data_points else 0.0
            return [current_value] * (forecast_minutes // 5), 0.5
        
        try:
            # Extract recent values and timestamps
            recent_data = list(self.data_points)
            values = [point[1] for point in recent_data]
            timestamps = [point[0] for point in recent_data]
            
            # Simple trend analysis
            trend = self._calculate_trend(values)
            
            # Seasonal component
            seasonal_component = self._get_seasonal_component(timestamps[-1], forecast_minutes)
            
            # Noise level (for confidence calculation)
            noise_level = self._calculate_noise_level(values)
            
            # Generate predictions
            predictions = []
            current_time = timestamps[-1]
            current_value = values[-1]
            
            for i in range(0, forecast_minutes, 5):  # 5-minute intervals
                future_time = current_time + timedelta(minutes=i + 5)
                
                # Trend component
                trend_component = trend * (i + 5) / 60.0  # Per hour
                
                # Seasonal component
                seasonal_factor = self._get_seasonal_factor(future_time)
                
                # Combine components
                predicted_value = max(0, current_value + trend_component + seasonal_factor)
                predictions.append(predicted_value)
            
            # Calculate confidence based on data consistency
            confidence = max(0.1, min(0.9, 1.0 - noise_level))
            
            return predictions, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            current_value = self.data_points[-1][1] if self.data_points else 0.0
            return [current_value] * (forecast_minutes // 5), 0.3
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values"""
        if len(values) < 5:
            return 0.0
        
        try:
            # Simple linear regression
            n = len(values)
            x_values = list(range(n))
            
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            # Calculate slope (trend)
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
            
        except Exception:
            return 0.0
    
    def _get_seasonal_component(self, current_time: datetime, forecast_minutes: int) -> float:
        """Get seasonal component for prediction"""
        # This is a placeholder - would be more sophisticated in practice
        return 0.0
    
    def _get_seasonal_factor(self, future_time: datetime) -> float:
        """Get seasonal factor for specific time"""
        factors = []
        
        # Daily seasonal factor
        if "daily" in self.seasonal_patterns:
            hour = future_time.hour
            daily_avg = sum(self.seasonal_patterns["daily"]) / 24
            daily_factor = self.seasonal_patterns["daily"][hour] - daily_avg
            factors.append(daily_factor * 0.3)  # Weight daily pattern
        
        # Weekly seasonal factor
        if "weekly" in self.seasonal_patterns:
            day = future_time.weekday()
            weekly_avg = sum(self.seasonal_patterns["weekly"]) / 7
            weekly_factor = self.seasonal_patterns["weekly"][day] - weekly_avg
            factors.append(weekly_factor * 0.2)  # Weight weekly pattern
        
        return sum(factors)
    
    def _calculate_noise_level(self, values: List[float]) -> float:
        """Calculate noise level in data for confidence estimation"""
        if len(values) < 5:
            return 0.5
        
        try:
            # Calculate standard deviation of residuals from trend
            trend = self._calculate_trend(values)
            x_values = list(range(len(values)))
            
            # Calculate trend line values
            trend_values = [trend * x + values[0] for x in x_values]
            
            # Calculate residuals
            residuals = [abs(actual - predicted) for actual, predicted in zip(values, trend_values)]
            
            # Standard deviation of residuals
            mean_residual = sum(residuals) / len(residuals)
            variance = sum((r - mean_residual) ** 2 for r in residuals) / len(residuals)
            std_dev = math.sqrt(variance)
            
            # Normalize by mean value
            mean_value = sum(values) / len(values)
            if mean_value == 0:
                return 0.5
            
            noise_level = std_dev / mean_value
            return min(noise_level, 1.0)
            
        except Exception:
            return 0.5


class PredictiveScaling:
    """Predictive auto-scaling system with ML-driven resource allocation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Resource tracking
        self.resource_metrics: Dict[ResourceType, ResourceMetrics] = {}
        self.predictors: Dict[ResourceType, TimeSeriesPredictor] = {}
        self.scaling_history: List[ScalingEvent] = []
        
        # Scaling state
        self.scaling_in_progress: Dict[ResourceType, bool] = defaultdict(bool)
        self.last_scaling_action: Dict[ResourceType, datetime] = {}
        self.cooldown_periods: Dict[ResourceType, int] = {
            ResourceType.CPU: 300,      # 5 minutes
            ResourceType.MEMORY: 300,   # 5 minutes
            ResourceType.GPU: 600,      # 10 minutes
            ResourceType.INSTANCES: 600, # 10 minutes
            ResourceType.STORAGE: 1800, # 30 minutes
            ResourceType.BANDWIDTH: 180 # 3 minutes
        }
        
        # Cost tracking
        self.cost_history: deque = deque(maxlen=1000)
        
        # Background tasks
        self.monitoring_active = False
        self.prediction_task = None
        self.optimization_task = None
        
        # Initialize predictors
        self._initialize_predictors()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Predictive scaling system initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with intelligent defaults"""
        default_config = {
            "prediction_window_minutes": 30,
            "prediction_interval_seconds": 60,
            "optimization_interval_minutes": 10,
            "target_utilization": {
                "cpu": 70.0,
                "memory": 75.0,
                "gpu": 80.0,
                "storage": 85.0,
                "instances": 70.0,
                "bandwidth": 80.0
            },
            "scaling_thresholds": {
                "scale_up": 85.0,
                "scale_down": 40.0
            },
            "max_scaling_factor": 2.0,
            "min_instances": 1,
            "max_instances": 100,
            "cost_optimization_enabled": True,
            "predictive_scaling_enabled": True,
            "emergency_scaling_enabled": True
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load scaling config: {e}")
        
        return default_config
    
    def _initialize_predictors(self):
        """Initialize predictors for each resource type"""
        for resource_type in ResourceType:
            self.predictors[resource_type] = TimeSeriesPredictor(
                window_size=self.config.get("predictor_window_size", 200)
            )
    
    def _start_background_tasks(self):
        """Start background prediction and optimization tasks"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.prediction_task = asyncio.create_task(self._prediction_loop())
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            logger.info("Started predictive scaling background tasks")
    
    async def _prediction_loop(self):
        """Main prediction loop"""
        while self.monitoring_active:
            try:
                await self._run_predictions()
                await asyncio.sleep(self.config["prediction_interval_seconds"])
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(10)
    
    async def _optimization_loop(self):
        """Cost optimization loop"""
        while self.monitoring_active:
            try:
                if self.config["cost_optimization_enabled"]:
                    await self._optimize_resource_allocation()
                
                await asyncio.sleep(self.config["optimization_interval_minutes"] * 60)
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(30)
    
    def update_resource_metrics(self, resource_type: ResourceType, 
                               current_usage: float, current_capacity: float,
                               cost_per_unit: float = 0.0):
        """Update current resource metrics"""
        try:
            utilization = (current_usage / max(current_capacity, 1.0)) * 100
            
            metrics = ResourceMetrics(
                resource_type=resource_type,
                current_usage=current_usage,
                current_capacity=current_capacity,
                utilization_percent=utilization,
                timestamp=datetime.now(),
                cost_per_unit=cost_per_unit
            )
            
            self.resource_metrics[resource_type] = metrics
            
            # Add to predictor
            self.predictors[resource_type].add_data_point(utilization, datetime.now())
            
            # Track cost
            self.cost_history.append({
                "timestamp": datetime.now(),
                "resource_type": resource_type.value,
                "usage": current_usage,
                "capacity": current_capacity,
                "cost": current_usage * cost_per_unit
            })
            
        except Exception as e:
            logger.error(f"Failed to update metrics for {resource_type}: {e}")
    
    async def _run_predictions(self):
        """Run predictions for all resource types"""
        predictions = {}
        
        for resource_type in ResourceType:
            if resource_type in self.resource_metrics:
                try:
                    prediction = await self._predict_resource_demand(resource_type)
                    predictions[resource_type] = prediction
                    
                    # Check if scaling action is needed
                    if prediction and prediction.recommended_action != ScalingAction.NO_ACTION:
                        await self._evaluate_scaling_action(prediction)
                        
                except Exception as e:
                    logger.error(f"Prediction failed for {resource_type}: {e}")
        
        # Log prediction summary
        if predictions:
            summary = {
                rt.value: pred.recommended_action.value 
                for rt, pred in predictions.items() 
                if pred
            }
            logger.debug(f"Scaling predictions: {summary}")
    
    async def _predict_resource_demand(self, resource_type: ResourceType) -> Optional[ScalingPrediction]:
        """Predict resource demand for specific resource type"""
        try:
            if resource_type not in self.resource_metrics:
                return None
            
            current_metrics = self.resource_metrics[resource_type]
            predictor = self.predictors[resource_type]
            
            # Generate predictions
            predictions, confidence = predictor.predict(
                self.config["prediction_window_minutes"]
            )
            
            if not predictions:
                return None
            
            # Find peak predicted utilization
            peak_utilization = max(predictions)
            peak_time_index = predictions.index(peak_utilization)
            peak_time = datetime.now() + timedelta(minutes=(peak_time_index + 1) * 5)
            
            # Determine recommended action
            action, new_capacity, reasoning = self._determine_scaling_action(
                resource_type, peak_utilization, current_metrics
            )
            
            # Calculate cost impact
            cost_impact = self._calculate_cost_impact(
                resource_type, current_metrics.current_capacity, new_capacity
            )
            
            return ScalingPrediction(
                resource_type=resource_type,
                predicted_usage=peak_utilization,
                predicted_time=peak_time,
                confidence=confidence,
                recommended_action=action,
                recommended_capacity=new_capacity,
                cost_impact=cost_impact,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Resource demand prediction failed for {resource_type}: {e}")
            return None
    
    def _determine_scaling_action(self, resource_type: ResourceType, 
                                 predicted_utilization: float,
                                 current_metrics: ResourceMetrics) -> Tuple[ScalingAction, float, List[str]]:
        """Determine appropriate scaling action"""
        reasoning = []
        current_capacity = current_metrics.current_capacity
        target_utilization = self.config["target_utilization"][resource_type.value]
        scale_up_threshold = self.config["scaling_thresholds"]["scale_up"]
        scale_down_threshold = self.config["scaling_thresholds"]["scale_down"]
        
        # Check if in cooldown period
        if self._in_cooldown_period(resource_type):
            reasoning.append("In cooldown period, no action taken")
            return ScalingAction.NO_ACTION, current_capacity, reasoning
        
        # Scale up logic
        if predicted_utilization > scale_up_threshold:
            reasoning.append(f"Predicted utilization {predicted_utilization:.1f}% > threshold {scale_up_threshold}%")
            
            # Calculate required capacity
            required_capacity = (current_metrics.current_usage / target_utilization) * 100
            scaling_factor = min(
                required_capacity / current_capacity,
                self.config["max_scaling_factor"]
            )
            
            new_capacity = current_capacity * scaling_factor
            
            # Check resource limits
            if resource_type == ResourceType.INSTANCES:
                new_capacity = min(new_capacity, self.config["max_instances"])
            
            if new_capacity > current_capacity * 1.1:  # At least 10% increase
                if resource_type == ResourceType.INSTANCES:
                    reasoning.append(f"Scale out from {current_capacity} to {new_capacity} instances")
                    return ScalingAction.SCALE_OUT, new_capacity, reasoning
                else:
                    reasoning.append(f"Scale up from {current_capacity} to {new_capacity}")
                    return ScalingAction.SCALE_UP, new_capacity, reasoning
        
        # Scale down logic
        elif predicted_utilization < scale_down_threshold:
            reasoning.append(f"Predicted utilization {predicted_utilization:.1f}% < threshold {scale_down_threshold}%")
            
            # Calculate optimal capacity
            optimal_capacity = (current_metrics.current_usage / target_utilization) * 100
            scaling_factor = max(
                optimal_capacity / current_capacity,
                1.0 / self.config["max_scaling_factor"]
            )
            
            new_capacity = current_capacity * scaling_factor
            
            # Check minimum limits
            if resource_type == ResourceType.INSTANCES:
                new_capacity = max(new_capacity, self.config["min_instances"])
            
            if new_capacity < current_capacity * 0.9:  # At least 10% decrease
                if resource_type == ResourceType.INSTANCES:
                    reasoning.append(f"Scale in from {current_capacity} to {new_capacity} instances")
                    return ScalingAction.SCALE_IN, new_capacity, reasoning
                else:
                    reasoning.append(f"Scale down from {current_capacity} to {new_capacity}")
                    return ScalingAction.SCALE_DOWN, new_capacity, reasoning
        
        reasoning.append(f"Current utilization {predicted_utilization:.1f}% within acceptable range")
        return ScalingAction.NO_ACTION, current_capacity, reasoning
    
    def _in_cooldown_period(self, resource_type: ResourceType) -> bool:
        """Check if resource is in cooldown period"""
        if resource_type not in self.last_scaling_action:
            return False
        
        last_action_time = self.last_scaling_action[resource_type]
        cooldown_seconds = self.cooldown_periods[resource_type]
        
        return (datetime.now() - last_action_time).total_seconds() < cooldown_seconds
    
    def _calculate_cost_impact(self, resource_type: ResourceType, 
                              current_capacity: float, new_capacity: float) -> float:
        """Calculate cost impact of scaling action"""
        try:
            if resource_type not in self.resource_metrics:
                return 0.0
            
            cost_per_unit = self.resource_metrics[resource_type].cost_per_unit
            capacity_change = new_capacity - current_capacity
            
            # Hourly cost impact
            hourly_cost_impact = capacity_change * cost_per_unit
            
            return hourly_cost_impact
            
        except Exception as e:
            logger.error(f"Cost impact calculation failed: {e}")
            return 0.0
    
    async def _evaluate_scaling_action(self, prediction: ScalingPrediction):
        """Evaluate and potentially execute scaling action"""
        try:
            resource_type = prediction.resource_type
            
            # Check if scaling is already in progress
            if self.scaling_in_progress[resource_type]:
                logger.debug(f"Scaling already in progress for {resource_type}")
                return
            
            # Emergency scaling (immediate high utilization)
            current_utilization = self.resource_metrics[resource_type].utilization_percent
            emergency_threshold = 95.0
            
            if (current_utilization > emergency_threshold and 
                self.config["emergency_scaling_enabled"]):
                logger.warning(f"Emergency scaling triggered for {resource_type}: {current_utilization:.1f}%")
                await self._execute_scaling_action(prediction, emergency=True)
                return
            
            # Regular predictive scaling
            if (prediction.confidence > 0.6 and 
                self.config["predictive_scaling_enabled"]):
                
                # Additional validation for cost-sensitive actions
                if prediction.cost_impact > 100:  # High cost impact
                    logger.info(f"High cost scaling action requires validation: {prediction.cost_impact:.2f}")
                    # In production, this might require approval or additional validation
                
                await self._execute_scaling_action(prediction)
                
        except Exception as e:
            logger.error(f"Scaling action evaluation failed: {e}")
    
    async def _execute_scaling_action(self, prediction: ScalingPrediction, emergency: bool = False):
        """Execute the scaling action"""
        try:
            resource_type = prediction.resource_type
            action = prediction.recommended_action
            
            if action == ScalingAction.NO_ACTION:
                return
            
            # Mark scaling in progress
            self.scaling_in_progress[resource_type] = True
            
            # Create scaling event record
            event = ScalingEvent(
                event_id=f"scale_{resource_type.value}_{int(time.time())}",
                timestamp=datetime.now(),
                resource_type=resource_type,
                action=action,
                from_capacity=self.resource_metrics[resource_type].current_capacity,
                to_capacity=prediction.recommended_capacity,
                trigger_reason=f"{'Emergency' if emergency else 'Predictive'}: {'; '.join(prediction.reasoning)}",
                success=False,
                duration_seconds=0.0,
                cost_impact=prediction.cost_impact
            )
            
            logger.info(
                f"Executing scaling action: {action.value} for {resource_type.value} "
                f"from {event.from_capacity} to {event.to_capacity}"
            )
            
            start_time = time.time()
            
            # Execute the actual scaling (delegated to platform-specific implementation)
            success = await self._perform_scaling(
                resource_type, action, event.to_capacity
            )
            
            # Update event record
            event.success = success
            event.duration_seconds = time.time() - start_time
            
            if success:
                # Update last scaling action time
                self.last_scaling_action[resource_type] = datetime.now()
                
                # Update current capacity
                self.resource_metrics[resource_type].current_capacity = event.to_capacity
                
                logger.info(f"Scaling action completed successfully in {event.duration_seconds:.1f}s")
            else:
                logger.error(f"Scaling action failed after {event.duration_seconds:.1f}s")
            
            # Record event
            self.scaling_history.append(event)
            
            # Keep history manageable
            if len(self.scaling_history) > 1000:
                self.scaling_history = self.scaling_history[-800:]
            
        except Exception as e:
            logger.error(f"Scaling action execution failed: {e}")
            success = False
            
        finally:
            # Clear scaling in progress flag
            self.scaling_in_progress[resource_type] = False
    
    async def _perform_scaling(self, resource_type: ResourceType, 
                             action: ScalingAction, new_capacity: float) -> bool:
        """Perform the actual scaling operation (platform-specific)"""
        try:
            # This is a simulation - in production, this would interface with
            # container orchestrators, cloud APIs, etc.
            
            logger.info(f"Simulating {action.value} for {resource_type.value} to {new_capacity}")
            
            # Simulate scaling time based on resource type
            scaling_delays = {
                ResourceType.CPU: 5,
                ResourceType.MEMORY: 5,
                ResourceType.GPU: 30,
                ResourceType.INSTANCES: 60,
                ResourceType.STORAGE: 120,
                ResourceType.BANDWIDTH: 10
            }
            
            await asyncio.sleep(scaling_delays.get(resource_type, 10))
            
            # Simulate success rate (95% success in simulation)
            import random
            return random.random() < 0.95
            
        except Exception as e:
            logger.error(f"Scaling operation failed: {e}")
            return False
    
    async def _optimize_resource_allocation(self):
        """Optimize resource allocation for cost efficiency"""
        try:
            # Analyze resource utilization patterns
            utilization_analysis = self._analyze_utilization_patterns()
            
            # Identify optimization opportunities
            opportunities = self._identify_optimization_opportunities(utilization_analysis)
            
            # Apply cost optimizations
            for opportunity in opportunities:
                await self._apply_optimization(opportunity)
            
            logger.info(f"Resource optimization completed: {len(opportunities)} opportunities processed")
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
    
    def _analyze_utilization_patterns(self) -> Dict[ResourceType, Dict[str, Any]]:
        """Analyze historical utilization patterns"""
        analysis = {}
        
        for resource_type, predictor in self.predictors.items():
            if len(predictor.data_points) < 20:
                continue
            
            # Extract utilization values
            values = [point[1] for point in list(predictor.data_points)]
            
            # Calculate statistics
            avg_utilization = sum(values) / len(values)
            max_utilization = max(values)
            min_utilization = min(values)
            
            # Calculate variance
            variance = sum((x - avg_utilization) ** 2 for x in values) / len(values)
            std_dev = math.sqrt(variance)
            
            analysis[resource_type] = {
                "avg_utilization": avg_utilization,
                "max_utilization": max_utilization,
                "min_utilization": min_utilization,
                "std_dev": std_dev,
                "variance": variance,
                "sample_count": len(values)
            }
        
        return analysis
    
    def _identify_optimization_opportunities(self, analysis: Dict[ResourceType, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities"""
        opportunities = []
        
        for resource_type, stats in analysis.items():
            # Over-provisioned resources (low average utilization)
            if stats["avg_utilization"] < 40 and stats["max_utilization"] < 70:
                opportunities.append({
                    "type": "over_provisioned",
                    "resource_type": resource_type,
                    "current_avg": stats["avg_utilization"],
                    "recommended_action": "downsize",
                    "potential_savings": self._estimate_savings(resource_type, 0.7)
                })
            
            # High variance (inefficient scaling)
            if stats["std_dev"] > 20 and stats["avg_utilization"] > 50:
                opportunities.append({
                    "type": "high_variance",
                    "resource_type": resource_type,
                    "variance": stats["std_dev"],
                    "recommended_action": "optimize_scaling_parameters",
                    "potential_savings": self._estimate_savings(resource_type, 0.85)
                })
            
            # Underutilized high-cost resources
            if (resource_type in [ResourceType.GPU, ResourceType.INSTANCES] and 
                stats["avg_utilization"] < 60):
                opportunities.append({
                    "type": "underutilized_expensive",
                    "resource_type": resource_type,
                    "current_avg": stats["avg_utilization"],
                    "recommended_action": "consolidate_or_downsize",
                    "potential_savings": self._estimate_savings(resource_type, 0.8)
                })
        
        return opportunities
    
    def _estimate_savings(self, resource_type: ResourceType, efficiency_factor: float) -> float:
        """Estimate potential cost savings from optimization"""
        try:
            if resource_type not in self.resource_metrics:
                return 0.0
            
            current_metrics = self.resource_metrics[resource_type]
            current_cost = current_metrics.current_capacity * current_metrics.cost_per_unit
            
            # Estimate optimized cost
            optimized_capacity = current_metrics.current_capacity * efficiency_factor
            optimized_cost = optimized_capacity * current_metrics.cost_per_unit
            
            # Monthly savings estimate
            monthly_savings = (current_cost - optimized_cost) * 24 * 30
            
            return max(0, monthly_savings)
            
        except Exception:
            return 0.0
    
    async def _apply_optimization(self, opportunity: Dict[str, Any]):
        """Apply cost optimization"""
        try:
            resource_type = opportunity["resource_type"]
            action_type = opportunity["recommended_action"]
            
            logger.info(f"Applying optimization: {action_type} for {resource_type.value}")
            
            if action_type == "downsize":
                # Create scaling prediction for downsizing
                current_metrics = self.resource_metrics[resource_type]
                new_capacity = current_metrics.current_capacity * 0.8
                
                prediction = ScalingPrediction(
                    resource_type=resource_type,
                    predicted_usage=opportunity["current_avg"],
                    predicted_time=datetime.now(),
                    confidence=0.8,
                    recommended_action=ScalingAction.SCALE_DOWN,
                    recommended_capacity=new_capacity,
                    cost_impact=-opportunity["potential_savings"] / 30,  # Daily savings
                    reasoning=["Cost optimization: over-provisioned resource"]
                )
                
                await self._execute_scaling_action(prediction)
            
            elif action_type == "optimize_scaling_parameters":
                # Adjust scaling thresholds for more efficient scaling
                self._optimize_scaling_parameters(resource_type)
            
            elif action_type == "consolidate_or_downsize":
                # Similar to downsize but more aggressive
                current_metrics = self.resource_metrics[resource_type]
                new_capacity = current_metrics.current_capacity * 0.7
                
                prediction = ScalingPrediction(
                    resource_type=resource_type,
                    predicted_usage=opportunity["current_avg"],
                    predicted_time=datetime.now(),
                    confidence=0.7,
                    recommended_action=ScalingAction.SCALE_DOWN,
                    recommended_capacity=new_capacity,
                    cost_impact=-opportunity["potential_savings"] / 30,
                    reasoning=["Cost optimization: underutilized expensive resource"]
                )
                
                await self._execute_scaling_action(prediction)
            
        except Exception as e:
            logger.error(f"Failed to apply optimization: {e}")
    
    def _optimize_scaling_parameters(self, resource_type: ResourceType):
        """Optimize scaling parameters for resource type"""
        # Adjust thresholds based on observed patterns
        if resource_type.value in self.config["target_utilization"]:
            current_target = self.config["target_utilization"][resource_type.value]
            
            # Make scaling more aggressive for high-variance resources
            self.config["target_utilization"][resource_type.value] = min(80.0, current_target + 5)
            
            logger.info(f"Optimized scaling parameters for {resource_type.value}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        status = {
            "resource_metrics": {},
            "scaling_in_progress": {},
            "last_scaling_actions": {},
            "active_predictions": {}
        }
        
        # Resource metrics
        for resource_type, metrics in self.resource_metrics.items():
            status["resource_metrics"][resource_type.value] = {
                "utilization_percent": metrics.utilization_percent,
                "current_usage": metrics.current_usage,
                "current_capacity": metrics.current_capacity,
                "cost_per_unit": metrics.cost_per_unit,
                "last_updated": metrics.timestamp.isoformat()
            }
        
        # Scaling status
        for resource_type, in_progress in self.scaling_in_progress.items():
            status["scaling_in_progress"][resource_type.value] = in_progress
        
        # Last actions
        for resource_type, timestamp in self.last_scaling_action.items():
            status["last_scaling_actions"][resource_type.value] = timestamp.isoformat()
        
        return status
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling history"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_events = [
            event for event in self.scaling_history
            if event.timestamp >= cutoff
        ]
        return [asdict(event) for event in recent_events]
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis"""
        if not self.cost_history:
            return {"total_cost": 0, "cost_by_resource": {}, "cost_trend": []}
        
        # Calculate total cost
        total_cost = sum(entry["cost"] for entry in self.cost_history)
        
        # Cost by resource type
        cost_by_resource = defaultdict(float)
        for entry in self.cost_history:
            cost_by_resource[entry["resource_type"]] += entry["cost"]
        
        # Cost trend (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        recent_costs = [
            entry for entry in self.cost_history
            if entry["timestamp"] >= cutoff
        ]
        
        return {
            "total_cost": total_cost,
            "cost_by_resource": dict(cost_by_resource),
            "cost_trend": recent_costs[-50:],  # Last 50 data points
            "average_hourly_cost": total_cost / max(len(self.cost_history) / 60, 1)
        }
    
    async def shutdown(self):
        """Gracefully shutdown predictive scaling system"""
        self.monitoring_active = False
        
        # Cancel background tasks
        for task in [self.prediction_task, self.optimization_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Predictive scaling system shutdown complete")