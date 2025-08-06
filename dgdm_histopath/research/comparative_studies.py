"""
Comparative Studies Framework for DGDM Research

Implements comprehensive benchmarking and statistical validation for
comparative studies against state-of-the-art baselines.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import time

try:
    from scipy import stats
    from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

from dgdm_histopath.utils.exceptions import DGDMException
from dgdm_histopath.utils.monitoring import metrics_collector


@dataclass
class BenchmarkResult:
    """Results from a single benchmark experiment."""
    model_name: str
    dataset_name: str
    metric_name: str
    score: float
    std_dev: float
    num_runs: int
    runtime_seconds: float
    memory_usage_mb: float
    hyperparameters: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ComparativeAnalysis:
    """Results from comparative statistical analysis."""
    primary_model: str
    baseline_models: List[str]
    metric: str
    primary_score: float
    baseline_scores: Dict[str, float]
    statistical_tests: Dict[str, Dict[str, float]]
    effect_sizes: Dict[str, float]
    significance_level: float
    conclusion: str
    confidence_interval: Tuple[float, float]
    

class BenchmarkSuite:
    """Comprehensive benchmarking suite for model evaluation."""
    
    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.logger = logging.getLogger("dgdm_histopath.research.benchmark")
        
        # Standard benchmarking datasets
        self.benchmark_datasets = {
            'TCGA-BRCA': {
                'task': 'molecular_subtyping',
                'num_samples': 1097,
                'num_classes': 4,
                'primary_metric': 'auc'
            },
            'CAMELYON16': {
                'task': 'metastasis_detection', 
                'num_samples': 400,
                'num_classes': 2,
                'primary_metric': 'auc'
            },
            'PANDA': {
                'task': 'gleason_grading',
                'num_samples': 10616,
                'num_classes': 6,
                'primary_metric': 'quadratic_kappa'
            },
            'BACH': {
                'task': 'breast_cancer_histology',
                'num_samples': 400,
                'num_classes': 4,
                'primary_metric': 'accuracy'
            }
        }
        
        # Standard baseline models
        self.baseline_models = {
            'ResNet50': {
                'type': 'cnn',
                'params': '25.6M',
                'paper': 'Deep Residual Learning for Image Recognition'
            },
            'ViT-Base': {
                'type': 'transformer',
                'params': '86M',
                'paper': 'An Image is Worth 16x16 Words'
            },
            'CTransPath': {
                'type': 'transformer',
                'params': '28.9M', 
                'paper': 'Transformer-based unsupervised contrastive learning'
            },
            'HIPT': {
                'type': 'hierarchical_transformer',
                'params': '122M',
                'paper': 'Hierarchical Image Pyramid Transformer'
            },
            'GraphSAGE': {
                'type': 'gnn',
                'params': '2.3M',
                'paper': 'Inductive Representation Learning on Large Graphs'
            },
            'GAT': {
                'type': 'gnn',
                'params': '1.8M',
                'paper': 'Graph Attention Networks'
            }
        }
    
    def run_comprehensive_benchmark(
        self,
        model_factory: Callable[[str], Any],
        datasets: Optional[List[str]] = None,
        baselines: Optional[List[str]] = None,
        num_runs: int = 5,
        cross_validation_folds: int = 5
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive benchmark across datasets and baselines.
        
        Args:
            model_factory: Function that creates model given name
            datasets: List of dataset names to benchmark on
            baselines: List of baseline model names to compare against
            num_runs: Number of runs for statistical significance
            cross_validation_folds: Number of CV folds
            
        Returns:
            Dictionary mapping dataset names to benchmark results
        """
        if datasets is None:
            datasets = list(self.benchmark_datasets.keys())
        
        if baselines is None:
            baselines = list(self.baseline_models.keys())
        
        all_results = {}
        
        self.logger.info(f"Starting comprehensive benchmark on {len(datasets)} datasets")
        
        for dataset_name in datasets:
            self.logger.info(f"Benchmarking on {dataset_name}")
            
            dataset_info = self.benchmark_datasets.get(dataset_name, {})
            dataset_results = []
            
            # Benchmark each baseline model
            for model_name in baselines:
                self.logger.info(f"  Running {model_name}")
                
                model_results = self._benchmark_model(
                    model_name=model_name,
                    model_factory=model_factory,
                    dataset_name=dataset_name,
                    dataset_info=dataset_info,
                    num_runs=num_runs,
                    cv_folds=cross_validation_folds
                )
                
                dataset_results.extend(model_results)
                
                # Record progress
                metrics_collector.record_custom_metric(
                    'benchmark_progress',
                    len(dataset_results),
                    tags={'dataset': dataset_name, 'model': model_name}
                )
            
            all_results[dataset_name] = dataset_results
        
        # Save results
        self._save_benchmark_results(all_results)
        
        self.logger.info("Comprehensive benchmark completed")
        return all_results
    
    def _benchmark_model(
        self,
        model_name: str,
        model_factory: Callable[[str], Any],
        dataset_name: str,
        dataset_info: Dict[str, Any],
        num_runs: int,
        cv_folds: int
    ) -> List[BenchmarkResult]:
        """Benchmark single model on dataset."""
        results = []
        
        # Standard metrics for different tasks
        task_metrics = {
            'molecular_subtyping': ['auc', 'accuracy', 'f1_macro'],
            'metastasis_detection': ['auc', 'accuracy', 'sensitivity', 'specificity'],
            'gleason_grading': ['quadratic_kappa', 'accuracy', 'mae'],
            'breast_cancer_histology': ['accuracy', 'f1_macro', 'precision_macro']
        }
        
        task = dataset_info.get('task', 'classification')
        metrics_to_evaluate = task_metrics.get(task, ['accuracy', 'auc'])
        
        # Run multiple iterations for statistical significance
        for run_idx in range(num_runs):
            run_start_time = time.time()
            
            try:
                # Create model (placeholder - would use actual factory)
                model = model_factory(model_name)
                
                # Run cross-validation (placeholder implementation)
                fold_results = self._run_cross_validation(
                    model, dataset_name, dataset_info, cv_folds
                )
                
                runtime = time.time() - run_start_time
                
                # Calculate metrics for each evaluated metric
                for metric_name in metrics_to_evaluate:
                    scores = [fold[metric_name] for fold in fold_results if metric_name in fold]
                    
                    if scores:
                        result = BenchmarkResult(
                            model_name=model_name,
                            dataset_name=dataset_name,
                            metric_name=metric_name,
                            score=np.mean(scores),
                            std_dev=np.std(scores),
                            num_runs=cv_folds,
                            runtime_seconds=runtime,
                            memory_usage_mb=self._estimate_memory_usage(model_name),
                            hyperparameters=self._get_model_hyperparameters(model_name),
                            timestamp=datetime.now()
                        )
                        
                        results.append(result)
                        self.results.append(result)
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {model_name} on {dataset_name}, run {run_idx}: {e}")
        
        return results
    
    def _run_cross_validation(
        self, 
        model: Any, 
        dataset_name: str, 
        dataset_info: Dict[str, Any],
        cv_folds: int
    ) -> List[Dict[str, float]]:
        """Run cross-validation for model evaluation."""
        # Placeholder implementation - would implement actual CV
        fold_results = []
        
        for fold in range(cv_folds):
            # Simulate different performance across folds
            base_performance = {
                'accuracy': np.random.normal(0.85, 0.05),
                'auc': np.random.normal(0.90, 0.03),
                'f1_macro': np.random.normal(0.82, 0.04),
                'sensitivity': np.random.normal(0.87, 0.06),
                'specificity': np.random.normal(0.89, 0.05),
                'quadratic_kappa': np.random.normal(0.78, 0.07),
                'mae': np.random.normal(0.15, 0.02),
                'precision_macro': np.random.normal(0.83, 0.05)
            }
            
            # Clip to valid ranges
            for metric in base_performance:
                if metric == 'mae':
                    base_performance[metric] = max(0, base_performance[metric])
                else:
                    base_performance[metric] = np.clip(base_performance[metric], 0, 1)
            
            fold_results.append(base_performance)
        
        return fold_results
    
    def _estimate_memory_usage(self, model_name: str) -> float:
        """Estimate memory usage for model."""
        # Placeholder estimation based on model type
        memory_estimates = {
            'ResNet50': 512,
            'ViT-Base': 1024,
            'CTransPath': 768,
            'HIPT': 1536,
            'GraphSAGE': 256,
            'GAT': 384,
            'DGDM': 896
        }
        
        return memory_estimates.get(model_name, 512)
    
    def _get_model_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """Get model hyperparameters."""
        # Placeholder hyperparameters
        hyperparams = {
            'ResNet50': {'learning_rate': 1e-4, 'batch_size': 32, 'epochs': 100},
            'ViT-Base': {'learning_rate': 1e-5, 'batch_size': 16, 'epochs': 50},
            'DGDM': {'learning_rate': 2e-4, 'batch_size': 4, 'epochs': 200, 'quantum_dim': 64}
        }
        
        return hyperparams.get(model_name, {'learning_rate': 1e-4, 'batch_size': 32})
    
    def _save_benchmark_results(self, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for dataset, dataset_results in results.items():
            serializable_results[dataset] = [result.to_dict() for result in dataset_results]
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {output_file}")


class ModelComparator:
    """Statistical comparison of model performances."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger("dgdm_histopath.research.comparator")
        
        if not SCIPY_AVAILABLE:
            self.logger.warning("SciPy not available - statistical tests will be limited")
    
    def compare_models(
        self,
        primary_results: List[BenchmarkResult],
        baseline_results: Dict[str, List[BenchmarkResult]],
        metric: str = 'auc'
    ) -> ComparativeAnalysis:
        """
        Perform comprehensive statistical comparison between models.
        
        Args:
            primary_results: Results for the primary model being evaluated
            baseline_results: Dict mapping baseline names to their results
            metric: Metric to compare on
            
        Returns:
            Comprehensive comparative analysis
        """
        # Extract scores for primary model
        primary_scores = [r.score for r in primary_results if r.metric_name == metric]
        primary_mean = np.mean(primary_scores) if primary_scores else 0.0
        
        # Extract scores for baselines
        baseline_scores = {}
        baseline_means = {}
        
        for baseline_name, results in baseline_results.items():
            scores = [r.score for r in results if r.metric_name == metric]
            baseline_scores[baseline_name] = scores
            baseline_means[baseline_name] = np.mean(scores) if scores else 0.0
        
        # Perform statistical tests
        statistical_tests = {}
        effect_sizes = {}
        
        for baseline_name, baseline_score_list in baseline_scores.items():
            if len(primary_scores) > 1 and len(baseline_score_list) > 1:
                tests = self._perform_statistical_tests(primary_scores, baseline_score_list)
                statistical_tests[baseline_name] = tests
                
                # Calculate effect size (Cohen's d)
                effect_size = self._calculate_cohens_d(primary_scores, baseline_score_list)
                effect_sizes[baseline_name] = effect_size
        
        # Calculate confidence interval for primary model
        confidence_interval = self._calculate_confidence_interval(primary_scores)
        
        # Generate conclusion
        conclusion = self._generate_conclusion(
            primary_mean, baseline_means, statistical_tests, effect_sizes
        )
        
        analysis = ComparativeAnalysis(
            primary_model="DGDM",
            baseline_models=list(baseline_results.keys()),
            metric=metric,
            primary_score=primary_mean,
            baseline_scores=baseline_means,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            significance_level=self.significance_level,
            conclusion=conclusion,
            confidence_interval=confidence_interval
        )
        
        return analysis
    
    def _perform_statistical_tests(
        self, 
        primary_scores: List[float], 
        baseline_scores: List[float]
    ) -> Dict[str, float]:
        """Perform various statistical significance tests."""
        tests = {}
        
        if not SCIPY_AVAILABLE:
            return {'error': 'SciPy not available for statistical tests'}
        
        try:
            # Two-sample t-test
            t_stat, t_pvalue = ttest_ind(primary_scores, baseline_scores)
            tests['t_test'] = {'statistic': t_stat, 'p_value': t_pvalue}
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = mannwhitneyu(
                primary_scores, baseline_scores, alternative='two-sided'
            )
            tests['mann_whitney'] = {'statistic': u_stat, 'p_value': u_pvalue}
            
            # Wilcoxon signed-rank test (if paired)
            if len(primary_scores) == len(baseline_scores):
                w_stat, w_pvalue = wilcoxon(primary_scores, baseline_scores)
                tests['wilcoxon'] = {'statistic': w_stat, 'p_value': w_pvalue}
        
        except Exception as e:
            self.logger.error(f"Statistical test failed: {e}")
            tests['error'] = str(e)
        
        return tests
    
    def _calculate_cohens_d(
        self, 
        primary_scores: List[float], 
        baseline_scores: List[float]
    ) -> float:
        """Calculate Cohen's d effect size."""
        try:
            mean_diff = np.mean(primary_scores) - np.mean(baseline_scores)
            pooled_std = np.sqrt(
                ((len(primary_scores) - 1) * np.var(primary_scores, ddof=1) +
                 (len(baseline_scores) - 1) * np.var(baseline_scores, ddof=1)) /
                (len(primary_scores) + len(baseline_scores) - 2)
            )
            
            return mean_diff / pooled_std if pooled_std > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Effect size calculation failed: {e}")
            return 0.0
    
    def _calculate_confidence_interval(
        self, 
        scores: List[float], 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        if len(scores) <= 1:
            mean_score = scores[0] if scores else 0.0
            return (mean_score, mean_score)
        
        try:
            if SCIPY_AVAILABLE:
                mean = np.mean(scores)
                sem = stats.sem(scores)  # Standard error of mean
                interval = stats.t.interval(
                    confidence, len(scores) - 1, loc=mean, scale=sem
                )
                return interval
            else:
                # Fallback without scipy
                mean = np.mean(scores)
                std = np.std(scores, ddof=1)
                margin = 1.96 * std / np.sqrt(len(scores))  # Approximate 95% CI
                return (mean - margin, mean + margin)
                
        except Exception as e:
            self.logger.error(f"Confidence interval calculation failed: {e}")
            mean_score = np.mean(scores)
            return (mean_score, mean_score)
    
    def _generate_conclusion(
        self,
        primary_mean: float,
        baseline_means: Dict[str, float],
        statistical_tests: Dict[str, Dict[str, Any]],
        effect_sizes: Dict[str, float]
    ) -> str:
        """Generate human-readable conclusion from statistical analysis."""
        conclusions = []
        
        # Compare performance
        better_than = []
        similar_to = []
        worse_than = []
        
        for baseline_name, baseline_mean in baseline_means.items():
            if primary_mean > baseline_mean:
                better_than.append(baseline_name)
            elif abs(primary_mean - baseline_mean) < 0.01:  # Practically equivalent
                similar_to.append(baseline_name)
            else:
                worse_than.append(baseline_name)
        
        # Performance comparison
        if better_than:
            conclusions.append(f"DGDM outperformed {', '.join(better_than)}")
        if similar_to:
            conclusions.append(f"DGDM achieved similar performance to {', '.join(similar_to)}")
        if worse_than:
            conclusions.append(f"DGDM underperformed compared to {', '.join(worse_than)}")
        
        # Statistical significance
        significant_improvements = []
        for baseline_name, tests in statistical_tests.items():
            if 't_test' in tests:
                p_value = tests['t_test']['p_value']
                if p_value < self.significance_level and primary_mean > baseline_means[baseline_name]:
                    significant_improvements.append(f"{baseline_name} (p={p_value:.4f})")
        
        if significant_improvements:
            conclusions.append(f"Statistically significant improvements over: {', '.join(significant_improvements)}")
        
        # Effect sizes
        large_effects = []
        for baseline_name, effect_size in effect_sizes.items():
            if abs(effect_size) > 0.8:  # Large effect size
                large_effects.append(f"{baseline_name} (Cohen's d={effect_size:.2f})")
        
        if large_effects:
            conclusions.append(f"Large practical effect sizes versus: {', '.join(large_effects)}")
        
        return ". ".join(conclusions) if conclusions else "No significant differences found."


class StatisticalValidator:
    """Validates statistical significance and reproducibility of results."""
    
    def __init__(self, min_sample_size: int = 30, min_effect_size: float = 0.5):
        self.min_sample_size = min_sample_size
        self.min_effect_size = min_effect_size
        self.logger = logging.getLogger("dgdm_histopath.research.validator")
    
    def validate_experimental_design(
        self,
        experiment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate experimental design for scientific rigor."""
        validation_results = {
            'valid': True,
            'issues': [],
            'recommendations': [],
            'power_analysis': {}
        }
        
        # Check sample size
        num_runs = experiment_config.get('num_runs', 1)
        if num_runs < self.min_sample_size:
            validation_results['issues'].append(
                f"Sample size ({num_runs}) below recommended minimum ({self.min_sample_size})"
            )
            validation_results['recommendations'].append(
                f"Increase num_runs to at least {self.min_sample_size} for reliable statistics"
            )
        
        # Check cross-validation
        cv_folds = experiment_config.get('cv_folds', 1)
        if cv_folds < 5:
            validation_results['issues'].append(
                f"Cross-validation folds ({cv_folds}) insufficient for robust evaluation"
            )
            validation_results['recommendations'].append(
                "Use at least 5-fold cross-validation for reliable performance estimates"
            )
        
        # Check baseline coverage
        baselines = experiment_config.get('baselines', [])
        recommended_baselines = ['ResNet50', 'ViT-Base', 'HIPT']
        missing_baselines = set(recommended_baselines) - set(baselines)
        
        if missing_baselines:
            validation_results['recommendations'].append(
                f"Consider adding standard baselines: {list(missing_baselines)}"
            )
        
        # Power analysis (simplified)
        if SCIPY_AVAILABLE:
            power = self._calculate_statistical_power(
                effect_size=self.min_effect_size,
                sample_size=num_runs,
                alpha=0.05
            )
            validation_results['power_analysis']['statistical_power'] = power
            
            if power < 0.8:
                validation_results['issues'].append(
                    f"Statistical power ({power:.3f}) below recommended 0.8"
                )
        
        # Set overall validity
        validation_results['valid'] = len(validation_results['issues']) == 0
        
        return validation_results
    
    def _calculate_statistical_power(
        self, 
        effect_size: float, 
        sample_size: int, 
        alpha: float = 0.05
    ) -> float:
        """Calculate statistical power for t-test."""
        if not SCIPY_AVAILABLE:
            return 0.0
        
        try:
            # Simplified power calculation for two-sample t-test
            from scipy.stats import norm
            
            # Critical value for two-tailed test
            z_alpha = norm.ppf(1 - alpha/2)
            
            # Non-centrality parameter
            delta = effect_size * np.sqrt(sample_size / 2)
            
            # Power calculation
            power = 1 - norm.cdf(z_alpha - delta) + norm.cdf(-z_alpha - delta)
            
            return power
            
        except Exception as e:
            self.logger.error(f"Power calculation failed: {e}")
            return 0.0
    
    def validate_reproducibility(
        self,
        results: List[BenchmarkResult],
        tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """Validate reproducibility of experimental results."""
        reproducibility = {
            'reproducible': True,
            'variance_analysis': {},
            'outliers': [],
            'recommendations': []
        }
        
        # Group results by model and metric
        grouped_results = {}
        for result in results:
            key = (result.model_name, result.metric_name)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result.score)
        
        # Analyze variance for each group
        for (model, metric), scores in grouped_results.items():
            if len(scores) > 1:
                cv = np.std(scores) / np.mean(scores)  # Coefficient of variation
                reproducibility['variance_analysis'][f"{model}_{metric}"] = {
                    'coefficient_of_variation': cv,
                    'acceptable': cv <= tolerance
                }
                
                if cv > tolerance:
                    reproducibility['reproducible'] = False
                    reproducibility['recommendations'].append(
                        f"High variance in {model} {metric} scores (CV={cv:.3f}). "
                        "Consider more stable training or additional runs."
                    )
                
                # Detect outliers (simple method)
                q75, q25 = np.percentile(scores, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                outlier_scores = [s for s in scores if s < lower_bound or s > upper_bound]
                if outlier_scores:
                    reproducibility['outliers'].append({
                        'model': model,
                        'metric': metric,
                        'outlier_scores': outlier_scores,
                        'num_outliers': len(outlier_scores),
                        'total_runs': len(scores)
                    })
        
        return reproducibility