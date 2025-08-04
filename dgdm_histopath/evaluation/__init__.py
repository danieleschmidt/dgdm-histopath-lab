"""Evaluation and visualization modules."""

from dgdm_histopath.evaluation.predictor import DGDMPredictor
from dgdm_histopath.evaluation.visualizer import AttentionVisualizer
from dgdm_histopath.evaluation.metrics import compute_classification_metrics, compute_regression_metrics

__all__ = [
    "DGDMPredictor",
    "AttentionVisualizer", 
    "compute_classification_metrics",
    "compute_regression_metrics"
]