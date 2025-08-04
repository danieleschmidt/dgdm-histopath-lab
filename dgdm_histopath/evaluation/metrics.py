"""
Evaluation metrics for histopathology analysis.

Implements specialized metrics for classification, regression, and survival analysis
tasks in digital pathology with DGDM models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = "weighted"
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_prob: Predicted probabilities (optional)
        class_names: Class names for labeling
        average: Averaging strategy for multi-class metrics
        
    Returns:
        Dictionary containing various classification metrics
    """
    
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Per-class metrics
    if len(np.unique(y_true)) > 2:  # Multi-class
        metrics["precision_per_class"] = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics["recall_per_class"] = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics["f1_per_class"] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # AUC metrics (require probabilities)
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                metrics["pr_auc"] = average_precision_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            else:  # Multi-class
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
                metrics["pr_auc"] = average_precision_score(y_true, y_prob, average=average)
        except ValueError:
            # Handle cases where AUC cannot be computed
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    
    # Class-specific metrics from confusion matrix
    if class_names:
        metrics["class_names"] = class_names
        
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multioutput: str = "uniform_average"
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        multioutput: How to handle multiple outputs
        
    Returns:
        Dictionary containing regression metrics
    """
    
    metrics = {}
    
    # Basic regression metrics
    metrics["mse"] = mean_squared_error(y_true, y_pred, multioutput=multioutput)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(y_true, y_pred, multioutput=multioutput)
    metrics["r2"] = r2_score(y_true, y_pred, multioutput=multioutput)
    
    # Additional metrics
    residuals = y_true - y_pred
    metrics["mean_residual"] = np.mean(residuals)
    metrics["std_residual"] = np.std(residuals)
    
    # Percentage errors
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    metrics["mape"] = mape
    
    return metrics


def compute_survival_metrics(
    y_true_time: np.ndarray,
    y_true_event: np.ndarray,
    y_pred_risk: np.ndarray
) -> Dict[str, float]:
    """
    Compute survival analysis metrics.
    
    Args:
        y_true_time: True survival times
        y_true_event: Event indicators (1=event, 0=censored)
        y_pred_risk: Predicted risk scores
        
    Returns:
        Dictionary containing survival metrics
    """
    
    metrics = {}
    
    try:
        from lifelines.utils import concordance_index
        
        # Concordance index (C-index)
        c_index = concordance_index(y_true_time, -y_pred_risk, y_true_event)
        metrics["c_index"] = c_index
        
    except ImportError:
        # Fallback implementation
        metrics["c_index"] = _compute_c_index_simple(y_true_time, y_true_event, y_pred_risk)
    
    return metrics


def _compute_c_index_simple(times, events, risk_scores):
    """Simple C-index implementation."""
    
    concordant = 0
    total = 0
    
    n = len(times)
    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 1 and times[i] < times[j]:
                # Patient i had event before patient j
                total += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
            elif events[j] == 1 and times[j] < times[i]:
                # Patient j had event before patient i  
                total += 1
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                    
    return concordant / total if total > 0 else 0.5


def compute_segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute segmentation metrics.
    
    Args:
        y_true: True segmentation masks
        y_pred: Predicted segmentation masks
        num_classes: Number of classes
        
    Returns:
        Dictionary containing segmentation metrics
    """
    
    metrics = {}
    
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Basic metrics
    metrics["pixel_accuracy"] = accuracy_score(y_true_flat, y_pred_flat)
    
    # IoU (Intersection over Union)
    if num_classes is None:
        num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))
        
    ious = []
    for class_id in range(num_classes):
        true_mask = (y_true_flat == class_id)
        pred_mask = (y_pred_flat == class_id)
        
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
            
        ious.append(iou)
        
    metrics["mean_iou"] = np.mean(ious)
    metrics["iou_per_class"] = np.array(ious)
    
    # Dice coefficient
    dice_scores = []
    for class_id in range(num_classes):
        true_mask = (y_true_flat == class_id)
        pred_mask = (y_pred_flat == class_id)
        
        intersection = np.logical_and(true_mask, pred_mask).sum()
        total = true_mask.sum() + pred_mask.sum()
        
        if total > 0:
            dice = 2 * intersection / total
        else:
            dice = 1.0 if intersection == 0 else 0.0
            
        dice_scores.append(dice)
        
    metrics["mean_dice"] = np.mean(dice_scores)
    metrics["dice_per_class"] = np.array(dice_scores)
    
    return metrics


def compute_graph_metrics(
    edge_true: np.ndarray,
    edge_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute graph reconstruction metrics.
    
    Args:
        edge_true: True edge existence (binary)
        edge_pred: Predicted edge probabilities
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary containing graph metrics
    """
    
    metrics = {}
    
    # Convert predictions to binary
    edge_pred_binary = (edge_pred > threshold).astype(int)
    
    # Basic metrics
    metrics["edge_accuracy"] = accuracy_score(edge_true, edge_pred_binary)
    metrics["edge_precision"] = precision_score(edge_true, edge_pred_binary, zero_division=0)
    metrics["edge_recall"] = recall_score(edge_true, edge_pred_binary, zero_division=0)
    metrics["edge_f1"] = f1_score(edge_true, edge_pred_binary, zero_division=0)
    
    # AUC for edge prediction
    try:
        metrics["edge_auc"] = roc_auc_score(edge_true, edge_pred)
    except ValueError:
        metrics["edge_auc"] = 0.5
        
    return metrics


def compute_clinical_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    task_type: str = "classification"
) -> Dict[str, float]:
    """
    Compute clinical-specific metrics for histopathology tasks.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        task_type: Type of clinical task
        
    Returns:
        Dictionary containing clinical metrics
    """
    
    metrics = {}
    
    if task_type == "classification":
        # Extract predictions and labels
        y_pred = [p.get("predicted_class", 0) for p in predictions]
        y_true = [g.get("label", 0) for g in ground_truth]
        y_prob = [p.get("classification_probs", [0.5, 0.5]) for p in predictions]
        
        # Compute standard classification metrics  
        classification_metrics = compute_classification_metrics(
            np.array(y_true), np.array(y_pred), np.array(y_prob)
        )
        metrics.update(classification_metrics)
        
    elif task_type == "survival":
        # Extract survival data
        times = [g.get("survival_time", 0) for g in ground_truth]
        events = [g.get("event", 0) for g in ground_truth]
        risks = [p.get("risk_score", 0.5) for p in predictions]
        
        # Compute survival metrics
        survival_metrics = compute_survival_metrics(
            np.array(times), np.array(events), np.array(risks)
        )
        metrics.update(survival_metrics)
        
    # Clinical-specific metrics
    if len(predictions) > 0 and "confidence" in predictions[0]:
        confidences = [p["confidence"] for p in predictions]
        metrics["mean_confidence"] = np.mean(confidences)
        metrics["std_confidence"] = np.std(confidences)
        
        # High-confidence accuracy
        high_conf_mask = np.array(confidences) > 0.8
        if np.sum(high_conf_mask) > 0:
            high_conf_acc = accuracy_score(
                np.array([g.get("label", 0) for g in ground_truth])[high_conf_mask],
                np.array([p.get("predicted_class", 0) for p in predictions])[high_conf_mask]
            )
            metrics["high_confidence_accuracy"] = high_conf_acc
            
    return metrics


def bootstrap_confidence_interval(
    metric_func,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    **metric_kwargs
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        metric_func: Metric function to evaluate
        y_true: True values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        **metric_kwargs: Additional arguments for metric function
        
    Returns:
        Tuple of (metric_value, lower_ci, upper_ci)
    """
    
    # Original metric
    original_metric = metric_func(y_true, y_pred, **metric_kwargs)
    
    # Bootstrap sampling
    n_samples = len(y_true)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        try:
            metric_boot = metric_func(y_true_boot, y_pred_boot, **metric_kwargs)
            bootstrap_metrics.append(metric_boot)
        except:
            continue
            
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(bootstrap_metrics, lower_percentile)
    upper_ci = np.percentile(bootstrap_metrics, upper_percentile)
    
    return original_metric, lower_ci, upper_ci