"""Standardized metrics collection for all experiments."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score
from typing import Dict, Any, Union, List


def calculate_classification_metrics(y_true: Union[np.ndarray, List], 
                                   y_pred: Union[np.ndarray, List]) -> Dict[str, float]:
    """Calculate standardized classification metrics.
    
    Primary metric: accuracy
    Secondary metrics: precision, recall, f1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Primary metric
    accuracy = accuracy_score(y_true, y_pred)
    
    # Secondary metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    return {
        'primary_metric': 'accuracy',
        'primary_value': float(accuracy),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def calculate_regression_metrics(y_true: Union[np.ndarray, List], 
                               y_pred: Union[np.ndarray, List]) -> Dict[str, float]:
    """Calculate standardized regression metrics.
    
    Primary metric: mse
    Secondary metrics: r2
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Primary metric
    mse = mean_squared_error(y_true, y_pred)
    
    # Secondary metrics
    r2 = r2_score(y_true, y_pred)
    
    return {
        'primary_metric': 'mse',
        'primary_value': float(mse),
        'mse': float(mse),
        'r2': float(r2)
    }


def calculate_metrics(y_true: Union[np.ndarray, List], 
                     y_pred: Union[np.ndarray, List], 
                     task_type: str) -> Dict[str, Any]:
    """Calculate metrics based on task type.
    
    Args:
        y_true: Ground truth labels/values
        y_pred: Predicted labels/values
        task_type: Either 'classification' or 'regression'
    
    Returns:
        Standardized metrics dictionary
    """
    if task_type == 'classification':
        return calculate_classification_metrics(y_true, y_pred)
    elif task_type == 'regression':
        return calculate_regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'classification' or 'regression'")


def format_metrics_for_logging(metrics: Dict[str, Any]) -> str:
    """Format metrics for clean logging output."""
    primary = metrics['primary_metric']
    primary_val = metrics['primary_value']
    
    if primary == 'accuracy':
        return f"accuracy={primary_val:.3f}, f1={metrics.get('f1', 0):.3f}"
    else:  # mse
        return f"mse={primary_val:.4f}, r2={metrics.get('r2', 0):.3f}"


def extract_primary_metric(metrics: Dict[str, Any]) -> float:
    """Extract the primary metric value for comparison."""
    return metrics['primary_value']