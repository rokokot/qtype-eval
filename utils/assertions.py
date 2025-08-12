# utils/assertions.py
"""
Custom assertion functions for TF-IDF integration testing.
Provides specialized assertions for matrices, metrics, and data validation.
"""

import numpy as np
import scipy.sparse
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def assert_valid_shape(
    obj: Any,
    expected_shape: Tuple[int, ...],
    name: str = "object"
):
    """
    Assert that an object has the expected shape.
    
    Args:
        obj: Object to check (array, matrix, etc.)
        expected_shape: Expected shape tuple
        name: Name of object for error messages
    """
    if not hasattr(obj, 'shape'):
        raise AssertionError(f"{name} does not have a 'shape' attribute")
    
    actual_shape = obj.shape
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
        )


def assert_sparse_matrix(
    matrix: Any,
    name: str = "matrix",
    min_sparsity: float = 0.5
):
    """
    Assert that a matrix is sparse and meets sparsity requirements.
    
    Args:
        matrix: Matrix to check
        name: Name for error messages
        min_sparsity: Minimum required sparsity (fraction of zeros)
    """
    # Check if it's a sparse matrix
    if not scipy.sparse.issparse(matrix):
        raise AssertionError(f"{name} is not a sparse matrix")
    
    # Check sparsity
    total_elements = np.prod(matrix.shape)
    non_zero_elements = matrix.nnz
    sparsity = 1 - (non_zero_elements / total_elements)
    
    if sparsity < min_sparsity:
        raise AssertionError(
            f"{name} sparsity too low: {sparsity:.2%} < {min_sparsity:.2%}"
        )


def assert_valid_metrics(
    metrics: Dict[str, Any],
    task_type: str,
    required_metrics: Optional[List[str]] = None,
    metric_ranges: Optional[Dict[str, Tuple[float, float]]] = None
):
    """
    Assert that metrics dictionary is valid for the given task type.
    
    Args:
        metrics: Metrics dictionary to validate
        task_type: 'classification' or 'regression'
        required_metrics: List of required metric names
        metric_ranges: Dictionary mapping metric names to (min, max) ranges
    """
    if not isinstance(metrics, dict):
        raise AssertionError(f"Metrics must be a dictionary, got {type(metrics)}")
    
    # Default required metrics based on task type
    if required_metrics is None:
        if task_type == 'classification':
            required_metrics = ['accuracy', 'f1']
        elif task_type == 'regression':
            required_metrics = ['mse', 'r2']
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    # Check required metrics exist
    for metric in required_metrics:
        if metric not in metrics:
            raise AssertionError(f"Missing required metric: {metric}")
    
    # Default metric ranges
    if metric_ranges is None:
        metric_ranges = {
            'accuracy': (0.0, 1.0),
            'f1': (0.0, 1.0),
            'precision': (0.0, 1.0),
            'recall': (0.0, 1.0),
            'mse': (0.0, float('inf')),
            'rmse': (0.0, float('inf')),
            'mae': (0.0, float('inf')),
            'r2': (float('-inf'), 1.0)  # R² can be negative
        }
    
    # Validate metric values
    for metric, value in metrics.items():
        if not isinstance(value, (int, float)):
            raise AssertionError(f"Metric {metric} must be numeric, got {type(value)}")
        
        if np.isnan(value):
            raise AssertionError(f"Metric {metric} is NaN")
        
        if np.isinf(value):
            raise AssertionError(f"Metric {metric} is infinite")
        
        # Check ranges if specified
        if metric in metric_ranges:
            min_val, max_val = metric_ranges[metric]
            if not (min_val <= value <= max_val):
                raise AssertionError(
                    f"Metric {metric} = {value} not in valid range [{min_val}, {max_val}]"
                )


def assert_classification_metrics(metrics: Dict[str, Any]):
    """Assert classification metrics are valid."""
    assert_valid_metrics(metrics, 'classification')


def assert_regression_metrics(metrics: Dict[str, Any]):
    """Assert regression metrics are valid."""
    assert_valid_metrics(metrics, 'regression')


def assert_matrix_properties(
    matrix: Any,
    expected_shape: Optional[Tuple[int, ...]] = None,
    is_sparse: Optional[bool] = None,
    min_sparsity: Optional[float] = None,
    dtype: Optional[type] = None,
    name: str = "matrix"
):
    """
    Assert multiple matrix properties at once.
    
    Args:
        matrix: Matrix to check
        expected_shape: Expected shape tuple
        is_sparse: Whether matrix should be sparse
        min_sparsity: Minimum sparsity if sparse
        dtype: Expected data type
        name: Name for error messages
    """
    # Check shape
    if expected_shape is not None:
        assert_valid_shape(matrix, expected_shape, name)
    
    # Check sparsity
    if is_sparse is not None:
        if is_sparse:
            if not scipy.sparse.issparse(matrix):
                raise AssertionError(f"{name} should be sparse but is not")
            if min_sparsity is not None:
                assert_sparse_matrix(matrix, name, min_sparsity)
        else:
            if scipy.sparse.issparse(matrix):
                raise AssertionError(f"{name} should not be sparse but is")
    
    # Check dtype
    if dtype is not None:
        if hasattr(matrix, 'dtype'):
            if matrix.dtype != dtype:
                raise AssertionError(f"{name} dtype mismatch: expected {dtype}, got {matrix.dtype}")
        else:
            raise AssertionError(f"{name} does not have dtype attribute")


def assert_dataset_consistency(
    dataset: Dict[str, List[Any]],
    expected_size: Optional[int] = None,
    required_fields: Optional[List[str]] = None
):
    """
    Assert dataset consistency across all fields.
    
    Args:
        dataset: Dataset dictionary
        expected_size: Expected number of samples
        required_fields: Required field names
    """
    if not isinstance(dataset, dict):
        raise AssertionError(f"Dataset must be a dictionary, got {type(dataset)}")
    
    if not dataset:
        raise AssertionError("Dataset is empty")
    
    # Check required fields
    if required_fields is not None:
        for field in required_fields:
            if field not in dataset:
                raise AssertionError(f"Missing required field: {field}")
    
    # Check size consistency
    sizes = [len(values) for values in dataset.values() if hasattr(values, '__len__')]
    
    if not sizes:
        raise AssertionError("No fields with length found in dataset")
    
    if len(set(sizes)) > 1:
        size_info = {field: len(values) for field, values in dataset.items()}
        raise AssertionError(f"Inconsistent field sizes: {size_info}")
    
    actual_size = sizes[0]
    
    if expected_size is not None and actual_size != expected_size:
        raise AssertionError(f"Dataset size mismatch: expected {expected_size}, got {actual_size}")


def assert_valid_labels(
    labels: Union[List, np.ndarray],
    task_type: str,
    expected_size: Optional[int] = None
):
    """
    Assert that labels are valid for the given task type.
    
    Args:
        labels: Labels to validate
        task_type: 'classification' or 'regression'
        expected_size: Expected number of labels
    """
    if not hasattr(labels, '__len__'):
        raise AssertionError("Labels must have length")
    
    if expected_size is not None and len(labels) != expected_size:
        raise AssertionError(f"Label size mismatch: expected {expected_size}, got {len(labels)}")
    
    if len(labels) == 0:
        raise AssertionError("Labels are empty")
    
    if task_type == 'classification':
        # Check for binary classification
        unique_labels = set(labels)
        if not unique_labels.issubset({0, 1}):
            raise AssertionError(f"Classification labels must be 0 or 1, got: {unique_labels}")
    
    elif task_type == 'regression':
        # Check that all labels are numeric
        for i, label in enumerate(labels):
            if not isinstance(label, (int, float, np.integer, np.floating)):
                raise AssertionError(f"Regression label at index {i} is not numeric: {label}")
            if np.isnan(label):
                raise AssertionError(f"Regression label at index {i} is NaN")
            if np.isinf(label):
                raise AssertionError(f"Regression label at index {i} is infinite")
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def assert_features_labels_match(
    features: Any,
    labels: Union[List, np.ndarray],
    name: str = "features/labels"
):
    """
    Assert that features and labels have matching number of samples.
    
    Args:
        features: Feature matrix
        labels: Labels array
        name: Name for error messages
    """
    if not hasattr(features, 'shape'):
        raise AssertionError(f"{name}: features do not have shape attribute")
    
    n_feature_samples = features.shape[0]
    n_label_samples = len(labels)
    
    if n_feature_samples != n_label_samples:
        raise AssertionError(
            f"{name}: feature samples ({n_feature_samples}) != label samples ({n_label_samples})"
        )


def assert_experiment_result_structure(
    result: Dict[str, Any],
    experiment_type: str = "tfidf"
):
    """
    Assert that experiment result has the expected structure.
    
    Args:
        result: Experiment result dictionary
        experiment_type: Type of experiment
    """
    if not isinstance(result, dict):
        raise AssertionError(f"Experiment result must be a dictionary, got {type(result)}")
    
    required_fields = [
        'experiment_name',
        'model_type',
        'task',
        'task_type',
        'test_metrics'
    ]
    
    for field in required_fields:
        if field not in result:
            raise AssertionError(f"Missing required field in experiment result: {field}")
    
    # Check test metrics
    test_metrics = result['test_metrics']
    task_type = result['task_type']
    assert_valid_metrics(test_metrics, task_type)
    
    # Check specific fields based on experiment type
    if experiment_type == "tfidf":
        tfidf_fields = ['languages', 'model_info']
        for field in tfidf_fields:
            if field not in result:
                raise AssertionError(f"Missing TF-IDF specific field: {field}")


def assert_model_compatibility(
    model_type: str,
    task_type: str,
    expected_compatible: bool = True
):
    """
    Assert model-task compatibility.
    
    Args:
        model_type: Type of model
        task_type: Type of task
        expected_compatible: Whether combination should be compatible
    """
    # Known incompatible combinations
    incompatible_combinations = {
        ('logistic', 'regression'),
        ('ridge', 'classification')
    }
    
    is_compatible = (model_type, task_type) not in incompatible_combinations
    
    if expected_compatible and not is_compatible:
        raise AssertionError(f"Model {model_type} should be compatible with {task_type} but is not")
    
    if not expected_compatible and is_compatible:
        raise AssertionError(f"Model {model_type} should not be compatible with {task_type} but is")


def assert_file_exists(
    file_path: Union[str, Any],
    name: str = "file"
):
    """
    Assert that a file exists.
    
    Args:
        file_path: Path to file
        name: Name for error messages
    """
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists():
        raise AssertionError(f"{name} does not exist: {file_path}")
    
    if not path.is_file():
        raise AssertionError(f"{name} is not a file: {file_path}")


def assert_directory_structure(
    directory: Union[str, Any],
    expected_files: List[str],
    name: str = "directory"
):
    """
    Assert that directory contains expected files.
    
    Args:
        directory: Path to directory
        expected_files: List of expected file names
        name: Name for error messages
    """
    from pathlib import Path
    
    dir_path = Path(directory)
    if not dir_path.exists():
        raise AssertionError(f"{name} does not exist: {directory}")
    
    if not dir_path.is_dir():
        raise AssertionError(f"{name} is not a directory: {directory}")
    
    missing_files = []
    for filename in expected_files:
        file_path = dir_path / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    if missing_files:
        raise AssertionError(f"{name} missing files: {missing_files}")


def assert_json_structure(
    json_data: Dict[str, Any],
    required_keys: List[str],
    name: str = "JSON data"
):
    """
    Assert that JSON data has required structure.
    
    Args:
        json_data: JSON data dictionary
        required_keys: List of required keys
        name: Name for error messages
    """
    if not isinstance(json_data, dict):
        raise AssertionError(f"{name} must be a dictionary, got {type(json_data)}")
    
    missing_keys = []
    for key in required_keys:
        if key not in json_data:
            missing_keys.append(key)
    
    if missing_keys:
        raise AssertionError(f"{name} missing required keys: {missing_keys}")


def assert_numeric_range(
    value: Union[int, float],
    min_val: float,
    max_val: float,
    name: str = "value"
):
    """
    Assert that a numeric value is within a specified range.
    
    Args:
        value: Value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name for error messages
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise AssertionError(f"{name} must be numeric, got {type(value)}")
    
    if np.isnan(value):
        raise AssertionError(f"{name} is NaN")
    
    if np.isinf(value):
        raise AssertionError(f"{name} is infinite")
    
    if not (min_val <= value <= max_val):
        raise AssertionError(f"{name} = {value} not in range [{min_val}, {max_val}]")


def assert_probability_distribution(
    probabilities: Union[List, np.ndarray],
    tolerance: float = 1e-6,
    name: str = "probabilities"
):
    """
    Assert that values form a valid probability distribution.
    
    Args:
        probabilities: Probability values
        tolerance: Numerical tolerance for sum check
        name: Name for error messages
    """
    if not hasattr(probabilities, '__len__'):
        raise AssertionError(f"{name} must have length")
    
    # Check all values are non-negative
    for i, prob in enumerate(probabilities):
        if prob < 0:
            raise AssertionError(f"{name}[{i}] = {prob} is negative")
    
    # Check sum is approximately 1
    total = sum(probabilities)
    if abs(total - 1.0) > tolerance:
        raise AssertionError(f"{name} sum = {total} is not close to 1.0 (tolerance: {tolerance})")


def assert_no_missing_values(
    data: Union[List, np.ndarray, Dict],
    name: str = "data"
):
    """
    Assert that data contains no missing values (None, NaN, etc.).
    
    Args:
        data: Data to check
        name: Name for error messages
    """
    if isinstance(data, dict):
        for key, values in data.items():
            assert_no_missing_values(values, f"{name}[{key}]")
    elif hasattr(data, '__iter__'):
        for i, value in enumerate(data):
            if value is None:
                raise AssertionError(f"{name}[{i}] is None")
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                raise AssertionError(f"{name}[{i}] is NaN")
    else:
        if data is None:
            raise AssertionError(f"{name} is None")
        if isinstance(data, (float, np.floating)) and np.isnan(data):
            raise AssertionError(f"{name} is NaN")


def assert_positive_definite(
    matrix: np.ndarray,
    name: str = "matrix"
):
    """
    Assert that a matrix is positive definite.
    
    Args:
        matrix: Matrix to check
        name: Name for error messages
    """
    if not isinstance(matrix, np.ndarray):
        raise AssertionError(f"{name} must be numpy array")
    
    if matrix.ndim != 2:
        raise AssertionError(f"{name} must be 2D matrix")
    
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError(f"{name} must be square matrix")
    
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        if not np.all(eigenvalues > 0):
            raise AssertionError(f"{name} is not positive definite (has non-positive eigenvalues)")
    except np.linalg.LinAlgError:
        raise AssertionError(f"{name} eigenvalue computation failed")


def assert_memory_usage_reasonable(
    max_memory_mb: float = 1000,
    name: str = "operation"
):
    """
    Assert that memory usage is reasonable.
    
    Args:
        max_memory_mb: Maximum allowed memory usage in MB
        name: Name for error messages
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > max_memory_mb:
            raise AssertionError(f"{name} memory usage too high: {memory_mb:.1f}MB > {max_memory_mb}MB")
    
    except ImportError:
        logger.warning("psutil not available, skipping memory usage check")


def assert_execution_time_reasonable(
    execution_time: float,
    max_time_seconds: float,
    name: str = "operation"
):
    """
    Assert that execution time is reasonable.
    
    Args:
        execution_time: Actual execution time in seconds
        max_time_seconds: Maximum allowed time in seconds
        name: Name for error messages
    """
    if execution_time > max_time_seconds:
        raise AssertionError(f"{name} took too long: {execution_time:.2f}s > {max_time_seconds}s")


def assert_language_support(
    languages: List[str],
    supported_languages: Optional[List[str]] = None
):
    """
    Assert that all languages are supported.
    
    Args:
        languages: List of language codes to check
        supported_languages: List of supported language codes
    """
    if supported_languages is None:
        supported_languages = ['ar', 'en', 'fi', 'id', 'ja', 'ko', 'ru']
    
    unsupported = []
    for lang in languages:
        if lang not in supported_languages:
            unsupported.append(lang)
    
    if unsupported:
        raise AssertionError(f"Unsupported languages: {unsupported}")


def assert_config_validity(
    config: Dict[str, Any],
    config_type: str = "experiment"
):
    """
    Assert that configuration is valid.
    
    Args:
        config: Configuration dictionary
        config_type: Type of configuration
    """
    if not isinstance(config, dict):
        raise AssertionError(f"Config must be dictionary, got {type(config)}")
    
    if config_type == "experiment":
        required_keys = ['models', 'tasks', 'languages']
        for key in required_keys:
            if key not in config:
                raise AssertionError(f"Missing required config key: {key}")
        
        # Validate models
        if not isinstance(config['models'], list) or len(config['models']) == 0:
            raise AssertionError("Config 'models' must be non-empty list")
        
        # Validate tasks
        if not isinstance(config['tasks'], list) or len(config['tasks']) == 0:
            raise AssertionError("Config 'tasks' must be non-empty list")
        
        # Validate languages
        if not isinstance(config['languages'], list) or len(config['languages']) == 0:
            raise AssertionError("Config 'languages' must be non-empty list")


def assert_backwards_compatibility(
    old_interface_func,
    new_interface_func,
    test_args: Tuple = (),
    test_kwargs: Dict = None
):
    """
    Assert that new interface maintains backwards compatibility.
    
    Args:
        old_interface_func: Function with old interface
        new_interface_func: Function with new interface
        test_args: Test arguments
        test_kwargs: Test keyword arguments
    """
    if test_kwargs is None:
        test_kwargs = {}
    
    try:
        old_result = old_interface_func(*test_args, **test_kwargs)
        new_result = new_interface_func(*test_args, **test_kwargs)
        
        # Results should be compatible (might not be identical due to improvements)
        if type(old_result) != type(new_result):
            logger.warning(f"Return types differ: {type(old_result)} vs {type(new_result)}")
        
    except Exception as e:
        raise AssertionError(f"Backwards compatibility test failed: {e}")


def assert_gradient_properties(
    gradients: Dict[str, np.ndarray],
    max_gradient_norm: float = 10.0,
    min_gradient_norm: float = 1e-8
):
    """
    Assert that gradients have reasonable properties.
    
    Args:
        gradients: Dictionary of gradient arrays
        max_gradient_norm: Maximum allowed gradient norm
        min_gradient_norm: Minimum expected gradient norm
    """
    for name, grad in gradients.items():
        if not isinstance(grad, (np.ndarray, type(None))):
            raise AssertionError(f"Gradient {name} must be numpy array or None")
        
        if grad is None:
            continue
        
        # Check for NaN or infinite gradients
        if np.any(np.isnan(grad)):
            raise AssertionError(f"Gradient {name} contains NaN values")
        
        if np.any(np.isinf(grad)):
            raise AssertionError(f"Gradient {name} contains infinite values")
        
        # Check gradient magnitude
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm > max_gradient_norm:
            raise AssertionError(f"Gradient {name} norm too large: {grad_norm} > {max_gradient_norm}")
        
        if grad_norm < min_gradient_norm:
            logger.warning(f"Gradient {name} norm very small: {grad_norm} < {min_gradient_norm}")


# Composite assertion functions for common patterns

def assert_valid_tfidf_features(
    features: Dict[str, Any],
    expected_splits: List[str] = None,
    min_vocab_size: int = 10
):
    """
    Assert that TF-IDF features are valid.
    
    Args:
        features: Features dictionary
        expected_splits: Expected split names
        min_vocab_size: Minimum vocabulary size
    """
    if expected_splits is None:
        expected_splits = ['train', 'val', 'test']
    
    # Check that all expected splits are present
    for split in expected_splits:
        if split not in features:
            raise AssertionError(f"Missing split in features: {split}")
    
    # Check each split
    vocab_sizes = set()
    for split, matrix in features.items():
        assert_sparse_matrix(matrix, f"{split} features")
        vocab_sizes.add(matrix.shape[1])
    
    # All splits should have same vocabulary size
    if len(vocab_sizes) > 1:
        raise AssertionError(f"Inconsistent vocabulary sizes across splits: {vocab_sizes}")
    
    vocab_size = vocab_sizes.pop()
    if vocab_size < min_vocab_size:
        raise AssertionError(f"Vocabulary size too small: {vocab_size} < {min_vocab_size}")


def assert_valid_experiment_pipeline(
    input_data: Dict[str, Any],
    model_output: Any,
    evaluation_metrics: Dict[str, Any],
    task_type: str
):
    """
    Assert that complete experiment pipeline is valid.
    
    Args:
        input_data: Input data dictionary
        model_output: Model predictions
        evaluation_metrics: Evaluation metrics
        task_type: Task type
    """
    # Check input data
    assert_dataset_consistency(input_data)
    
    # Check model output
    if hasattr(model_output, '__len__'):
        n_predictions = len(model_output)
        n_samples = len(next(iter(input_data.values())))
        
        if n_predictions != n_samples:
            raise AssertionError(f"Prediction count mismatch: {n_predictions} vs {n_samples}")
    
    # Check evaluation metrics
    assert_valid_metrics(evaluation_metrics, task_type)


def assert_cross_lingual_consistency(
    results: Dict[str, Dict[str, Any]],
    languages: List[str]
):
    """
    Assert consistency across cross-lingual results.
    
    Args:
        results: Results dictionary mapping languages to metrics
        languages: Expected languages
    """
    # Check all languages present
    for lang in languages:
        if lang not in results:
            raise AssertionError(f"Missing results for language: {lang}")
    
    # Check result structure consistency
    result_keys = None
    for lang, lang_results in results.items():
        if result_keys is None:
            result_keys = set(lang_results.keys())
        else:
            if set(lang_results.keys()) != result_keys:
                raise AssertionError(f"Inconsistent result keys for language {lang}")


if __name__ == "__main__":
    # Example usage of assertion functions
    print("Testing assertion functions...")
    
    # Test matrix assertions
    import scipy.sparse as sparse
    
    # Create test sparse matrix
    test_matrix = sparse.csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    
    try:
        assert_sparse_matrix(test_matrix, min_sparsity=0.5)
        print("Sparse matrix assertion passed")
    except AssertionError as e:
        print(f"Sparse matrix assertion failed: {e}")
    
    # Test metrics assertions
    test_metrics = {
        'accuracy': 0.85,
        'f1': 0.82,
        'precision': 0.88,
        'recall': 0.77
    }
    
    try:
        assert_classification_metrics(test_metrics)
        print("Classification metrics assertion passed")
    except AssertionError as e:
        print(f"Classification metrics assertion failed: {e}")
    
    # Test dataset assertions
    test_dataset = {
        'text': ['Sample 1', 'Sample 2', 'Sample 3'],
        'language': ['en', 'en', 'en'],
        'question_type': [1, 0, 1]
    }
    
    try:
        assert_dataset_consistency(test_dataset, expected_size=3)
        print("Dataset consistency assertion passed")
    except AssertionError as e:
        print(f"Dataset consistency assertion failed: {e}")
    
    print("Assertion function tests completed")