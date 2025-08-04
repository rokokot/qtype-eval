# utils/test_helpers.py
"""
Test helper utilities for TF-IDF integration testing.
Provides common testing utilities, validation functions, and test data management.
"""

import sys
import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import json
import importlib
import subprocess

logger = logging.getLogger(__name__)


def validate_test_environment() -> Dict[str, Any]:
    """
    Validate that the test environment is properly set up.
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'python_path': sys.path,
        'environment_variables': {},
        'dependencies': {},
        'errors': [],
        'warnings': []
    }
    
    # Check Python version
    if sys.version_info < (3, 7):
        validation['errors'].append(f"Python version too old: {sys.version}")
    
    # Check critical environment variables
    env_vars = ['PYTHONPATH', 'HF_HOME', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']
    for var in env_vars:
        validation['environment_variables'][var] = os.environ.get(var, 'Not set')
    
    # Check project structure
    project_root = Path.cwd()
    required_dirs = ['src', 'src/data', 'src/models', 'src/training', 'tests']
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            validation['errors'].append(f"Missing required directory: {dir_path}")
    
    # Check critical dependencies
    critical_deps = [
        'numpy', 'scipy', 'pandas', 'sklearn', 'pytest',
        'transformers', 'datasets', 'torch', 'omegaconf'
    ]
    
    for dep in critical_deps:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'Unknown')
            validation['dependencies'][dep] = version
        except ImportError:
            validation['dependencies'][dep] = 'Not installed'
            validation['errors'].append(f"Missing critical dependency: {dep}")
    
    # Check if project modules can be imported
    try:
        from src.data.tfidf_features import TfidfFeatureLoader
        from src.models.tfidf_baselines import create_tfidf_baseline_model
        from src.training.sklearn_trainer import SklearnTrainer
        validation['project_imports'] = 'Success'
    except ImportError as e:
        validation['project_imports'] = f'Failed: {e}'
        validation['errors'].append(f"Cannot import project modules: {e}")
    
    # Check GPU availability (optional)
    try:
        import torch
        if torch.cuda.is_available():
            validation['gpu_available'] = True
            validation['gpu_count'] = torch.cuda.device_count()
        else:
            validation['gpu_available'] = False
            validation['warnings'].append("GPU not available")
    except:
        validation['gpu_available'] = False
    
    return validation


def create_temporary_workspace(prefix: str = "tfidf_test_") -> Path:
    """
    Create a temporary workspace for testing.
    
    Args:
        prefix: Prefix for temporary directory name
        
    Returns:
        Path to temporary workspace
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    workspace = Path(temp_dir)
    
    # Create standard subdirectories
    subdirs = ['data', 'features', 'outputs', 'cache', 'models', 'logs']
    for subdir in subdirs:
        (workspace / subdir).mkdir(exist_ok=True)
    
    logger.info(f"Created temporary workspace: {workspace}")
    return workspace


def cleanup_workspace(workspace: Path):
    """
    Clean up temporary workspace.
    
    Args:
        workspace: Path to workspace to clean up
    """
    if workspace.exists() and workspace.is_dir():
        shutil.rmtree(workspace)
        logger.info(f"Cleaned up workspace: {workspace}")


def compare_matrices(
    matrix1: Union[np.ndarray, Any], 
    matrix2: Union[np.ndarray, Any],
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Compare two matrices (dense or sparse) for testing.
    
    Args:
        matrix1: First matrix
        matrix2: Second matrix
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dictionary with comparison results
    """
    result = {
        'are_equal': False,
        'shape_match': False,
        'type_match': False,
        'max_difference': None,
        'mean_difference': None,
        'errors': []
    }
    
    try:
        # Convert to dense if sparse
        if hasattr(matrix1, 'toarray'):
            matrix1 = matrix1.toarray()
        if hasattr(matrix2, 'toarray'):
            matrix2 = matrix2.toarray()
        
        # Check types
        result['type_match'] = type(matrix1) == type(matrix2)
        
        # Check shapes
        if hasattr(matrix1, 'shape') and hasattr(matrix2, 'shape'):
            result['shape_match'] = matrix1.shape == matrix2.shape
            
            if result['shape_match']:
                # Calculate differences
                diff = np.abs(matrix1 - matrix2)
                result['max_difference'] = float(np.max(diff))
                result['mean_difference'] = float(np.mean(diff))
                
                # Check if equal within tolerance
                result['are_equal'] = result['max_difference'] <= tolerance
            else:
                result['errors'].append(f"Shape mismatch: {matrix1.shape} vs {matrix2.shape}")
        else:
            result['errors'].append("One or both objects don't have shape attribute")
            
    except Exception as e:
        result['errors'].append(f"Comparison failed: {str(e)}")
    
    return result


def validate_dataset_structure(
    dataset: Dict[str, Any],
    required_fields: Optional[List[str]] = None,
    expected_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validate dataset structure for testing.
    
    Args:
        dataset: Dataset dictionary to validate
        required_fields: List of required field names
        expected_samples: Expected number of samples
        
    Returns:
        Validation results dictionary
    """
    if required_fields is None:
        required_fields = ['text', 'language', 'question_type', 'lang_norm_complexity_score']
    
    result = {
        'is_valid': True,
        'field_counts': {},
        'missing_fields': [],
        'extra_fields': [],
        'data_types': {},
        'sample_count': 0,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Check if dataset is empty
        if not dataset:
            result['errors'].append("Dataset is empty")
            result['is_valid'] = False
            return result
        
        # Get sample count from first field
        first_field = next(iter(dataset.values()))
        result['sample_count'] = len(first_field) if hasattr(first_field, '__len__') else 0
        
        # Check field consistency
        for field, values in dataset.items():
            field_length = len(values) if hasattr(values, '__len__') else 0
            result['field_counts'][field] = field_length
            result['data_types'][field] = type(values).__name__
            
            if field_length != result['sample_count']:
                result['errors'].append(f"Field {field} has {field_length} samples, expected {result['sample_count']}")
                result['is_valid'] = False
        
        # Check required fields
        for field in required_fields:
            if field not in dataset:
                result['missing_fields'].append(field)
                result['is_valid'] = False
        
        # Check for extra fields
        dataset_fields = set(dataset.keys())
        required_fields_set = set(required_fields)
        extra_fields = dataset_fields - required_fields_set
        result['extra_fields'] = list(extra_fields)
        
        # Check expected sample count
        if expected_samples is not None and result['sample_count'] != expected_samples:
            result['warnings'].append(f"Expected {expected_samples} samples, got {result['sample_count']}")
        
        # Validate specific field types and ranges
        if 'question_type' in dataset:
            qt_values = set(dataset['question_type'])
            if not qt_values.issubset({0, 1}):
                result['errors'].append(f"Invalid question_type values: {qt_values - {0, 1}}")
                result['is_valid'] = False
        
        if 'lang_norm_complexity_score' in dataset:
            complexity_scores = dataset['lang_norm_complexity_score']
            if any(not (0 <= score <= 1) for score in complexity_scores if isinstance(score, (int, float))):
                result['warnings'].append("Some complexity scores are outside [0, 1] range")
        
        # Check language field
        if 'language' in dataset:
            languages = set(dataset['language'])
            result['unique_languages'] = list(languages)
            if len(languages) == 0:
                result['errors'].append("No languages found in dataset")
                result['is_valid'] = False
        
    except Exception as e:
        result['errors'].append(f"Validation failed with exception: {str(e)}")
        result['is_valid'] = False
    
    return result


def create_test_config(
    base_config: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a test configuration with sensible defaults.
    
    Args:
        base_config: Base configuration dictionary
        overrides: Configuration overrides
        
    Returns:
        Test configuration dictionary
    """
    if base_config is None:
        base_config = {
            'experiment_type': 'tfidf_baselines',
            'models': ['dummy'],
            'tasks': ['question_type'],
            'languages': [['en']],
            'controls': {'enabled': False},
            'wandb': {'mode': 'disabled'},
            'model_params': {
                'dummy': {
                    'classification': {'strategy': 'most_frequent'},
                    'regression': {'strategy': 'mean'}
                },
                'logistic': {
                    'C': 1.0,
                    'solver': 'liblinear',
                    'max_iter': 1000,
                    'random_state': 42
                },
                'ridge': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            }
        }
    
    config = base_config.copy()
    
    if overrides:
        config.update(overrides)
    
    return config


def monitor_resource_usage():
    """
    Monitor system resource usage during tests.
    
    Returns:
        Resource monitoring context manager
    """
    class ResourceMonitor:
        def __init__(self):
            self.start_memory = None
            self.peak_memory = None
            self.start_time = None
            self.duration = None
        
        def __enter__(self):
            import psutil
            import time
            
            self.process = psutil.Process()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
            self.start_time = time.time()
            
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            import time
            
            self.duration = time.time() - self.start_time
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, final_memory)
            
            logger.info(f"Resource usage - Duration: {self.duration:.2f}s, "
                       f"Memory: {self.start_memory:.1f}MB -> {final_memory:.1f}MB "
                       f"(Peak: {self.peak_memory:.1f}MB)")
        
        def update_peak_memory(self):
            """Update peak memory usage."""
            current_memory = self.process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current_memory)
    
    return ResourceMonitor()


def run_with_timeout(func, timeout_seconds: int = 30, *args, **kwargs):
    """
    Run a function with a timeout for testing.
    
    Args:
        func: Function to run
        timeout_seconds: Timeout in seconds
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or raises TimeoutError
    """
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
    
    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        raise
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def capture_logs(logger_name: str = None, level: int = logging.INFO):
    """
    Capture logs during testing.
    
    Args:
        logger_name: Name of logger to capture (None for root logger)
        level: Logging level to capture
        
    Returns:
        Log capture context manager
    """
    import io
    
    class LogCapture:
        def __init__(self, logger_name, level):
            self.logger = logging.getLogger(logger_name)
            self.level = level
            self.log_capture = io.StringIO()
            self.handler = logging.StreamHandler(self.log_capture)
            self.original_level = None
        
        def __enter__(self):
            self.original_level = self.logger.level
            self.logger.setLevel(self.level)
            self.logger.addHandler(self.handler)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.logger.removeHandler(self.handler)
            if self.original_level is not None:
                self.logger.setLevel(self.original_level)
        
        def get_logs(self) -> str:
            """Get captured log content."""
            return self.log_capture.getvalue()
        
        def has_message(self, message: str) -> bool:
            """Check if logs contain a specific message."""
            return message in self.get_logs()
    
    return LogCapture(logger_name, level)


def mock_file_system(files: Dict[str, Any]) -> Any:
    """
    Create a mock file system for testing.
    
    Args:
        files: Dictionary mapping file paths to content
        
    Returns:
        Mock file system context manager
    """
    from unittest.mock import mock_open, patch
    
    class MockFileSystem:
        def __init__(self, files):
            self.files = files
            self.patches = []
        
        def __enter__(self):
            # Mock open function
            def mock_open_func(filename, mode='r', *args, **kwargs):
                if filename in self.files:
                    content = self.files[filename]
                    if isinstance(content, dict):
                        # JSON content
                        content = json.dumps(content)
                    elif not isinstance(content, str):
                        # Convert to string
                        content = str(content)
                    
                    return mock_open(read_data=content)()
                else:
                    raise FileNotFoundError(f"No such file: {filename}")
            
            self.open_patch = patch('builtins.open', side_effect=mock_open_func)
            self.open_patch.start()
            
            # Mock Path.exists
            def mock_exists(path):
                return str(path) in self.files
            
            self.exists_patch = patch('pathlib.Path.exists', side_effect=mock_exists)
            self.exists_patch.start()
            
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.open_patch.stop()
            self.exists_patch.stop()
    
    return MockFileSystem(files)


def assert_experiment_output_structure(output_dir: Path, experiment_name: str):
    """
    Assert that experiment output has the expected structure.
    
    Args:
        output_dir: Output directory path
        experiment_name: Expected experiment name
    """
    # Check main output directory exists
    assert output_dir.exists(), f"Output directory does not exist: {output_dir}"
    
    # Check experiment subdirectory
    exp_dir = output_dir / experiment_name
    assert exp_dir.exists(), f"Experiment directory does not exist: {exp_dir}"
    
    # Check for expected files
    expected_files = ['results.json', 'config.yaml']
    for filename in expected_files:
        file_path = exp_dir / filename
        assert file_path.exists(), f"Expected file missing: {file_path}"
    
    # Check results.json structure
    with open(exp_dir / 'results.json') as f:
        results = json.load(f)
    
    required_keys = ['experiment_name', 'model_type', 'task', 'test_metrics']
    for key in required_keys:
        assert key in results, f"Missing key in results: {key}"


def validate_metrics_structure(metrics: Dict[str, Any], task_type: str):
    """
    Validate metrics structure for given task type.
    
    Args:
        metrics: Metrics dictionary
        task_type: 'classification' or 'regression'
    """
    assert isinstance(metrics, dict), f"Metrics should be dict, got {type(metrics)}"
    
    if task_type == 'classification':
        required_metrics = ['accuracy', 'f1']
        for metric in required_metrics:
            assert metric in metrics, f"Missing classification metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Invalid metric type for {metric}"
            assert 0 <= metrics[metric] <= 1, f"Metric {metric} out of range [0,1]: {metrics[metric]}"
    
    elif task_type == 'regression':
        required_metrics = ['mse', 'r2']
        for metric in required_metrics:
            assert metric in metrics, f"Missing regression metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Invalid metric type for {metric}"
        
        # MSE should be non-negative
        assert metrics['mse'] >= 0, f"MSE should be non-negative: {metrics['mse']}"
        
        # R² should be <= 1 (can be negative for very bad models)
        assert metrics['r2'] <= 1, f"R² should be <= 1: {metrics['r2']}"
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def create_minimal_test_features(output_dir: str, n_samples: int = 30):
    """
    Create minimal TF-IDF features for quick testing.
    
    Args:
        output_dir: Directory to save features
        n_samples: Number of samples per split
    """
    from src.data.tfidf_features import create_test_features
    
    create_test_features(
        output_dir=output_dir,
        n_samples=n_samples,
        vocab_size=50,  # Small vocabulary for speed
        sparsity=0.8    # High sparsity for efficiency
    )


def check_backward_compatibility(test_func, *args, **kwargs):
    """
    Check that changes maintain backward compatibility.
    
    Args:
        test_func: Function to test for backward compatibility
        *args: Function arguments
        **kwargs: Function keyword arguments
    """
    # Test with old-style parameters
    old_style_kwargs = kwargs.copy()
    
    # Remove new TF-IDF specific parameters if present
    tfidf_params = ['use_tfidf_loader', 'tfidf_features_dir']
    for param in tfidf_params:
        old_style_kwargs.pop(param, None)
    
    try:
        # Should work with old parameters (might fail due to missing data, but not due to API changes)
        result = test_func(*args, **old_style_kwargs)
        logger.info("Backward compatibility check passed")
        return True
    except TypeError as e:
        if any(param in str(e) for param in tfidf_params):
            logger.error(f"Backward compatibility broken: {e}")
            raise AssertionError(f"Backward compatibility test failed: {e}")
        else:
            # This might be an expected error due to missing data files
            logger.warning(f"Backward compatibility test failed with expected error: {e}")
            return False
    except Exception as e:
        logger.warning(f"Backward compatibility test failed with error: {e}")
        return False


def setup_test_logging(level: int = logging.INFO):
    """
    Set up logging for tests.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test.log')
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def parametrize_language_combinations():
    """
    Generate parameter combinations for multilingual testing.
    
    Returns:
        List of language combinations for pytest.mark.parametrize
    """
    single_languages = [['en'], ['ru'], ['ar'], ['fi']]
    multi_languages = [['en', 'ru'], ['en', 'ar'], ['ru', 'ar'], ['en', 'ru', 'ar']]
    
    return single_languages + multi_languages


def parametrize_model_task_combinations():
    """
    Generate parameter combinations for model-task testing.
    
    Returns:
        List of (model, task, task_type) combinations
    """
    combinations = [
        ('dummy', 'question_type', 'classification'),
        ('dummy', 'complexity', 'regression'),
        ('logistic', 'question_type', 'classification'),
        ('ridge', 'complexity', 'regression'),
        ('random_forest', 'question_type', 'classification'),
        ('random_forest', 'complexity', 'regression')
    ]
    
    return combinations


def skip_if_no_gpu():
    """Pytest skip decorator for GPU-required tests."""
    import pytest
    
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    
    return pytest.mark.skipif(not has_gpu, reason="GPU not available")


def skip_if_no_internet():
    """Pytest skip decorator for tests requiring internet."""
    import pytest
    import urllib.request
    
    def has_internet():
        try:
            urllib.request.urlopen('http://www.google.com', timeout=5)
            return True
        except:
            return False
    
    return pytest.mark.skipif(not has_internet(), reason="Internet connection not available")


def temporary_environment_vars(**env_vars):
    """
    Context manager to temporarily set environment variables.
    
    Args:
        **env_vars: Environment variables to set
    """
    import os
    
    class TempEnvVars:
        def __init__(self, env_vars):
            self.env_vars = env_vars
            self.original_values = {}
        
        def __enter__(self):
            for key, value in self.env_vars.items():
                self.original_values[key] = os.environ.get(key)
                os.environ[key] = str(value)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            for key in self.env_vars:
                if self.original_values[key] is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = self.original_values[key]
    
    return TempEnvVars(env_vars)


def measure_execution_time(func, *args, **kwargs):
    """
    Measure execution time of a function.
    
    Args:
        func: Function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, execution_time_seconds)
    """
    import time
    
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    return result, execution_time


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage of test helpers
    print("Testing environment validation...")
    
    validation = validate_test_environment()
    print(f"Environment validation: {'PASSED' if not validation['errors'] else 'FAILED'}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    print(f"Dependencies: {validation['dependencies']}")
    
    # Test workspace creation
    workspace = create_temporary_workspace()
    print(f"Created workspace: {workspace}")
    
    # Test configuration creation
    config = create_test_config()
    print(f"Test config: {config}")
    
    # Cleanup
    cleanup_workspace(workspace)
    print("Workspace cleaned up")