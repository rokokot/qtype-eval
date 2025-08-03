# tests/conftest.py
"""
Complete pytest configuration and shared fixtures for TF-IDF integration testing.
"""

import pytest
import tempfile
import shutil
import os
import sys
from pathlib import Path
import logging
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Import test utilities
from tests.fixtures.sample_data import (
    create_sample_dataset,
    create_sample_tfidf_features,
    generate_sample_labels
)
from tests.fixtures.mock_datasets import MockDatasetLoader
from utils.test_helpers import validate_test_environment
from utils.assertions import assert_valid_metrics, assert_valid_shape

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a session-wide test data directory."""
    temp_dir = tempfile.mkdtemp(prefix="tfidf_test_session_")
    logger.info(f"Created session test directory: {temp_dir}")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)
    logger.info(f"Cleaned up session test directory: {temp_dir}")

@pytest.fixture(scope="function")
def temp_dir():
    """Create a function-scoped temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="tfidf_func_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def sample_tfidf_features(test_data_dir):
    """Create sample TF-IDF features for testing."""
    from src.data.tfidf_features import create_test_features
    
    features_dir = test_data_dir / "sample_features"
    create_test_features(str(features_dir), n_samples=50, vocab_size=100)
    
    # Verify features were created
    assert (features_dir / "metadata.json").exists()
    assert (features_dir / "X_train_sparse.npz").exists()
    assert (features_dir / "X_val_sparse.npz").exists()
    assert (features_dir / "X_test_sparse.npz").exists()
    
    logger.info(f"Created sample TF-IDF features in {features_dir}")
    return features_dir

@pytest.fixture(scope="session")
def large_tfidf_features(test_data_dir):
    """Create larger TF-IDF features for performance testing."""
    from src.data.tfidf_features import create_test_features
    
    features_dir = test_data_dir / "large_features"
    create_test_features(str(features_dir), n_samples=200, vocab_size=1000)
    
    logger.info(f"Created large TF-IDF features in {features_dir}")
    return features_dir

@pytest.fixture
def mock_dataset_loader():
    """Create a mock dataset loader for testing."""
    return MockDatasetLoader()

@pytest.fixture
def sample_classification_data():
    """Generate sample classification data."""
    return create_sample_dataset(
        n_samples=100,
        task_type='classification',
        n_languages=3
    )

@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    return create_sample_dataset(
        n_samples=100,
        task_type='regression',
        n_languages=3
    )

@pytest.fixture
def mock_hf_dataset():
    """Mock HuggingFace dataset for testing."""
    with patch('src.data.datasets.load_dataset') as mock_load:
        mock_data = {
            'text': [
                "Is this a question?",
                "What is the answer?", 
                "This is a statement.",
                "How are you doing?"
            ],
            'language': ['en', 'en', 'en', 'en'],
            'question_type': [1, 1, 0, 1],
            'lang_norm_complexity_score': [0.5, 0.7, 0.3, 0.6],
            'avg_links_len': [0.4, 0.6, 0.2, 0.5],
            'avg_max_depth': [0.3, 0.5, 0.1, 0.4],
            'avg_subordinate_chain_len': [0.2, 0.4, 0.0, 0.3],
            'avg_verb_edges': [0.6, 0.8, 0.4, 0.7],
            'lexical_density': [0.7, 0.9, 0.5, 0.8],
            'n_tokens': [0.4, 0.6, 0.2, 0.5]
        }
        
        # Create mock dataset split
        from datasets import Dataset
        mock_split = Dataset.from_dict(mock_data)
        
        # Mock the return structure
        mock_dataset = {
            'train': mock_split,
            'validation': mock_split,
            'test': mock_split
        }
        
        mock_load.return_value = mock_dataset
        yield mock_load

@pytest.fixture
def tfidf_test_models():
    """Create test TF-IDF models for validation."""
    models = {}
    
    # Import after adding to path
    from src.models.tfidf_baselines import create_tfidf_baseline_model
    
    # Create different model types for testing
    model_configs = [
        ('dummy', 'classification'),
        ('dummy', 'regression'),
        ('logistic', 'classification'),
        ('ridge', 'regression')
    ]
    
    for model_type, task_type in model_configs:
        key = f"{model_type}_{task_type}"
        models[key] = {
            'model_type': model_type,
            'task_type': task_type,
            'config': {
                'model_type': model_type,
                'task_type': task_type,
                'tfidf_features_dir': None,  # To be set by test
                'target_languages': ['en'],
                'random_state': 42
            }
        }
    
    return models

@pytest.fixture
def sklearn_test_environment(temp_dir, sample_tfidf_features):
    """Set up complete sklearn test environment."""
    env = {
        'features_dir': sample_tfidf_features,
        'output_dir': temp_dir / "outputs",
        'cache_dir': temp_dir / "cache",
        'languages': ['en', 'ru', 'ar'],
        'tasks': ['question_type', 'complexity'],
        'models': ['dummy', 'logistic', 'ridge']
    }
    
    # Create output directories
    env['output_dir'].mkdir(parents=True, exist_ok=True)
    env['cache_dir'].mkdir(parents=True, exist_ok=True)
    
    return env

@pytest.fixture
def integration_test_environment(temp_dir):
    """Set up complete integration test environment."""
    from src.data.tfidf_features import create_test_features
    
    # Create test features
    features_dir = temp_dir / "tfidf_features"
    create_test_features(str(features_dir), n_samples=30, vocab_size=50)
    
    # Set up environment
    env = {
        'features_dir': features_dir,
        'output_dir': temp_dir / "outputs",
        'cache_dir': temp_dir / "cache",
        'test_dir': temp_dir,
        'languages': ['en'],
        'sample_sizes': {
            'train': 20,
            'val': 5,
            'test': 5
        }
    }
    
    # Create directories
    env['output_dir'].mkdir(parents=True, exist_ok=True)
    env['cache_dir'].mkdir(parents=True, exist_ok=True)
    
    return env

@pytest.fixture
def performance_test_data():
    """Generate data for performance testing."""
    return {
        'small': {'n_samples': 50, 'vocab_size': 100},
        'medium': {'n_samples': 200, 'vocab_size': 500},
        'large': {'n_samples': 1000, 'vocab_size': 2000}
    }

# Mock patches for external dependencies
@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock external dependencies that might not be available in test environment."""
    patches = []
    
    # Mock wandb if not available
    try:
        import wandb
    except ImportError:
        wandb_mock = Mock()
        patches.append(patch('src.training.sklearn_trainer.wandb', wandb_mock))
        patches.append(patch('src.training.lm_trainer.wandb', wandb_mock))
    
    # Mock transformers if needed for unit tests
    # Only mock if we're not testing the actual model functionality
    
    # Start all patches
    started_patches = []
    for p in patches:
        started_patches.append(p.start())
    
    yield started_patches
    
    # Stop all patches
    for p in patches:
        p.stop()

# Pytest markers configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "tfidf: mark test as TF-IDF related"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "datasets: mark test as requiring external datasets"
    )

# Skip tests based on environment
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers and environment."""
    
    # Add skip marker for slow tests if not explicitly requested
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip GPU tests if no GPU available
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Skip dataset tests if no external datasets
    if not config.getoption("--datasets"):
        skip_datasets = pytest.mark.skip(reason="need --datasets option to run")
        for item in items:
            if "datasets" in item.keywords:
                item.add_marker(skip_datasets)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--gpu", action="store_true", default=False, help="run GPU tests"
    )
    parser.addoption(
        "--datasets", action="store_true", default=False, help="run tests requiring external datasets"
    )
    parser.addoption(
        "--integration-only", action="store_true", default=False, help="run only integration tests"
    )
    parser.addoption(
        "--unit-only", action="store_true", default=False, help="run only unit tests"
    )

# Validation fixtures
@pytest.fixture
def validate_environment():
    """Validate test environment setup."""
    def _validate():
        # Check Python environment
        validate_test_environment()
        
        # Check required directories exist
        project_root = Path(__file__).parent.parent
        assert (project_root / "src").exists(), "src directory not found"
        assert (project_root / "src" / "data").exists(), "src/data directory not found"
        assert (project_root / "src" / "models").exists(), "src/models directory not found"
        
        # Check imports work
        try:
            from src.data.tfidf_features import TfidfFeatureLoader
            from src.models.tfidf_baselines import create_tfidf_baseline_model
            from src.training.sklearn_trainer import SklearnTrainer
        except ImportError as e:
            pytest.fail(f"Failed to import required modules: {e}")
        
        return True
    
    return _validate

# Helper fixtures for common test patterns
@pytest.fixture
def assert_helpers():
    """Provide assertion helpers for testing."""
    return {
        'assert_valid_metrics': assert_valid_metrics,
        'assert_valid_shape': assert_valid_shape,
        'assert_sparse_matrix': lambda x: hasattr(x, 'nnz') and hasattr(x, 'toarray'),
        'assert_classification_metrics': lambda m: all(k in m for k in ['accuracy', 'f1']),
        'assert_regression_metrics': lambda m: all(k in m for k in ['mse', 'rmse', 'r2'])
    }

@pytest.fixture
def experiment_runner_setup(temp_dir):
    """Set up for experiment runner tests."""
    setup = {
        'output_dir': temp_dir / "experiment_outputs",
        'config_overrides': {
            'wandb.mode': 'disabled',
            'training.num_epochs': 1,
            'training.batch_size': 4,
        },
        'test_configs': [
            {
                'experiment': 'tfidf_baselines',
                'models': ['dummy'],
                'tasks': ['question_type'],
                'languages': [['en']]
            }
        ]
    }
    
    setup['output_dir'].mkdir(parents=True, exist_ok=True)
    return setup

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_environment():
    """Ensure clean environment for each test."""
    # Store original environment
    original_env = {}
    env_vars = ['PYTHONPATH', 'HF_HOME', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']
    
    for var in env_vars:
        original_env[var] = os.environ.get(var)
    
    yield
    
    # Restore original environment
    for var, value in original_env.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value

# Logging configuration for tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for tests."""
    # Reduce logging noise during tests
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    yield
    
    # Reset logging if needed
    logging.getLogger('transformers').setLevel(logging.INFO)

# Performance monitoring
@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import time
    import psutil
    
    def _monitor(test_name: str):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        def stop():
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(f"Performance - {test_name}: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")
            
            return {
                'duration': duration,
                'memory_delta': memory_delta,
                'start_memory': start_memory,
                'end_memory': end_memory
            }
        
        return stop
    
    return _monitor