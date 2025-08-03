
# tests/conftest.py
"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a session-wide test data directory."""
    temp_dir = tempfile.mkdtemp(prefix="tfidf_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

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
    create_test_features(str(features_dir), n_samples=50)
    return features_dir

# Pytest markers
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

# Skip tests that require external resources in CI
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    import pytest
    
    # Add skip marker for slow tests if not explicitly requested
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )