import pytest
import os
import subprocess
import tempfile
import numpy as np


@pytest.mark.system
@pytest.mark.slow

def test_dummy_end_to_end():
    
    if os.environ.get("CI") != "true" and not os.environ.get("RUN_SLOW_TESTS"):
        pytest.skip("Skipping slow tests. Set RUN_SLOW_TESTS=1 to run.")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run a small dummy experiment
        cmd = [
            "python", "-m", "src.experiments.run_experiment",
            f"output_dir={tmpdir}",
            "experiment=question_type",
            "model=dummy",
            "data.languages=[en]",
            "experiment_name=test_end_to_end",
            "wandb.mode=disabled"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check that the command ran successfully
        assert result.returncode == 0, f"Command failed with: {result.stderr}"
        
        # Check that output files were created
        assert os.path.exists(os.path.join(tmpdir, "results_with_metadata.json"))


@pytest.mark.system
@pytest.mark.slow
def test_creation_of_tfidf_features(monkeypatch):
    """Test creating TF-IDF features for a small sample dataset."""
    # Mock the dataset loading
    def mock_load_dataset(*args, **kwargs):
        from datasets import Dataset
        
        # Create small sample data
        data = {
            "text": ["Is this a question?", "What is a question?", "Hello world", "Testing TF-IDF"],
            "language": ["en", "en", "en", "en"],
            "question_type": [1, 0, 1, 0]
        }
        return Dataset.from_dict(data)
    
    # Mock the load_tfidf_features to create features
    def create_mock_tfidf_features(split, vectors_dir):
        """Create mock TF-IDF features for testing."""
        import pickle
        import scipy.sparse as sparse
        import os
        
        os.makedirs(vectors_dir, exist_ok=True)
        
        # Create a very small sparse matrix based on the split
        if split == "train":
            matrix = sparse.csr_matrix((4, 10))
        elif split == "dev":
            matrix = sparse.csr_matrix((2, 10))
        else:  # test
            matrix = sparse.csr_matrix((2, 10))
            
        # Save the matrix
        with open(os.path.join(vectors_dir, f"tfidf_vectors_{split}.pkl"), "wb") as f:
            pickle.dump(matrix, f)
            
        return matrix
    
    monkeypatch.setattr("src.data.datasets.load_dataset", mock_load_dataset)
    monkeypatch.setattr("src.data.datasets.load_tfidf_features", create_mock_tfidf_features)
    
    # Test creating and loading features
    with tempfile.TemporaryDirectory() as tmpdir:
        vectors_dir = os.path.join(tmpdir, "features")
        os.makedirs(vectors_dir, exist_ok=True)
        
        # Import the function after patching
        from src.data.datasets import load_sklearn_data
        
        # Test loading the sklearn data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
            languages=["en"],
            task="question_type",
            vectors_dir=vectors_dir
        )
        
        # Check that the matrices have the expected shapes
        assert X_train.shape == (4, 10)
        assert X_val.shape == (2, 10)
        assert X_test.shape == (2, 10)
        
        # Check that labels are correct type
        assert y_train.dtype == np.int64
        assert len(y_train) == 4