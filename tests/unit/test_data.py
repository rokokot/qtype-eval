import pytest
import os
import numpy as np
from src.data.datasets import load_combined_dataset, load_tfidf_features
from datasets import Dataset


@pytest.fixture
def mock_dataset(monkeypatch):
    def mock_load_dataset(*args, **kwargs):
        data = {
            "text": ["Is this a question?", "What is a question?"],
            "language": ["en", "en"],
            "question_type": [1, 0],
            "lang_norm_complexity_score": [0.5, 0.2],
            "avg_links_len": [0.3, 0.1],
            "avg_max_depth": [0.4, 0.1],
            "avg_subordinate_chain_len": [0.0, 0.2],
            "avg_verb_edges": [0.5, 0.5],
            "lexical_density": [0.7, 0.8],
            "n_tokens": [0.4, 0.5]
        }
        return Dataset.from_dict(data)
    
    monkeypatch.setattr("src.data.datasets.load_dataset", mock_load_dataset)


@pytest.mark.unit
def test_load_combined_dataset(mock_dataset):
    df = load_combined_dataset(split="train", task="question_type")
    
    assert len(df) == 2
    assert "question_type" in df.columns
    assert "language" in df.columns
    assert df["language"].iloc[0] == "en"


@pytest.mark.unit
def test_load_tfidf_features(monkeypatch, tmp_path):

    import pickle
    import scipy.sparse as sparse
    
    vectors_dir = tmp_path / "features"
    vectors_dir.mkdir()
    
    # Create a sparse matrix and save it
    matrix = sparse.csr_matrix((10, 100))
    with open(vectors_dir / "tfidf_vectors_train.pkl", "wb") as f:
        pickle.dump(matrix, f)
    
    features = load_tfidf_features("train", vectors_dir=str(vectors_dir))
    
    assert sparse.issparse(features)
    assert features.shape == (10, 100)


@pytest.mark.unit
def test_task_to_feature_mapping():
    from src.data.datasets import TASK_TO_FEATURE
    
    assert TASK_TO_FEATURE["question_type"] == "question_type"
    assert TASK_TO_FEATURE["complexity"] == "lang_norm_complexity_score"
    assert "avg_links_len" in TASK_TO_FEATURE
    assert "n_tokens" in TASK_TO_FEATURE