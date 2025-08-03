# tests/unit/test_tfidf_features.py
"""
Unit tests for TfidfFeatureLoader class.
"""

import pytest
import tempfile
import numpy as np
import scipy.sparse
from pathlib import Path
import json

from src.data.tfidf_features import TfidfFeatureLoader, create_test_features


class TestTfidfFeatureLoader:
    """Unit tests for TfidfFeatureLoader."""
    
    @pytest.fixture(autouse=True)
    def setup_test_features(self, tmp_path):
        """Create test TF-IDF features for testing."""
        self.features_dir = tmp_path / "features"
        create_test_features(str(self.features_dir), n_samples=20)
        
    def test_initialization(self):
        """Test TfidfFeatureLoader initialization."""
        loader = TfidfFeatureLoader(str(self.features_dir))
        assert loader.features_dir.exists()
        assert loader.metadata is not None
        
    def test_initialization_invalid_directory(self):
        """Test initialization with invalid directory."""
        with pytest.raises(FileNotFoundError):
            TfidfFeatureLoader("/nonexistent/directory")
    
    def test_load_features_individual_splits(self):
        """Test loading individual feature splits."""
        loader = TfidfFeatureLoader(str(self.features_dir))
        
        for split in ['train', 'val', 'test']:
            features = loader.load_features(split)
            assert scipy.sparse.issparse(features)
            assert features.shape[0] > 0
            assert features.shape[1] > 0
    
    def test_load_features_invalid_split(self):
        """Test loading features with invalid split name."""
        loader = TfidfFeatureLoader(str(self.features_dir))
        
        with pytest.raises(ValueError):
            loader.load_features('invalid_split')
    
    def test_load_all_features(self):
        """Test loading all feature splits at once."""
        loader = TfidfFeatureLoader(str(self.features_dir))
        
        all_features = loader.load_all_features()
        assert isinstance(all_features, dict)
        assert 'train' in all_features
        assert 'val' in all_features
        assert 'test' in all_features
        
        # Check that all splits have same number of features
        feature_dims = [matrix.shape[1] for matrix in all_features.values()]
        assert len(set(feature_dims)) == 1
    
    def test_verify_features(self):
        """Test feature verification."""
        loader = TfidfFeatureLoader(str(self.features_dir))
        assert loader.verify_features()
    
    def test_get_vocab_size(self):
        """Test vocabulary size retrieval."""
        loader = TfidfFeatureLoader(str(self.features_dir))
        vocab_size = loader.get_vocab_size()
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
    
    def test_get_metadata(self):
        """Test metadata retrieval."""
        loader = TfidfFeatureLoader(str(self.features_dir))
        metadata = loader.get_metadata()
        assert isinstance(metadata, dict)
        assert 'vocab_size' in metadata or 'generation_info' in metadata
    
    def test_get_statistics(self):
        """Test statistics generation."""
        loader = TfidfFeatureLoader(str(self.features_dir))
        stats = loader.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'vocab_size' in stats
        assert 'splits' in stats
        
        for split in ['train', 'val', 'test']:
            assert split in stats['splits']
            split_stats = stats['splits'][split]
            assert 'n_samples' in split_stats
            assert 'n_features' in split_stats
            assert 'sparsity' in split_stats