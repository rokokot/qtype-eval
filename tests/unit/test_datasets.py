# tests/unit/test_datasets.py
"""
Unit tests for enhanced dataset loading functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.data.datasets import (
    ensure_string_task,
    get_available_tasks,
    get_available_languages,
    get_task_type,
    load_tfidf_features_new,
    load_tfidf_vectors,
    extract_labels_from_split
)


class TestDatasetUtilities:
    """Test dataset utility functions."""
    
    def test_ensure_string_task(self):
        """Test task name normalization."""
        assert ensure_string_task('question_type') == 'question_type'
        assert ensure_string_task(['question_type']) == 'question_type'
        assert ensure_string_task(['complexity', 'other']) == 'complexity'
        assert ensure_string_task([]) == 'question_type'
        assert ensure_string_task(None) == 'question_type'
    
    def test_get_available_tasks(self):
        """Test available tasks retrieval."""
        tasks = get_available_tasks()
        assert isinstance(tasks, list)
        assert 'question_type' in tasks
        assert 'complexity' in tasks
        assert 'single_submetric' in tasks
    
    def test_get_available_languages(self):
        """Test available languages retrieval."""
        languages = get_available_languages()
        assert isinstance(languages, list)
        assert 'en' in languages
        assert 'ru' in languages
        assert 'ar' in languages
        assert len(languages) == 7  # Expected number of languages
    
    def test_get_task_type(self):
        """Test task type determination."""
        assert get_task_type('question_type') == 'classification'
        assert get_task_type('complexity') == 'regression'
        assert get_task_type('single_submetric') == 'regression'
        assert get_task_type('avg_links_len') == 'regression'
        assert get_task_type('unknown_task') == 'classification'  # Default


class TestTfidfFeatureLoading:
    """Test TF-IDF feature loading functions."""
    
    @patch('src.data.datasets.TfidfFeatureLoader')
    def test_load_tfidf_features_new(self, mock_loader_class):
        """Test new TF-IDF feature loading."""
        # Mock the loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        # Mock feature matrices
        mock_features = {
            'train': Mock(),
            'val': Mock(), 
            'test': Mock()
        }
        mock_loader.load_all_features.return_value = mock_features
        mock_loader.filter_by_languages.return_value = mock_features
        
        # Test the function
        result = load_tfidf_features_new('/test/dir', ['en'])
        
        # Verify calls
        mock_loader_class.assert_called_once_with('/test/dir')
        mock_loader.load_all_features.assert_called_once()
        mock_loader.filter_by_languages.assert_called_once_with(mock_features, ['en'])
        
        # Verify result
        assert len(result) == 3
    
    @patch('builtins.open')
    @patch('pickle.load')
    @patch('os.path.exists')
    def test_load_tfidf_vectors_legacy(self, mock_exists, mock_pickle_load, mock_open):
        """Test legacy TF-IDF vector loading."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock loaded vectors
        mock_vectors = [Mock(), Mock(), Mock()]
        mock_pickle_load.side_effect = mock_vectors
        
        # Mock sparse matrix check
        for vector in mock_vectors:
            vector.toarray.return_value = np.array([[1, 2, 3]])
        
        with patch('scipy.sparse.issparse', return_value=True):
            result = load_tfidf_vectors('/test/dir', ['en'])
        
        # Verify result
        assert len(result) == 3
        
        # Verify file operations
        assert mock_open.call_count == 3
        assert mock_pickle_load.call_count == 3


class TestLabelExtraction:
    """Test label extraction functionality."""
    
    def test_extract_labels_from_split(self):
        """Test label extraction from dataset split."""
        # Mock dataset split
        mock_split = Mock()
        mock_df = Mock()
        mock_split.to_pandas.return_value = mock_df
        
        # Mock DataFrame operations
        mock_df.__getitem__.return_value = mock_df  # For language filtering
        mock_df.columns = ['text', 'language', 'question_type']
        mock_df.values = np.array([0, 1, 0, 1])
        
        # Mock pandas isna
        with patch('pandas.isna') as mock_isna:
            mock_isna.return_value = np.array([False, False, False, False])
            
            result = extract_labels_from_split(mock_split, 'question_type', ['en'])
            
            # Verify result
            assert isinstance(result, np.ndarray)
            assert len(result) == 4
    
    def test_extract_labels_missing_feature(self):
        """Test label extraction with missing feature."""
        mock_split = Mock()
        mock_df = Mock()
        mock_split.to_pandas.return_value = mock_df
        
        # Mock missing column
        mock_df.columns = ['text', 'language']  # Missing 'question_type'
        
        with pytest.raises(ValueError, match="Target feature .* not found"):
            extract_labels_from_split(mock_split, 'question_type', ['en'])
