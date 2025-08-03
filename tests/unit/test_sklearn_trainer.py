# tests/unit/test_sklearn_trainer.py
"""
Unit tests for sklearn trainer integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.training.sklearn_trainer import SklearnTrainer


class TestSklearnTrainer:
    """Test sklearn trainer functionality."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        mock_model = Mock()
        trainer = SklearnTrainer(
            model=mock_model,
            task_type='classification'
        )
        
        assert trainer.model == mock_model
        assert trainer.task_type == 'classification'
    
    def test_trainer_classification_metrics(self):
        """Test calculation of classification metrics."""
        mock_model = Mock()
        trainer = SklearnTrainer(
            model=mock_model,
            task_type='classification'
        )
        
        # Mock predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        metrics = trainer._calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_trainer_regression_metrics(self):
        """Test calculation of regression metrics."""
        mock_model = Mock()
        trainer = SklearnTrainer(
            model=mock_model,
            task_type='regression'
        )
        
        # Mock predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9])
        
        metrics = trainer._calculate_metrics(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
