
# tests/unit/test_tfidf_baselines.py
"""
Unit tests for TfidfBaselineModel class.
"""

import pytest
import numpy as np
from src.models.tfidf_baselines import (
    TfidfBaselineModel, 
    create_tfidf_baseline_model,
    get_default_model_params,
    validate_model_task_compatibility
)
from src.data.tfidf_features import create_test_features


class TestTfidfBaselineModel:
    """Unit tests for TfidfBaselineModel."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self, tmp_path):
        """Set up test environment."""
        self.features_dir = tmp_path / "features"
        create_test_features(str(self.features_dir), n_samples=30)
        
    def test_model_creation_dummy(self):
        """Test creating dummy models."""
        for task_type in ['classification', 'regression']:
            model = create_tfidf_baseline_model(
                model_type='dummy',
                task_type=task_type,
                tfidf_features_dir=str(self.features_dir)
            )
            assert model.model_type == 'dummy'
            assert model.task_type == task_type
            assert not model.is_fitted
    
    def test_model_creation_logistic(self):
        """Test creating logistic regression model."""
        model = create_tfidf_baseline_model(
            model_type='logistic',
            task_type='classification',
            tfidf_features_dir=str(self.features_dir)
        )
        assert model.model_type == 'logistic'
        assert model.task_type == 'classification'
    
    def test_model_creation_ridge(self):
        """Test creating ridge regression model."""
        model = create_tfidf_baseline_model(
            model_type='ridge',
            task_type='regression',
            tfidf_features_dir=str(self.features_dir)
        )
        assert model.model_type == 'ridge'
        assert model.task_type == 'regression'
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError):
            create_tfidf_baseline_model(
                model_type='invalid',
                task_type='classification',
                tfidf_features_dir=str(self.features_dir)
            )
    
    def test_invalid_task_type(self):
        """Test invalid task type raises error."""
        with pytest.raises(ValueError):
            create_tfidf_baseline_model(
                model_type='dummy',
                task_type='invalid',
                tfidf_features_dir=str(self.features_dir)
            )
    
    def test_incompatible_model_task(self):
        """Test incompatible model-task combinations."""
        with pytest.raises(ValueError):
            create_tfidf_baseline_model(
                model_type='logistic',
                task_type='regression',
                tfidf_features_dir=str(self.features_dir)
            )
    
    def test_model_training_classification(self):
        """Test model training for classification."""
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=str(self.features_dir)
        )
        
        # Load features to get sample size
        model.load_features()
        n_train = model.features['train'].shape[0]
        
        # Generate labels and train
        train_labels = np.random.randint(0, 2, n_train)
        model.fit(train_labels)
        
        assert model.is_fitted
    
    def test_model_training_regression(self):
        """Test model training for regression."""
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='regression',
            tfidf_features_dir=str(self.features_dir)
        )
        
        # Load features to get sample size
        model.load_features()
        n_train = model.features['train'].shape[0]
        
        # Generate labels and train
        train_labels = np.random.randn(n_train)
        model.fit(train_labels)
        
        assert model.is_fitted
    
    def test_model_prediction(self):
        """Test model prediction."""
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=str(self.features_dir)
        )
        
        # Train model
        model.load_features()
        n_train = model.features['train'].shape[0]
        train_labels = np.random.randint(0, 2, n_train)
        model.fit(train_labels)
        
        # Test predictions
        for split in ['train', 'val', 'test']:
            preds = model.predict(split)
            expected_length = model.features[split].shape[0]
            assert len(preds) == expected_length
    
    def test_model_evaluation_classification(self):
        """Test model evaluation for classification."""
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=str(self.features_dir)
        )
        
        # Train and evaluate
        model.load_features()
        n_train = model.features['train'].shape[0]
        n_test = model.features['test'].shape[0]
        
        train_labels = np.random.randint(0, 2, n_train)
        test_labels = np.random.randint(0, 2, n_test)
        
        model.fit(train_labels)
        metrics = model.evaluate(test_labels, 'test')
        
        # Check required metrics
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_model_evaluation_regression(self):
        """Test model evaluation for regression."""
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='regression',
            tfidf_features_dir=str(self.features_dir)
        )
        
        # Train and evaluate
        model.load_features()
        n_train = model.features['train'].shape[0]
        n_test = model.features['test'].shape[0]
        
        train_labels = np.random.randn(n_train)
        test_labels = np.random.randn(n_test)
        
        model.fit(train_labels)
        metrics = model.evaluate(test_labels, 'test')
        
        # Check required metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=str(self.features_dir)
        )
        
        info = model.get_model_info()
        assert isinstance(info, dict)
        assert info['model_type'] == 'dummy'
        assert info['task_type'] == 'classification'
        assert 'is_fitted' in info


class TestTfidfUtilityFunctions:
    """Test utility functions."""
    
    def test_get_default_model_params(self):
        """Test default parameter retrieval."""
        # Test dummy parameters
        params = get_default_model_params('dummy', 'classification')
        assert 'strategy' in params
        assert params['strategy'] == 'most_frequent'
        
        params = get_default_model_params('dummy', 'regression')
        assert 'strategy' in params
        assert params['strategy'] == 'mean'
        
        # Test logistic parameters
        params = get_default_model_params('logistic', 'classification')
        assert 'C' in params
        assert 'solver' in params
        
        # Test ridge parameters
        params = get_default_model_params('ridge', 'regression')
        assert 'alpha' in params
    
    def test_validate_model_task_compatibility(self):
        """Test model-task compatibility validation."""
        # Valid combinations
        assert validate_model_task_compatibility('dummy', 'classification')
        assert validate_model_task_compatibility('dummy', 'regression')
        assert validate_model_task_compatibility('logistic', 'classification')
        assert validate_model_task_compatibility('ridge', 'regression')
        
        # Invalid combinations
        assert not validate_model_task_compatibility('logistic', 'regression')
        assert not validate_model_task_compatibility('ridge', 'classification')
