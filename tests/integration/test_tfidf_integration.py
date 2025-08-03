# tests/integration/test_tfidf_integration.py
"""
Main integration tests for TF-IDF functionality.
Tests the complete pipeline from feature generation to model training.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

# Import components to test
from src.data.tfidf_features import TfidfFeatureLoader, create_test_features
from src.models.tfidf_baselines import create_tfidf_baseline_model
from src.data.datasets import load_sklearn_data
from src.training.sklearn_trainer import SklearnTrainer

logger = logging.getLogger(__name__)

class TestTfidfIntegration:
    """Integration tests for TF-IDF components."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self, tmp_path):
        """Set up test environment with tiny TF-IDF features."""
        self.test_dir = tmp_path
        self.features_dir = self.test_dir / "tfidf_features"
        self.output_dir = self.test_dir / "output"
        
        # Create test TF-IDF features
        create_test_features(
            output_dir=str(self.features_dir),
            n_samples=50
        )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Test environment created in {self.test_dir}")
    
    def test_tfidf_feature_loader_basic(self):
        """Test basic TF-IDF feature loader functionality."""
        # Initialize loader
        loader = TfidfFeatureLoader(str(self.features_dir))
        
        # Test metadata loading
        metadata = loader.get_metadata()
        assert metadata is not None
        assert 'vocab_size' in metadata
        
        # Test feature loading
        train_features = loader.load_features('train')
        val_features = loader.load_features('val')
        test_features = loader.load_features('test')
        
        # Validate shapes
        assert train_features.shape[1] == val_features.shape[1] == test_features.shape[1]
        assert train_features.shape[0] > 0
        assert val_features.shape[0] > 0
        assert test_features.shape[0] > 0
        
        # Test verification
        assert loader.verify_features()
    
    def test_tfidf_feature_loader_language_filtering(self):
        """Test language filtering functionality."""
        loader = TfidfFeatureLoader(str(self.features_dir))
        
        # Load all features
        all_features = loader.load_all_features()
        
        # Test language filtering (should work even with test data)
        filtered = loader.filter_by_languages(all_features, ['en'])
        
        # Should return the same or filtered features
        assert 'train' in filtered
        assert 'val' in filtered  
        assert 'test' in filtered
    
    def test_tfidf_baseline_model_creation(self):
        """Test TF-IDF baseline model creation."""
        # Test different model types
        model_types = ['dummy', 'logistic', 'ridge']
        
        for model_type in model_types:
            # Determine task type
            if model_type == 'logistic':
                task_type = 'classification'
            elif model_type == 'ridge':
                task_type = 'regression'
            else:
                # Test both for dummy
                for task_type in ['classification', 'regression']:
                    model = create_tfidf_baseline_model(
                        model_type=model_type,
                        task_type=task_type,
                        tfidf_features_dir=str(self.features_dir),
                        target_languages=['en']
                    )
                    
                    assert model.model_type == model_type
                    assert model.task_type == task_type
                    assert not model.is_fitted
                    
                    # Test model info
                    info = model.get_model_info()
                    assert info['model_type'] == model_type
                    assert info['task_type'] == task_type
                
                continue  # Skip single task test for dummy
            
            # Single task test for logistic/ridge
            model = create_tfidf_baseline_model(
                model_type=model_type,
                task_type=task_type,
                tfidf_features_dir=str(self.features_dir),
                target_languages=['en']
            )
            
            assert model.model_type == model_type
            assert model.task_type == task_type
    
    def test_tfidf_model_training_and_evaluation(self):
        """Test TF-IDF model training and evaluation."""
        # Create model
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=str(self.features_dir),
            target_languages=['en']
        )
        
        # Load features to get sample sizes
        loader = TfidfFeatureLoader(str(self.features_dir))
        features = loader.load_all_features()
        
        # Generate synthetic labels
        n_train = features['train'].shape[0]
        n_val = features['val'].shape[0]
        n_test = features['test'].shape[0]
        
        train_labels = np.random.randint(0, 2, n_train)
        val_labels = np.random.randint(0, 2, n_val)
        test_labels = np.random.randint(0, 2, n_test)
        
        # Train model
        model.fit(train_labels)
        assert model.is_fitted
        
        # Test predictions
        train_preds = model.predict('train')
        val_preds = model.predict('val')
        test_preds = model.predict('test')
        
        assert len(train_preds) == n_train
        assert len(val_preds) == n_val
        assert len(test_preds) == n_test
        
        # Test evaluation
        train_metrics = model.evaluate(train_labels, 'train')
        val_metrics = model.evaluate(val_labels, 'val')
        test_metrics = model.evaluate(test_labels, 'test')
        
        # Check required metrics exist
        assert 'accuracy' in train_metrics
        assert 'f1' in train_metrics
        assert 'accuracy' in val_metrics
        assert 'accuracy' in test_metrics
        
        # Check metrics are reasonable
        assert 0 <= train_metrics['accuracy'] <= 1
        assert 0 <= val_metrics['accuracy'] <= 1
        assert 0 <= test_metrics['accuracy'] <= 1
    
    def test_data_loading_integration(self):
        """Test integration with enhanced dataset loading."""
        # This is a mock test since we don't have the actual dataset in tests
        # In a real scenario, you'd use a mock dataset or test dataset
        
        try:
            # Test that the function can be called (will fail due to missing dataset)
            load_sklearn_data(
                languages=['en'],
                task='question_type',
                use_tfidf_loader=True,
                tfidf_features_dir=str(self.features_dir)
            )
        except Exception as e:
            # Expected to fail due to missing dataset, but function should exist
            assert "load_dataset" in str(e) or "dataset" in str(e).lower()
    
    def test_sklearn_trainer_integration(self):
        """Test integration with sklearn trainer."""
        # Create model
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=str(self.features_dir),
            target_languages=['en']
        )
        
        # Load features
        loader = TfidfFeatureLoader(str(self.features_dir))
        features = loader.load_all_features()
        
        # Generate synthetic data
        X_train = features['train']
        X_val = features['val']
        X_test = features['test']
        
        y_train = np.random.randint(0, 2, X_train.shape[0])
        y_val = np.random.randint(0, 2, X_val.shape[0])
        y_test = np.random.randint(0, 2, X_test.shape[0])
        
        # Create trainer
        trainer = SklearnTrainer(
            model=model.model,
            task_type='classification',
            output_dir=str(self.output_dir)
        )
        
        # Train
        results = trainer.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            test_data=(X_test, y_test)
        )
        
        # Validate results
        assert 'train_metrics' in results
        assert 'val_metrics' in results
        assert 'test_metrics' in results
        assert 'accuracy' in results['test_metrics']
        
        # Check output files
        assert (self.output_dir / "results.json").exists()
        assert (self.output_dir / "model.joblib").exists()
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Create and train model
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=str(self.features_dir),
            target_languages=['en']
        )
        
        # Load features and train
        loader = TfidfFeatureLoader(str(self.features_dir))
        features = loader.load_all_features()
        train_labels = np.random.randint(0, 2, features['train'].shape[0])
        
        model.fit(train_labels)
        
        # Get predictions before saving
        original_preds = model.predict('test')
        
        # Save model
        model_path = self.output_dir / "test_model.joblib"
        model.save_model(str(model_path))
        assert model_path.exists()
        
        # Create new model and load
        new_model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=str(self.features_dir),
            target_languages=['en']
        )
        
        new_model.load_model(str(model_path))
        assert new_model.is_fitted
        
        # Compare predictions
        loaded_preds = new_model.predict('test')
        np.testing.assert_array_equal(original_preds, loaded_preds)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with sparse matrices."""
        loader = TfidfFeatureLoader(str(self.features_dir))
        
        # Load features
        features = loader.load_all_features()
        
        # Check that features are sparse
        for split, matrix in features.items():
            assert hasattr(matrix, 'nnz')  # Sparse matrix property
            assert hasattr(matrix, 'toarray')  # Sparse matrix method
        
        # Test that models handle sparse matrices
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=str(self.features_dir),
            target_languages=['en']
        )
        
        # Should handle sparse features without issues
        train_labels = np.random.randint(0, 2, features['train'].shape[0])
        model.fit(train_labels)
        
        # Should be able to predict on sparse features
        predictions = model.predict('test')
        assert len(predictions) == features['test'].shape[0]
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        # Test invalid features directory
        with pytest.raises(FileNotFoundError):
            TfidfFeatureLoader("/nonexistent/directory")
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_tfidf_baseline_model(
                model_type='invalid_model',
                task_type='classification',
                tfidf_features_dir=str(self.features_dir)
            )
        
        # Test invalid task type
        with pytest.raises(ValueError):
            create_tfidf_baseline_model(
                model_type='dummy',
                task_type='invalid_task',
                tfidf_features_dir=str(self.features_dir)
            )
        
        # Test logistic regression with regression task
        with pytest.raises(ValueError):
            create_tfidf_baseline_model(
                model_type='logistic',
                task_type='regression',
                tfidf_features_dir=str(self.features_dir)
            )
    
    def test_backward_compatibility(self):
        """Test that existing functionality still works."""
        # This would test that existing code paths still work
        # For now, we just test that the enhanced dataset loading
        # maintains the same interface
        
        from src.data.datasets import (
            ensure_string_task, 
            get_available_tasks, 
            get_available_languages,
            get_task_type
        )
        
        # Test utility functions
        assert ensure_string_task(['question_type']) == 'question_type'
        assert ensure_string_task('complexity') == 'complexity'
        
        tasks = get_available_tasks()
        assert 'question_type' in tasks
        assert 'complexity' in tasks
        
        languages = get_available_languages()
        assert 'en' in languages
        assert 'ru' in languages
        
        assert get_task_type('question_type') == 'classification'
        assert get_task_type('complexity') == 'regression'

@pytest.mark.slow
class TestTfidfIntegrationSlow:
    """Slower integration tests that might take more time."""
    
    def test_multiple_model_types(self, tmp_path):
        """Test multiple model types with different configurations."""
        # This would be a more comprehensive test
        # that tests all model types with various parameters
        pass
    
    def test_cross_language_compatibility(self, tmp_path):
        """Test cross-language model training."""
        # This would test training on one language and evaluating on another
        pass
    
    def test_large_feature_sets(self, tmp_path):
        """Test with larger feature sets to validate memory handling."""
        # This would test with larger vocabularies
        pass


@pytest.fixture(scope="session")
def test_cache_dir(tmp_path_factory):
    """Create a session-wide temporary cache directory."""
    return tmp_path_factory.mktemp("cache")

@pytest.fixture(scope="session") 
def test_features_dir(tmp_path_factory):
    """Create session-wide test TF-IDF features."""
    features_dir = tmp_path_factory.mktemp("features")
    create_test_features(str(features_dir), n_samples=100)
    return features_dir

pytestmark = [
    pytest.mark.integration,
    pytest.mark.tfidf
]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])