# src/models/tfidf_baselines.py
"""
 TF-IDF baseline models with 
 multiple sklearn models with proper configuration and evaluation.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import logging
from pathlib import Path

# Sklearn imports
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error
)

# Try to import XGBoost (optional dependency)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

from ..data.tfidf_features import TfidfFeatureLoader

logger = logging.getLogger(__name__)

class TfidfBaselineModel:
    """
    Wrapper for sklearn models to work with TF-IDF features.
    Handles feature loading, model training, and evaluation.
    """
    
    def __init__(
        self, 
        model_type: str, 
        task_type: str,
        tfidf_features_dir: str,
        target_languages: List[str] = ['all'],
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """
        Initialize TF-IDF baseline model.
        
        Args:
            model_type: Type of sklearn model ('dummy', 'logistic', 'ridge', 'random_forest', 'xgboost')
            task_type: 'classification' or 'regression'
            tfidf_features_dir: Directory containing TF-IDF features
            target_languages: List of language codes to include
            model_params: Additional parameters for the model
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.task_type = task_type
        self.target_languages = target_languages
        self.model_params = model_params or {}
        self.random_state = random_state
        
        # Initialize feature loader
        self.tfidf_loader = TfidfFeatureLoader(tfidf_features_dir)
        
        # Create the sklearn model
        self.model = self._create_model()
        
        # Storage for loaded features
        self.features = None
        self.is_fitted = False
        
        logger.info(f"Created {model_type} model for {task_type} with languages: {target_languages}")
    
    def _create_model(self):
        """Create the appropriate sklearn model based on model_type and task_type."""
        
        # Validate model type
        valid_models = ['dummy', 'logistic', 'ridge', 'random_forest', 'xgboost']
        if self.model_type not in valid_models:
            raise ValueError(f"Unknown model type: {self.model_type}. Valid types: {valid_models}")
        
        # Check XGBoost availability
        if self.model_type == 'xgboost' and not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Please install it: pip install xgboost")
        
        # Validate task type
        if self.task_type not in ['classification', 'regression']:
            raise ValueError(f"Unknown task type: {self.task_type}. Must be 'classification' or 'regression'")
        
        # Validate model-task compatibility
        if self.model_type == 'logistic' and self.task_type != 'classification':
            raise ValueError("Logistic regression is only for classification tasks")
        
        # Add random state to model parameters
        params = self.model_params.copy()
        if 'random_state' not in params and self.model_type != 'dummy':
            params['random_state'] = self.random_state
        
        # Create model based on type
        if self.model_type == "dummy":
            if self.task_type == "classification":
                default_strategy = "most_frequent"
                return DummyClassifier(strategy=params.get('strategy', default_strategy))
            else:
                default_strategy = "mean"
                return DummyRegressor(strategy=params.get('strategy', default_strategy))
        
        elif self.model_type == "logistic":
            # Classification only
            default_params = {
                'max_iter': 1000,
                'solver': 'liblinear',  # Good for small datasets
                'C': 1.0
            }
            default_params.update(params)
            return LogisticRegression(**default_params)
        
        elif self.model_type == "ridge":
            # Regression only (Ridge regression)
            default_params = {'alpha': 1.0}
            default_params.update(params)
            return Ridge(**default_params)
        
        elif self.model_type == "random_forest":
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
            default_params.update(params)
            
            if self.task_type == "classification":
                return RandomForestClassifier(**default_params)
            else:
                return RandomForestRegressor(**default_params)
        
        elif self.model_type == "xgboost":
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 1.0,
                'colsample_bytree': 1.0
            }
            default_params.update(params)
            
            if self.task_type == "classification":
                # Additional XGBoost-specific params for classification
                default_params.update({
                    'eval_metric': 'logloss',
                    'use_label_encoder': False
                })
                return xgb.XGBClassifier(**default_params)
            else:
                # Additional XGBoost-specific params for regression
                default_params.update({
                    'eval_metric': 'rmse'
                })
                return xgb.XGBRegressor(**default_params)
        
        else:
            raise ValueError(f"Model creation not implemented for: {self.model_type}")
    
    def load_features(self) -> None:
        """Load TF-IDF features from the feature loader."""
        logger.info("Loading TF-IDF features...")
        
        # Load all features
        all_features = self.tfidf_loader.load_all_features()
        
        # Filter by target languages
        self.features = self.tfidf_loader.filter_by_languages(all_features, self.target_languages)
        
        # Log feature information
        for split, matrix in self.features.items():
            logger.info(f"Loaded {split} features: {matrix.shape[0]} samples × {matrix.shape[1]} features")
            sparsity = 1 - (matrix.nnz / np.prod(matrix.shape))
            logger.info(f"  {split} sparsity: {sparsity:.2%}")
    
    def fit(self, X=None, y=None, **kwargs) -> 'TfidfBaselineModel':
        """
        Train the model on TF-IDF features.
        
        Args:
            X: Features (ignored, we use our own TF-IDF features) 
            y: Training labels (can be passed as first or second argument)
            **kwargs: Additional arguments
            
        Returns:
            Self for chaining
        """
        # Handle different calling conventions
        if X is not None and y is None:
            # Called as fit(y_train) - X is actually y_train
            y_train = X
        elif X is not None and y is not None:
            # Called as fit(X_train, y_train) - use y_train
            y_train = y
        else:
            raise ValueError("Labels must be provided")
        
        if self.features is None:
            self.load_features()
        
        X_train = self.features['train']
        
        logger.info(f"Training {self.model_type} on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        # Convert sparse matrices to dense for models that require it
        if self._requires_dense_input():
            X_train = X_train.toarray()
            logger.info(f"Converted sparse matrix to dense for {self.model_type}")
        
        # Train the model
        logger.info("Training started...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        logger.info(f"Training completed for {self.model_type}")
        return self
    
    def _requires_dense_input(self) -> bool:
        """Check if the model requires dense input (doesn't support sparse matrices)."""
        # Most sklearn models support sparse matrices, but some exceptions:
        dense_only_models = ['DummyClassifier', 'DummyRegressor']
        return self.model.__class__.__name__ in dense_only_models
    
    def predict(self, split: str = 'test') -> np.ndarray:
        """
        Make predictions on a specific split.
        
        Args:
            split: Split to predict on ('train', 'val', 'test')
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.features is None:
            self.load_features()
        
        if split not in self.features:
            raise ValueError(f"Split '{split}' not found in loaded features")
        
        X = self.features[split]
        
        # Convert to dense if needed
        if self._requires_dense_input():
            X = X.toarray()
        
        return self.model.predict(X)
    
    def predict_proba(self, split: str = 'test') -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            split: Split to predict on
            
        Returns:
            Prediction probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {self.model_type} does not support predict_proba")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.features is None:
            self.load_features()
        
        X = self.features[split]
        
        # Convert to dense if needed
        if self._requires_dense_input():
            X = X.toarray()
        
        return self.model.predict_proba(X)
    
    def evaluate(self, y_true: np.ndarray, split: str = 'test') -> Dict[str, float]:
        """
        Evaluate the model on a specific split.
        
        Args:
            y_true: True labels
            split: Split to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(split)
        
        if self.task_type == "classification":
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
                'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            }
            
            # Add per-class metrics if binary classification
            if len(np.unique(y_true)) == 2:
                metrics.update({
                    'f1_binary': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
                    'precision_binary': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
                    'recall_binary': float(recall_score(y_true, y_pred, average='binary', zero_division=0))
                })
        
        else:  # regression
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
        
        # Add basic statistics
        metrics.update({
            'n_samples': len(y_true),
            'n_features': self.features[split].shape[1] if self.features else 0
        })
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None if not available
        """
        if not self.is_fitted:
            logger.warning("Model not fitted - cannot get feature importance")
            return None
        
        # Try different attribute names for feature importance
        importance_attrs = ['feature_importances_', 'coef_']
        
        for attr in importance_attrs:
            if hasattr(self.model, attr):
                importance = getattr(self.model, attr)
                if importance is not None:
                    # Handle different shapes (1D vs 2D arrays)
                    if importance.ndim > 1:
                        # For binary classification, sklearn might return (1, n_features)
                        importance = importance.flatten()
                    return np.array(importance)
        
        logger.info(f"Feature importance not available for {self.model_type}")
        return None
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        import joblib
        
        # Create save object with model and metadata
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'target_languages': self.target_languages,
            'model_params': self.model_params,
            'random_state': self.random_state,
            'vocab_size': self.get_vocab_size(),
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load a pre-trained model.
        
        Args:
            filepath: Path to load the model from
        """
        import joblib
        
        save_data = joblib.load(filepath)
        
        # Restore model and metadata
        self.model = save_data['model']
        self.model_type = save_data['model_type']
        self.task_type = save_data['task_type']
        self.target_languages = save_data['target_languages']
        self.model_params = save_data['model_params']
        self.random_state = save_data.get('random_state', 42)
        self.is_fitted = save_data.get('is_fitted', True)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': self.model_type,
            'task_type': self.task_type,
            'target_languages': self.target_languages,
            'is_fitted': self.is_fitted,
            'sklearn_model': self.model.__class__.__name__,
            'model_params': self.model_params,
            'random_state': self.random_state
        }
        
        # Add model-specific parameters if fitted
        if self.is_fitted and hasattr(self.model, 'get_params'):
            info['fitted_params'] = self.model.get_params()
        
        # Add feature information if loaded
        if self.features is not None:
            info['vocab_size'] = self.get_vocab_size()
            info['feature_shapes'] = {
                split: list(matrix.shape) for split, matrix in self.features.items()
            }
        
        # Add TF-IDF loader metadata
        tfidf_metadata = self.tfidf_loader.get_metadata()
        if tfidf_metadata:
            info['tfidf_metadata'] = tfidf_metadata
        
        return info
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.features is not None:
            # Get from loaded features
            return next(iter(self.features.values())).shape[1]
        else:
            # Get from TF-IDF loader metadata
            return self.tfidf_loader.get_vocab_size()


def create_tfidf_baseline_model(
    model_type: str,
    task_type: str, 
    tfidf_features_dir: str,
    target_languages: List[str] = ['all'],
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> TfidfBaselineModel:
    """
    Factory function to create TF-IDF baseline models.
    
    Args:
        model_type: Type of sklearn model to create
        task_type: 'classification' or 'regression'
        tfidf_features_dir: Directory containing TF-IDF features
        target_languages: List of language codes to include
        model_params: Additional parameters for the model
        random_state: Random seed for reproducibility
        
    Returns:
        Configured TfidfBaselineModel instance
    """
    return TfidfBaselineModel(
        model_type=model_type,
        task_type=task_type,
        tfidf_features_dir=tfidf_features_dir,
        target_languages=target_languages,
        model_params=model_params,
        random_state=random_state
    )


def get_default_model_params(model_type: str, task_type: str) -> Dict[str, Any]:
    """
    Get default parameters for different model types and tasks.
    
    Args:
        model_type: Type of model
        task_type: Type of task
        
    Returns:
        Dictionary of default parameters
    """
    defaults = {
        'dummy': {
            'classification': {'strategy': 'most_frequent'},
            'regression': {'strategy': 'mean'}
        },
        'logistic': {
            'classification': {
                'C': 1.0,
                'solver': 'liblinear',
                'max_iter': 1000,
                'class_weight': None
            }
        },
        'ridge': {
            'regression': {
                'alpha': 1.0,
                'solver': 'auto'
            }
        },
        'random_forest': {
            'classification': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'class_weight': None
            },
            'regression': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        },
        'xgboost': {
            'classification': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            },
            'regression': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'eval_metric': 'rmse'
            }
        }
    }
    
    if model_type not in defaults:
        return {}
    
    model_defaults = defaults[model_type]
    if task_type in model_defaults:
        return model_defaults[task_type].copy()
    
    # Return generic parameters if task-specific not found
    return next(iter(model_defaults.values())).copy() if model_defaults else {}


def validate_model_task_compatibility(model_type: str, task_type: str) -> bool:
    """
    Validate that a model type is compatible with a task type.
    
    Args:
        model_type: Type of model
        task_type: Type of task
        
    Returns:
        True if compatible, False otherwise
    """
    # Models that only work with specific tasks
    classification_only = ['logistic']
    regression_only = ['ridge']
    
    if model_type in classification_only and task_type != 'classification':
        return False
    
    if model_type in regression_only and task_type != 'regression':
        return False
    
    return True


def run_tfidf_baseline_experiment(
    model_type: str,
    task_type: str,
    tfidf_features_dir: str,
    train_labels: np.ndarray,
    val_labels: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    target_languages: List[str] = ['all'],
    model_params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a complete TF-IDF baseline experiment.
    
    Args:
        model_type: Type of sklearn model
        task_type: 'classification' or 'regression'
        tfidf_features_dir: Directory with TF-IDF features
        train_labels: Training labels
        val_labels: Validation labels (optional)
        test_labels: Test labels (optional)
        target_languages: Languages to include
        model_params: Model parameters
        output_dir: Directory to save results
        
    Returns:
        Dictionary with experiment results
    """
    # Validate compatibility
    if not validate_model_task_compatibility(model_type, task_type):
        raise ValueError(f"Model type '{model_type}' not compatible with task type '{task_type}'")
    
    # Create model
    model = create_tfidf_baseline_model(
        model_type=model_type,
        task_type=task_type,
        tfidf_features_dir=tfidf_features_dir,
        target_languages=target_languages,
        model_params=model_params
    )
    
    # Train model
    model.fit(train_labels)
    
    # Evaluate on all available splits
    results = {
        'model_type': model_type,
        'task_type': task_type,
        'target_languages': target_languages,
        'model_info': model.get_model_info()
    }
    
    # Training metrics
    train_metrics = model.evaluate(train_labels, 'train')
    results['train_metrics'] = train_metrics
    
    # Validation metrics
    if val_labels is not None:
        val_metrics = model.evaluate(val_labels, 'val')
        results['val_metrics'] = val_metrics
    
    # Test metrics
    if test_labels is not None:
        test_metrics = model.evaluate(test_labels, 'test')
        results['test_metrics'] = test_metrics
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance is not None:
        results['feature_importance'] = {
            'values': feature_importance.tolist(),
            'top_10_indices': np.argsort(np.abs(feature_importance))[-10:].tolist()
        }
    
    # Save results if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_model(output_path / "model.joblib")
        
        # Save results
        import json
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Experiment results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    # Test the TF-IDF baseline models
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TF-IDF baseline models")
    parser.add_argument("--features-dir", required=True, help="Directory containing TF-IDF features")
    parser.add_argument("--model-type", default="dummy", choices=['dummy', 'logistic', 'ridge', 'random_forest', 'xgboost'])
    parser.add_argument("--task-type", default="classification", choices=['classification', 'regression'])
    parser.add_argument("--languages", nargs="+", default=['all'], help="Target languages")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create and test model
    model = create_tfidf_baseline_model(
        model_type=args.model_type,
        task_type=args.task_type,
        tfidf_features_dir=args.features_dir,
        target_languages=args.languages
    )
    
    # Load features to test
    model.load_features()
    
    # Generate dummy labels for testing
    n_train = model.features['train'].shape[0]
    if args.task_type == 'classification':
        train_labels = np.random.randint(0, 2, n_train)
    else:
        train_labels = np.random.randn(n_train)
    
    # Test fitting
    model.fit(train_labels)
    
    # Test prediction
    train_preds = model.predict('train')
    print(f" Training predictions shape: {train_preds.shape}")
    
    # Test evaluation
    train_metrics = model.evaluate(train_labels, 'train')
    print(f" Training metrics: {train_metrics}")
    
    # Test model info
    info = model.get_model_info()
    print(f" Model info: {info['model_type']} with {info.get('vocab_size', 'unknown')} features")
    
    print(" All tests passed!")