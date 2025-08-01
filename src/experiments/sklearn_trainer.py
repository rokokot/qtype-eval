# src/experiments/sklearn_trainer.py
"""
Sklearn trainer for baseline experiments.
Compatible with existing experiment framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class SklearnTrainer:
    """Trainer for sklearn-based baseline models."""
    
    def __init__(self, model, task_type: str = "classification"):
        """
        Initialize trainer with a sklearn model.
        
        Args:
            model: Sklearn model instance
            task_type: "classification" or "regression"
        """
        self.model = model
        self.task_type = task_type
        self.is_fitted = False
        
    def train(self, X, y) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {type(self.model).__name__} on {X.shape[0]} examples")
        
        # Convert sparse matrices to dense if needed for certain models
        if hasattr(X, 'toarray'):
            # Check if model can handle sparse matrices
            model_name = type(self.model).__name__
            if model_name in ['DummyClassifier', 'DummyRegressor']:
                X = X.toarray()
        
        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Get training metrics
        train_metrics = self.evaluate(X, y, "train")
        
        logger.info(f"Training completed. Metrics: {train_metrics}")
        return train_metrics
    
    def evaluate(self, X, y, split_name: str = "eval") -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            X: Features
            y: True labels
            split_name: Name of the split (for logging)
            
        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            model_name = type(self.model).__name__
            if model_name in ['DummyClassifier', 'DummyRegressor']:
                X = X.toarray()
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        metrics = {}
        
        if self.task_type == "classification":
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y, y_pred)
            
            # Get precision, recall, f1 (handle different averaging scenarios)
            try:
                # Try macro averaging first
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y, y_pred, average='macro', zero_division=0
                )
                metrics['precision_macro'] = precision
                metrics['recall_macro'] = recall
                metrics['f1_macro'] = f1
                
                # Also compute weighted averages
                precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
                    y, y_pred, average='weighted', zero_division=0
                )
                metrics['precision_weighted'] = precision_w
                metrics['recall_weighted'] = recall_w
                metrics['f1_weighted'] = f1_w
                
            except Exception as e:
                logger.warning(f"Could not compute precision/recall/f1: {e}")
                metrics['precision_macro'] = 0.0
                metrics['recall_macro'] = 0.0
                metrics['f1_macro'] = 0.0
        
        elif self.task_type == "regression":
            # Regression metrics
            mse = mean_squared_error(y, y_pred)
            metrics['mse'] = mse
            metrics['rmse'] = np.sqrt(mse)
            
            try:
                r2 = r2_score(y, y_pred)
                metrics['r2'] = r2
            except Exception as e:
                logger.warning(f"Could not compute R²: {e}")
                metrics['r2'] = 0.0
        
        # Add some basic stats
        metrics['n_samples'] = len(y)
        metrics['n_features'] = X.shape[1] if hasattr(X, 'shape') else len(X[0]) if len(X) > 0 else 0
        
        logger.info(f"{split_name} metrics: {metrics}")
        return metrics
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            model_name = type(self.model).__name__
            if model_name in ['DummyClassifier', 'DummyRegressor']:
                X = X.toarray()
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {type(self.model).__name__} does not support predict_proba")
        
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            model_name = type(self.model).__name__
            if model_name in ['DummyClassifier']:
                X = X.toarray()
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if not self.is_fitted:
            return None
        
        # Try different attribute names for feature importance
        for attr in ['feature_importances_', 'coef_']:
            if hasattr(self.model, attr):
                importance = getattr(self.model, attr)
                if importance is not None:
                    return np.array(importance).flatten()
        
        return None
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        import joblib
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        import joblib
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        info = {
            'model_type': type(self.model).__name__,
            'task_type': self.task_type,
            'is_fitted': self.is_fitted,
        }
        
        # Add model-specific parameters
        if hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()
        
        return info