# src/training/sklearn_trainer.py
"""
Fixed sklearn trainer for baseline experiments.
Addresses unhashable numpy array issues and improves compatibility.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import time
import joblib
import json
import os
from src.evaluation.metrics import calculate_metrics, format_metrics_for_logging

logger = logging.getLogger(__name__)

class SklearnTrainer:
    """Fixed trainer for sklearn-based baseline models."""
    
    def __init__(
        self,
        model,
        task_type: str = "classification",
        output_dir: Optional[str] = None,
        wandb_run: Optional[Any] = None,
    ):
        """
        Initialize trainer with a sklearn model.
        
        Args:
            model: Sklearn model instance
            task_type: "classification" or "regression"
            output_dir: Directory to save results
            wandb_run: Weights & Biases run object (optional)
        """
        self.model = model
        self.task_type = task_type
        self.output_dir = output_dir
        self.wandb_run = wandb_run
        self.is_fitted = False
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized SklearnTrainer for {task_type} with model: {type(model).__name__}")
    
    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model and evaluate on all provided splits.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Optional tuple of (X_val, y_val)
            test_data: Optional tuple of (X_test, y_test)
            
        Returns:
            Dictionary with training results and metrics
        """
        X_train, y_train = train_data
        
        logger.info(f"Training {self.model.__class__.__name__} on {X_train.shape[0]} examples")
        logger.info(f"Features shape: {X_train.shape}")
        
        start_time = time.time()
        
        # Convert sparse matrices to dense for models that need it
        X_train_processed = self._prepare_features(X_train)
        y_train_processed = self._prepare_labels(y_train)
        
        # Handle XGBoost early stopping if applicable
        is_xgboost = "XGB" in self.model.__class__.__name__
        
        if is_xgboost and val_data is not None:
            X_val, y_val = val_data
            X_val_processed = self._prepare_features(X_val)
            y_val_processed = self._prepare_labels(y_val)
            
            try:
                self.model.fit(
                    X_train_processed,
                    y_train_processed,
                    eval_set=[(X_train_processed, y_train_processed), (X_val_processed, y_val_processed)],
                    eval_metric="logloss" if self.task_type == "classification" else "rmse",
                    verbose=False,  # Reduce verbose output
                )
                
                # Log XGBoost training curve if wandb available
                if self.wandb_run and hasattr(self.model, 'evals_result_'):
                    self._log_xgboost_training_curve()
                    
            except Exception as e:
                logger.warning(f"XGBoost early stopping failed: {e}, falling back to regular training")
                self.model.fit(X_train_processed, y_train_processed)
        else:
            # Regular training
            self.model.fit(X_train_processed, y_train_processed)
        
        self.is_fitted = True
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Evaluate on training data
        train_preds = self.model.predict(X_train_processed)
        train_metrics = self._calculate_metrics(y_train_processed, train_preds)
        logger.info(f"Training metrics: {format_metrics_for_logging(train_metrics)}")
        
        # Evaluate on validation data if provided
        val_metrics = None
        if val_data is not None:
            X_val, y_val = val_data
            X_val_processed = self._prepare_features(X_val)
            y_val_processed = self._prepare_labels(y_val)
            
            val_preds = self.model.predict(X_val_processed)
            val_metrics = self._calculate_metrics(y_val_processed, val_preds)
            logger.info(f"Validation metrics: {format_metrics_for_logging(val_metrics)}")
        
        # Evaluate on test data if provided
        test_metrics = None
        if test_data is not None:
            X_test, y_test = test_data
            X_test_processed = self._prepare_features(X_test)
            y_test_processed = self._prepare_labels(y_test)
            
            test_preds = self.model.predict(X_test_processed)
            test_metrics = self._calculate_metrics(y_test_processed, test_preds)
            logger.info(f"Test metrics: {test_metrics}")
        
        # Log feature importance for tree-based models
        if hasattr(self.model, "feature_importances_") and self.wandb_run:
            self._log_feature_importance()
        
        # Log metrics to wandb if available
        if self.wandb_run:
            wandb_metrics = {
                "train_time": train_time,
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }
            if val_metrics:
                wandb_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            if test_metrics:
                wandb_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
            
            self.wandb_run.log(wandb_metrics)
        
        # Compile results
        results = {
            "model_type": self.model.__class__.__name__,
            "task_type": self.task_type,
            "train_time": train_time,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        
        # Save results and model if output directory provided
        if self.output_dir:
            self._save_results(results)
            self._save_model()
        
        return results
    
    def _prepare_features(self, X):
        """Prepare features for the model (handle sparse matrices, etc.)."""
        # Convert sparse matrices to dense for models that require it
        model_name = self.model.__class__.__name__
        
        if model_name in ['DummyClassifier', 'DummyRegressor']:
            # Dummy models don't handle sparse matrices well
            if hasattr(X, 'toarray'):
                return X.toarray()
        
        return X
    
    def _prepare_labels(self, y):
        """Prepare labels for the model."""
        # Ensure labels are numpy array
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Handle potential data type issues
        if self.task_type == "classification":
            # Ensure integer labels for classification
            return y.astype(int)
        else:
            # Ensure float labels for regression
            return y.astype(float)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standardized metrics based on task type."""
        try:
            return calculate_metrics(y_true, y_pred, self.task_type)
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return fallback metrics
            if self.task_type == "classification":
                return {"primary_metric": "accuracy", "primary_value": 0.0, "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                return {"primary_metric": "mse", "primary_value": float('inf'), "mse": float('inf'), "r2": 0.0}
    
    def _log_xgboost_training_curve(self):
        """Log XGBoost training curve to wandb."""
        if not hasattr(self.model, 'evals_result_'):
            return
        
        try:
            evals_result = self.model.evals_result_
            
            # Get training metrics (first eval set)
            train_key = list(evals_result.keys())[0]
            metric_key = list(evals_result[train_key].keys())[0]
            train_metrics = evals_result[train_key][metric_key]
            
            # Log each epoch
            for i, train_val in enumerate(train_metrics):
                log_dict = {"epoch": i, f"train_{metric_key}": train_val}
                
                # Add validation metrics if available
                if len(evals_result) > 1:
                    val_key = list(evals_result.keys())[1]
                    val_metrics = evals_result[val_key][metric_key]
                    if i < len(val_metrics):
                        log_dict[f"val_{metric_key}"] = val_metrics[i]
                
                self.wandb_run.log(log_dict)
                
        except Exception as e:
            logger.warning(f"Could not log XGBoost training curve: {e}")
    
    def _log_feature_importance(self):
        """Log feature importance to wandb."""
        try:
            import wandb
            
            feature_importance = self.model.feature_importances_
            
            # Only log top features to avoid overwhelming wandb
            if len(feature_importance) > 50:
                top_indices = np.argsort(feature_importance)[-50:]
                top_importances = feature_importance[top_indices]
                feature_names = [f"feature_{i}" for i in top_indices]
            else:
                top_importances = feature_importance
                feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
            
            # Create wandb table
            importance_data = [[name, float(importance)] for name, importance in zip(feature_names, top_importances)]
            importance_table = wandb.Table(
                data=importance_data,
                columns=["feature", "importance"]
            )
            self.wandb_run.log({"feature_importance": importance_table})
            
        except Exception as e:
            logger.warning(f"Could not log feature importance: {e}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        try:
            results_path = os.path.join(self.output_dir, "results.json")
            
            # Convert any numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            results_serializable = convert_numpy_types(results)
            
            with open(results_path, "w") as f:
                json.dump(results_serializable, f, indent=2)
            
            logger.info(f"Results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Could not save results: {e}")
    
    def _save_model(self):
        """Save the trained model."""
        try:
            model_path = os.path.join(self.output_dir, "model.joblib")
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Log model as wandb artifact if available
            if self.wandb_run:
                try:
                    import wandb
                    model_artifact = wandb.Artifact(
                        name=f"model_{self.model.__class__.__name__}",
                        type="model",
                        description=f"Trained {self.model.__class__.__name__} for {self.task_type}",
                    )
                    model_artifact.add_file(model_path)
                    self.wandb_run.log_artifact(model_artifact)
                except Exception as e:
                    logger.warning(f"Could not log model artifact to wandb: {e}")
            
        except Exception as e:
            logger.error(f"Could not save model: {e}")
    
    def predict(self, X):
        """Make predictions with the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self._prepare_features(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """Get prediction probabilities (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {type(self.model).__name__} does not support predict_proba")
        
        X_processed = self._prepare_features(X)
        return self.model.predict_proba(X_processed)
    
    def evaluate(self, X, y, split_name: str = "eval") -> Dict[str, Any]:
        """Evaluate the model on given data."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X)
        metrics = self._calculate_metrics(y, y_pred)
        
        logger.info(f"{split_name} metrics: {metrics}")
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        info = {
            'model_class': self.model.__class__.__name__,
            'task_type': self.task_type,
            'is_fitted': self.is_fitted,
        }
        
        # Add model parameters if available
        if hasattr(self.model, 'get_params'):
            try:
                info['parameters'] = self.model.get_params()
            except Exception as e:
                logger.warning(f"Could not get model parameters: {e}")
        
        return info