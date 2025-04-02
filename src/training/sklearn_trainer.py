# trainer class for sklearn model


import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import logging
import time
import joblib
import wandb
import json
import os

logger = logging.getLogger(__name__)

class SklearnTrainer:
     
    def __init__(
        self,
        model,
        task_type: str = "classification",
        output_dir: Optional[str] = None,
        wandb_run: Optional[Any] = None):
        self.model = model
        self.task_type = task_type
        self.output_dir = output_dir
        self.wandb_run = wandb_run 
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def train(self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
  
        X_train, y_train = train_data
        
      
        logger.info(f"Training {self.model.__class__.__name__} on {X_train.shape[0]} examples")
        logger.info(f"Features shape: {X_train.shape}")
        
      
        start_time = time.time()
        
   
        is_xgboost = "XGB" in self.model.__class__.__name__
        
        if is_xgboost and val_data is not None:
            X_val, y_val = val_data
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric='logloss' if self.task_type == "classification" else 'rmse',
                verbose=True)
            
            if self.wandb_run:
                evals_result = self.model.evals_result()
                train_metrics = list(evals_result['validation_0'].values())[0]
                val_metrics = list(evals_result['validation_1'].values())[0]
                
                for i, (train_val, val_val) in enumerate(zip(train_metrics, val_metrics)):
                    self.wandb_run.log({'epoch': i,'train_metric': train_val,'val_metric': val_val})
        else:
            self.model.fit(X_train, y_train)
        

        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        train_preds = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_preds)
        

        logger.info(f"Training metrics: {train_metrics}")

        if is_xgboost and hasattr(self.model, 'feature_importances_') and self.wandb_run:
            feature_importance = self.model.feature_importances_
            
            if len(feature_importance) > 50:
                top_indices = np.argsort(feature_importance)[-50:]
                top_importances = feature_importance[top_indices]
                feature_names = [f"feature_{i}" for i in top_indices]
            else:
                top_importances = feature_importance
                feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
            
            importance_table = wandb.Table(
                data=[[name, importance] for name, importance in zip(feature_names, top_importances)],
                columns=["feature", "importance"]
            )
            self.wandb_run.log({"feature_importance": importance_table})



        val_metrics = None
        if val_data is not None:
            X_val, y_val = val_data
            val_preds = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_preds)
            
            logger.info(f"Validation metrics: {val_metrics}")
        
    
        test_metrics = None
        if test_data is not None:
            X_test, y_test = test_data
            test_preds = self.model.predict(X_test)
            test_metrics = self._calculate_metrics(y_test, test_preds)
            
            logger.info(f"Test metrics: {test_metrics}")
        
        if self.wandb_run:
            self.wandb_run.log({"train_time": train_time,**{f"train_{k}": v for k, v in train_metrics.items()},**({f"val_{k}": v for k, v in val_metrics.items()} if val_metrics else {}),**({f"test_{k}": v for k, v in test_metrics.items()} if test_metrics else {})})
        

        results = {
            "model_type": self.model.__class__.__name__,
            "task_type": self.task_type,
            "train_time": train_time,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics}
        
        if self.output_dir:
            with open(os.path.join(self.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
                

            try:
                joblib.dump(self.model, os.path.join(self.output_dir, "model.joblib"))
                logger.info(f"Model saved to {self.output_dir}/model.joblib")

                if self.wandb_run:
                    model_artifact = wandb.Artifact(name=f"model_{self.model.__class__.__name__}",type="model",description=f"Trained {self.model.__class__.__name__} for {self.task_type}")
                    model_artifact.add_file(os.path.join(self.output_dir, "model.joblib"))
                    self.wandb_run.log_artifact(model_artifact)
            except Exception as e:
                logger.warning(f"Could not save model: {e}")
        
        return results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray) -> Dict[str, float]:
       
        if self.task_type == "classification":
         
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, average="binary"))}
        else:  
          
            metrics = {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred))}
        
        return metrics