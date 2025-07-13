

# src/models/tfidf_baselines.py
"""
todo:

check seed
import format + metadata issues for models
parameters should be configurable from configs

"""

from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import xgboost as xgb

from ..data.tfidf_features import TfidfFeatureLoader

class TfidfBaselineModel:
    
    def __init__(
        self, 
        model_type: str, 
        task_type: str,
        tfidf_loader: TfidfFeatureLoader,
        target_languages: List[str] = ['all'],
        model_params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.task_type = task_type
        self.tfidf_loader = tfidf_loader
        self.target_languages = target_languages
        self.model_params = model_params or {}
        
        self.model = self._create_model()
        self.features = None
    
    def _create_model(self):
        
        if self.model_type == "dummy":
            if self.task_type == "classification":
                return DummyClassifier(strategy="most_frequent", **self.model_params)
            else:
                return DummyRegressor(strategy="mean", **self.model_params)
        
        elif self.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000, 
                random_state=42,
                **self.model_params            )
        

        elif self.model_type == "ridge":
            return Ridge(random_state=42, **self.model_params)
        
        elif self.model_type == "random_forest":
            if self.task_type == "classification":
                return RandomForestClassifier(random_state=42, **self.model_params)
            else:
                return RandomForestRegressor(random_state=42, **self.model_params)
        
        elif self.model_type == "xgboost":
            if self.task_type == "classification":
                return xgb.XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    **self.model_params)
            else:
                return xgb.XGBRegressor(
                    random_state=42,
                    eval_metric='rmse',
                    **self.model_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def load_features(self):
        self.features = self.tfidf_loader.load_all_features()
        self.features = self.tfidf_loader.filter_by_languages(self.features, self.target_languages)
    
    def fit(self, y_train: np.ndarray):
        if self.features is None:
            self.load_features()
        
        X_train = self.features['train']
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, split: str = 'test') -> np.ndarray:
        if self.features is None:
            self.load_features()
        
        X = self.features[split]
        return self.model.predict(X)
    
    def evaluate(self, y_true: np.ndarray, split: str = 'test'):
        y_pred = self.predict(split)
        
        if self.task_type == "classification":
            return {'accuracy': accuracy_score(y_true, y_pred),'f1': f1_score(y_true, y_pred, average='weighted')}
        else:
            return {'mse': mean_squared_error(y_true, y_pred),'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),'r2': r2_score(y_true, y_pred)}


def create_tfidf_baseline_model(model_type: str,
    task_type: str, 
    tfidf_features_dir: str,
    target_languages: List[str] = ['all'],
    model_params: Optional[Dict[str, Any]] = None):
    
    tfidf_loader = TfidfFeatureLoader(tfidf_features_dir)
    
    return TfidfBaselineModel(model_type=model_type,
        task_type=task_type,tfidf_loader=tfidf_loader,
        target_languages=target_languages,
        model_params=model_params)