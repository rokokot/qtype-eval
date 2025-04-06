# creating instances of models used in our experiments

import torch.nn as nn
import os
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)


class LMProbe(nn.Module):  # Neural probe for language model representations
    def __init__(
        self,
        model_name: str = "cis-lmu/glot500-base",
        task_type: str = "classification",
        num_outputs: int = 1,
        dropout: float = 0.1,
        freeze_model: bool = False,
    ):
        super().__init__()

        try:
            local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        
            try:
                if local_only:
                    self.model = AutoModel.from_pretrained(
                        model_name, 
                        local_files_only=True,  # Force using cached version
                        cache_dir=os.environ.get("HF_HOME", None)
                    )
                    logger.info(f"Loaded model from local cache: {model_name}")
                else:
                    self.model = AutoModel.from_pretrained(model_name)
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                raise

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
        

        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Language model parameters frozen")

        hidden_size = self.model.config.hidden_size

        if task_type == "classification":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_outputs),
                nn.Sigmoid() if num_outputs == 1 else nn.Identity(),
            )

            logger.info(f"Created classification head with {num_outputs} outputs")
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_outputs),
            )
            logger.info(f"Created regression head with {num_outputs} outputs")

        self.task_type = task_type
        self.num_outputs = num_outputs

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        model_type = self.model.__class__.__name__

        if "DistilBert" in model_type:
            # DistilBERT doesn't use token_type_ids
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            # BERT and others use token_type_ids
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if token_type_ids is not None else None,
            )

        # Get the sentence representation from [CLS] token
        sentence_repr = outputs.last_hidden_state[:, 0, :]

        outputs = self.head(sentence_repr)

        return outputs


def create_model(model_type, task_type, **kwargs):
    """Create a model for the given task type with improved validation."""
    logger.info(f"Creating {model_type} model for {task_type} task")
    
    # Normalize task type
    task_type = task_type.lower() if isinstance(task_type, str) else "classification"
    
    # Validate model type
    if model_type not in ["dummy", "logistic", "ridge", "xgboost", "lm_probe"]:
        logger.warning(f"Unknown model type: {model_type}. Using 'dummy' model.")
        model_type = "dummy"
    
    # For LM probe, ensure consistent parameters
    if model_type == "lm_probe":
        # Default num_outputs based on task type
        num_outputs = kwargs.get("num_outputs", 1)
        
        # For classification tasks, set num_outputs appropriately
        if task_type == "classification":
            num_outputs = 1  # Binary classification
        
        return LMProbe(
            model_name=kwargs.get("lm_name", "cis-lmu/glot500-base"),
            task_type=task_type,
            num_outputs=num_outputs,
            dropout=kwargs.get("dropout", 0.1),
            freeze_model=kwargs.get("freeze_model", False)
        )
    
    # Handle task type validation for sklearn models
    if model_type == "logistic" and task_type != "classification":
        logger.warning("Logistic regression is for classification only. Switching to ridge regression.")
        model_type = "ridge"
    
    elif model_type == "ridge" and task_type == "classification":
        logger.warning("Ridge regression is for regression only. Switching to logistic regression.")
        model_type = "logistic"
    
    # Create the appropriate model
    if model_type == "dummy":
        if task_type == "classification":
            return DummyClassifier(
                strategy=kwargs.get("strategy", "most_frequent"), 
                random_state=kwargs.get("random_state", 42)
            )
        else:
            return DummyRegressor(strategy=kwargs.get("strategy", "mean"))
    
    elif model_type == "logistic":
        return LogisticRegression(
            C=kwargs.get("C", 1.0),
            max_iter=kwargs.get("max_iter", 1000),
            random_state=kwargs.get("random_state", 42),
            n_jobs=kwargs.get("n_jobs", -1),
            solver=kwargs.get("solver", "liblinear")
        )
    
    elif model_type == "ridge":
        return Ridge(
            alpha=kwargs.get("alpha", 1.0), 
            random_state=kwargs.get("random_state", 42)
        )
    
    elif model_type == "xgboost":
        if task_type == "classification":
            return xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=kwargs.get("random_state", 42),
                use_label_encoder=False,
                eval_metric="logloss",
            )
        else:
            return xgb.XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=kwargs.get("random_state", 42),
            )
