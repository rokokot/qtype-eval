# creating instances of models used in our experiments
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)


class BaseLMModel(nn.Module):
    """Base class for language model based models with common functionality."""
    
    def __init__(
        self,
        model_name: str = "cis-lmu/glot500-base",
        task_type: str = "classification",
        num_outputs: int = 1,
        freeze_model: bool = True,
        layer_wise: bool = True,
        layer_index: int = -1,
    ):
        super().__init__()
        
        try:
            local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            
            self.model = AutoModel.from_pretrained(
                model_name, 
                local_files_only=local_only,
                cache_dir=os.environ.get("HF_HOME", None)
            )
            logger.info(f"Loaded model from {'local cache' if local_only else 'online'}: {model_name}")

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

        # Store configuration
        self.task_type = task_type
        self.num_outputs = num_outputs
        self.layer_wise = layer_wise
        self.layer_index = layer_index
        self.freeze_model = freeze_model
        
        # Set up model freezing
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Language model parameters frozen")
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("Language model parameters trainable")
            
        # Log model configuration
        logger.info(f"Base model configuration: layer-wise={layer_wise}, layer_index={layer_index}, freeze_model={freeze_model}")
            
    def get_representation(self, input_ids, attention_mask, token_type_ids=None):
        """Extract representation from the language model."""
        if self.layer_wise:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if token_type_ids is not None else None,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states
            layer_index = self.layer_index if self.layer_index >= 0 else len(hidden_states) + self.layer_index
            
            if layer_index < 0 or layer_index >= len(hidden_states):
                logger.warning(f"Layer index {self.layer_index} is out of bounds. Using last layer.")
                layer_output = hidden_states[-1]
            else:
                layer_output = hidden_states[layer_index]
            
            # Use CLS token representation
            sentence_repr = layer_output[:, 0, :]
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if token_type_ids is not None else None,
            )
            # Use CLS token representation from last layer
            sentence_repr = outputs.last_hidden_state[:, 0, :]
        
        return sentence_repr
    
    def log_parameter_stats(self):
        """Log model parameter statistics."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model has {trainable_params:,} trainable parameters out of {total_params:,} total parameters")
        
        # More detailed breakdown
        encoder_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        head_trainable = trainable_params - encoder_trainable
        logger.info(f"Encoder: {encoder_trainable:,} trainable parameters, Head: {head_trainable:,} trainable parameters")


class LMProbe(BaseLMModel):
    """Lightweight probe for language model representations."""
    
    def __init__(
        self,
        model_name: str = "cis-lmu/glot500-base",
        task_type: str = "classification",
        num_outputs: int = 1,
        dropout: float = 0.3,
        freeze_model: bool = True,
        layer_wise: bool = True,
        layer_index: int = -1,
        probe_hidden_size: int = 96
    ):
        super().__init__(
            model_name=model_name,
            task_type=task_type,
            num_outputs=num_outputs,
            freeze_model=freeze_model,
            layer_wise=layer_wise,
            layer_index=layer_index,
        )
        
        # Set up probe head
        hidden_size = self.model.config.hidden_size
        
        if probe_hidden_size is None or probe_hidden_size <= 0:
            probe_hidden_size = hidden_size // 8 if task_type == "classification" else hidden_size // 4
            logger.info(f"Using calculated probe_hidden_size: {probe_hidden_size}")
        else:
            logger.info(f"Using provided probe_hidden_size: {probe_hidden_size}")
        
        # Create a simple MLP probe
        self.head = nn.Sequential(
            nn.Linear(hidden_size, probe_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(probe_hidden_size, num_outputs),
            nn.Sigmoid() if num_outputs == 1 and task_type == "classification" else nn.Identity(),
        )
        
        # Log parameter statistics
        self.log_parameter_stats()
        logger.info(f"Probe configuration: hidden_size={probe_hidden_size}, dropout={dropout}")

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        sentence_repr = self.get_representation(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        return self.head(sentence_repr)


class LMFineTuner(BaseLMModel):
    """Full model fine-tuner with more expressive classification/regression head."""
    
    def __init__(
        self,
        model_name: str = "cis-lmu/glot500-base",
        task_type: str = "classification",
        num_outputs: int = 1,
        dropout: float = 0.1,
        head_hidden_size: int = 768,
        head_layers: int = 2,
        layer_wise: bool = False,  # Usually False for fine-tuning
        layer_index: int = -1,
    ):
        super().__init__(
            model_name=model_name,
            task_type=task_type,
            num_outputs=num_outputs,
            freeze_model=False,  # Always False for fine-tuning
            layer_wise=layer_wise,
            layer_index=layer_index,
        )
        
        # Set up a more expressive head for fine-tuning
        hidden_size = self.model.config.hidden_size
        
        logger.info(f"Using head_hidden_size: {head_hidden_size} for fine-tuning")
        
        # Create a multi-layer head based on the number of specified layers
        if head_layers == 1:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, num_outputs),
                nn.Sigmoid() if num_outputs == 1 and task_type == "classification" else nn.Identity(),
            )
        elif head_layers == 2:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden_size),
                nn.GELU(),  # GELU often works better in transformers
                nn.Dropout(dropout),
                nn.Linear(head_hidden_size, num_outputs),
                nn.Sigmoid() if num_outputs == 1 and task_type == "classification" else nn.Identity(),
            )
        else:  # 3+ layers
            layers = [nn.Linear(hidden_size, head_hidden_size), nn.GELU(), nn.Dropout(dropout)]
            
            # Add intermediate layers
            for _ in range(head_layers - 2):
                layers.extend([
                    nn.Linear(head_hidden_size, head_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
                
            # Add output layer
            layers.extend([
                nn.Linear(head_hidden_size, num_outputs),
                nn.Sigmoid() if num_outputs == 1 and task_type == "classification" else nn.Identity()
            ])
            
            self.head = nn.Sequential(*layers)
        
        # Log parameter statistics
        self.log_parameter_stats()
        logger.info(f"Fine-tuning head configuration: hidden_size={head_hidden_size}, layers={head_layers}, dropout={dropout}")

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        sentence_repr = self.get_representation(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        return self.head(sentence_repr)


def create_model(model_type, task_type, **kwargs):
    """Factory function to create the appropriate model type."""
    logger.info(f"Creating {model_type} model for {task_type} task")
    
    task_type = task_type.lower() if isinstance(task_type, str) else "classification"

    if model_type not in ["dummy", "logistic", "ridge", "xgboost", "lm_probe", "lm_finetune"]:
        logger.warning(f"Unknown model type: {model_type}. Using 'dummy' model.")
        model_type = "lm_probe"

    num_outputs = kwargs.get("num_outputs", 1)
    if task_type == "classification":
        num_outputs = 1  # Binary classification

    common_params = {
        "model_name": kwargs.get("lm_name", "cis-lmu/glot500-base"),
        "task_type": task_type,
        "num_outputs": num_outputs,
        "dropout": kwargs.get("dropout", 0.1),
        "layer_wise": kwargs.get("layer_wise", False),
        "layer_index": kwargs.get("layer_index", -1)
    }

    if model_type == "lm_probe":
        # Probe-specific parameters
        return LMProbe(
            **common_params,
            freeze_model=kwargs.get("freeze_model", True),
            probe_hidden_size=kwargs.get("probe_hidden_size", 96)
        )
    
    elif model_type == "lm_finetune":
        # Fine-tuning specific parameters
        return LMFineTuner(
            **common_params,
            freeze_model=kwargs.get("freeze_model", False),  # Default to unfreeze for fine-tuning
            head_hidden_size=kwargs.get("head_hidden_size", 768),
            head_layers=kwargs.get("head_layers", 2)
        )
    
    if model_type == "logistic" and task_type != "classification":
        logger.warning("Logistic regression is for classification only. Switching to ridge regression.")
        model_type = "ridge"
    
    elif model_type == "ridge" and task_type == "classification":
        logger.warning("Ridge regression is for regression only. Switching to logistic regression.")
        model_type = "logistic"
    
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