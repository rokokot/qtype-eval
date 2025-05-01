import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import AutoModel, AutoConfig
import logging
from typing import Optional, Dict, Any, Union

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


#
# PROBE IMPLEMENTATIONS
#

class LinearProbe(BaseLMModel):
    """Linear probe for linguistic analysis - simple linear mapping from representations to outputs."""
    
    def __init__(
        self,
        model_name: str = "cis-lmu/glot500-base",
        task_type: str = "classification",
        num_outputs: int = 1,
        dropout: float = 0.0,  # Default to no dropout for linear probes
        freeze_model: bool = True,
        layer_wise: bool = True,
        layer_index: int = -1,
        weight_normalization: bool = False,
        probe_rank: int = None  # For rank-constrained probes
    ):
        super().__init__(
            model_name=model_name,
            task_type=task_type,
            num_outputs=num_outputs,
            freeze_model=freeze_model,
            layer_wise=layer_wise,
            layer_index=layer_index,
        )
        
        # Set up linear probe head
        hidden_size = self.model.config.hidden_size
        
        if probe_rank is not None and probe_rank < hidden_size:
            # Create a rank-constrained linear probe using factorization
            # B = UV where U is (hidden_size x rank) and V is (rank x num_outputs)
            self.rank = probe_rank
            self.proj_down = nn.Linear(hidden_size, probe_rank, bias=False)
            self.proj_up = nn.Linear(probe_rank, num_outputs, bias=True)
            logger.info(f"Created rank-constrained linear probe with rank {probe_rank}")
        else:
            # Standard direct linear mapping
            self.linear = nn.Linear(hidden_size, num_outputs)
            if weight_normalization:
                # Apply weight normalization for more stable training
                self.linear = nn.utils.weight_norm(self.linear)
                logger.info("Applied weight normalization to linear probe")
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.use_rank_constraint = probe_rank is not None and probe_rank < hidden_size
        self.activation = nn.Sigmoid() if task_type == "classification" and num_outputs == 1 else nn.Identity()
        
        # Log parameter statistics
        self.log_parameter_stats()
        logger.info(f"Linear probe configuration: rank_constraint={self.use_rank_constraint}, dropout={dropout}")

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        sentence_repr = self.get_representation(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        
        if self.dropout is not None:
            sentence_repr = self.dropout(sentence_repr)
        
        if self.use_rank_constraint:
            output = self.proj_up(self.proj_down(sentence_repr))
        else:
            output = self.linear(sentence_repr)
        
        return self.activation(output)


class MLPProbe(BaseLMModel):
    """MLP probe for linguistic analysis with configurable complexity and regularization options."""
    
    def __init__(
        self,
        model_name: str = "cis-lmu/glot500-base",
        task_type: str = "classification",
        num_outputs: int = 1,
        dropout: float = 0.3,
        freeze_model: bool = True,
        layer_wise: bool = True,
        layer_index: int = -1,
        probe_hidden_size: int = 96,
        activation: str = "relu",
        normalization: str = "none",
        probe_depth: int = 1,  # Number of hidden layers
        use_bias: bool = True,
        weight_init: str = "normal"  # Initialization strategy
    ):
        super().__init__(
            model_name=model_name,
            task_type=task_type,
            num_outputs=num_outputs,
            freeze_model=freeze_model,
            layer_wise=layer_wise,
            layer_index=layer_index,
        )
        
        # Set up MLP probe head
        hidden_size = self.model.config.hidden_size
        
        # Calculate probe hidden size if not provided
        if probe_hidden_size is None:
            # For probes, we intentionally use a small fraction of the embedding dimension
            # following principles from probing literature to limit probe expressivity
            probe_hidden_size = hidden_size // 8 if task_type == "classification" else hidden_size // 6
            logger.info(f"Using calculated probe_hidden_size: {probe_hidden_size}")
        else:
            logger.info(f"Using provided probe_hidden_size: {probe_hidden_size}")
        
        # Select activation function
        if activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "gelu":
            act_fn = nn.GELU()
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh()
        elif activation.lower() == "sigmoid":
            act_fn = nn.Sigmoid()
        elif activation.lower() == "silu" or activation.lower() == "swish":
            act_fn = nn.SiLU()
        else:
            logger.warning(f"Unknown activation: {activation}, defaulting to ReLU")
            act_fn = nn.ReLU()
        
        # Build MLP layers
        layers = []
        
        # Input normalization (optional)
        if normalization.lower() == "layer":
            layers.append(nn.LayerNorm(hidden_size))
        elif normalization.lower() == "batch":
            layers.append(nn.BatchNorm1d(hidden_size))
        
        # Input dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # First hidden layer
        layers.append(nn.Linear(hidden_size, probe_hidden_size, bias=use_bias))
        layers.append(act_fn)
        
        # Additional hidden layers if requested
        for _ in range(probe_depth - 1):
            if normalization.lower() == "layer":
                layers.append(nn.LayerNorm(probe_hidden_size))
            elif normalization.lower() == "batch":
                layers.append(nn.BatchNorm1d(probe_hidden_size))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(probe_hidden_size, probe_hidden_size, bias=use_bias))
            layers.append(act_fn)
        
        # Output layer
        layers.append(nn.Linear(probe_hidden_size, num_outputs, bias=use_bias))
        
        # Output activation
        if task_type == "classification" and num_outputs == 1:
            layers.append(nn.Sigmoid())
        
        self.head = nn.Sequential(*layers)
        
        # Apply weight initialization
        self._initialize_weights(weight_init)
        
        # Log parameter statistics
        self.log_parameter_stats()
        logger.info(f"MLP probe configuration: hidden_size={probe_hidden_size}, depth={probe_depth}, activation={activation}, normalization={normalization}")

    def _initialize_weights(self, weight_init):
        """Apply weight initialization strategy to probe parameters."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                if weight_init == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                elif weight_init == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif weight_init == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=1.0)
                elif weight_init == "small":
                    # Small initialization for probes to stay close to linear
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        sentence_repr = self.get_representation(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        return self.head(sentence_repr)


class ClassificationProbe(MLPProbe):
    """Specialized probe for classification tasks with discrete categories."""
    
    def __init__(
        self,
        model_name: str = "cis-lmu/glot500-base",
        num_outputs: int = 1,
        probe_hidden_size: int = 128,  # Slightly larger for discrete categories
        dropout: float = 0.3,  # Higher dropout for classification
        freeze_model: bool = True,
        layer_wise: bool = True,
        layer_index: int = -1,
        activation: str = "gelu",  # GELU works well for classification
        normalization: str = "layer",  # Layer norm is more stable
        probe_depth: int = 1,
        weight_init: str = "xavier",  # Xavier works well for classification
        use_class_weights: bool = False
    ):
        super().__init__(
            model_name=model_name,
            task_type="classification",
            num_outputs=num_outputs,
            probe_hidden_size=probe_hidden_size,
            dropout=dropout,
            freeze_model=freeze_model,
            layer_wise=layer_wise,
            layer_index=layer_index,
            activation=activation,
            normalization=normalization,
            probe_depth=probe_depth,
            weight_init=weight_init
        )
        
        self.use_class_weights = use_class_weights
        if use_class_weights:
            # Initialize with uniform class weights, can be updated later
            self.class_weights = nn.Parameter(torch.ones(num_outputs), requires_grad=False)
        
        logger.info(f"Created specialized classification probe with {probe_depth} layers, {probe_hidden_size} hidden size")
    
    def set_class_weights(self, class_weights):
        """Update class weights based on data distribution."""
        if self.use_class_weights:
            self.class_weights.data = torch.tensor(class_weights)
            logger.info(f"Updated class weights: {self.class_weights}")


class RegressionProbe(MLPProbe):
    """Specialized probe for regression tasks with continuous outputs."""
    
    def __init__(
        self,
        model_name: str = "cis-lmu/glot500-base",
        num_outputs: int = 1,
        probe_hidden_size: int = 96,  # Smaller for continuous properties
        dropout: float = 0.1,  # Lower dropout for regression
        freeze_model: bool = True,
        layer_wise: bool = True,
        layer_index: int = -1,
        activation: str = "silu",  # SiLU/Swish works well for regression
        normalization: str = "layer",
        probe_depth: int = 1,
        weight_init: str = "small",  # Small init for regression stability
        output_standardization: bool = False  # Whether to standardize outputs
    ):
        super().__init__(
            model_name=model_name,
            task_type="regression",
            num_outputs=num_outputs,
            probe_hidden_size=probe_hidden_size,
            dropout=dropout,
            freeze_model=freeze_model,
            layer_wise=layer_wise,
            layer_index=layer_index,
            activation=activation,
            normalization=normalization,
            probe_depth=probe_depth,
            weight_init=weight_init
        )
        
        # Output standardization for better training stability
        self.output_standardization = output_standardization
        if output_standardization:
            self.register_buffer("output_mean", torch.zeros(num_outputs))
            self.register_buffer("output_std", torch.ones(num_outputs))
        
        logger.info(f"Created specialized regression probe with {probe_depth} layers, {probe_hidden_size} hidden size")
    
    def set_output_stats(self, mean, std):
        """Set output standardization statistics based on training data."""
        if self.output_standardization:
            self.output_mean = torch.tensor(mean)
            self.output_std = torch.tensor(std)
            logger.info(f"Set output standardization: mean={mean}, std={std}")
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        """Forward pass with optional output standardization."""
        outputs = super().forward(input_ids, attention_mask, token_type_ids, **kwargs)
        
        if self.output_standardization and self.training:
            # During training, standardize the outputs
            outputs = (outputs - self.output_mean) / self.output_std
        
        return outputs
    
    def predict(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        """Prediction with output destandardization."""
        outputs = self.forward(input_ids, attention_mask, token_type_ids, **kwargs)
        
        if self.output_standardization and not self.training:
            # During inference, destandardize the outputs
            outputs = outputs * self.output_std + self.output_mean
        
        return outputs


#
# FINE-TUNING IMPLEMENTATIONS
#

class ClassificationHead(nn.Module):
    """Optimized classification head for fine-tuning language models."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 768,
        num_layers: int = 2,
        dropout: float = 0.2, 
        num_outputs: int = 1
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(input_size)
        
        if num_layers == 1:
            self.layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_size, num_outputs),
                nn.Sigmoid() if num_outputs == 1 else nn.Identity()
            )
        else:
            layers = [
                nn.Dropout(dropout),
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout)
            ]
            
            # Add intermediate layers if requested
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout)
                ])
            
            # Add output layer
            layers.extend([
                nn.Linear(hidden_size, num_outputs),
                nn.Sigmoid() if num_outputs == 1 else nn.Identity()
            ])
            
            self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.norm(x)
        return self.layers(x)


class RegressionHead(nn.Module):
    """Optimized regression head for fine-tuning language models."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_outputs: int = 1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(input_size)
        self.use_residual = use_residual and num_layers > 1
        
        if num_layers == 1:
            self.layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_size, num_outputs)
            )
            self.skip = None
        else:
            # First hidden layer
            self.first_layer = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_size, hidden_size),
                nn.SiLU(),  # SiLU (Swish) often works better for regression
                nn.LayerNorm(hidden_size)
            )
            
            # Intermediate layers with residual connections
            middle_layers = []
            for _ in range(num_layers - 2):
                middle_layers.extend([
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_size)
                ])
            
            self.middle_layers = nn.Sequential(*middle_layers) if middle_layers else None
            
            # Final layer
            self.final_layer = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_outputs)
            )
            
            # Skip connection for residual
            self.skip = nn.Linear(input_size, hidden_size) if self.use_residual else None
    
    def forward(self, x):
        x = self.norm(x)
        
        if hasattr(self, 'layers'):
            # Simple case - one layer
            return self.layers(x)
        
        # Multi-layer case with potential residual connections
        first_output = self.first_layer(x)
        
        if self.skip is not None:
            residual = self.skip(x)
            first_output = first_output + residual
        
        if self.middle_layers:
            middle_output = first_output
            for i in range(0, len(self.middle_layers), 4):
                layer_output = self.middle_layers[i:i+4](middle_output)
                middle_output = layer_output + middle_output  # Residual within middle layers
            output = middle_output
        else:
            output = first_output
        
        return self.final_layer(output)


class LMFineTuner(BaseLMModel):
    """Enhanced fine-tuner for language models with task-specific heads."""
    
    def __init__(
        self,
        model_name: str = "cis-lmu/glot500-base",
        task_type: str = "classification",
        num_outputs: int = 1,
        dropout: float = 0.1,
        head_hidden_size: int = 768,
        head_layers: int = 2,
        layer_wise: bool = False,  
        layer_index: int = -1,
        freeze_model: bool = False,
        use_pooled_output: bool = False, 
        use_mean_pooling: bool = False,
        gradual_unfreezing: bool = False
    ):
        super().__init__(
            model_name=model_name,
            task_type=task_type,
            num_outputs=num_outputs,
            freeze_model=freeze_model,
            layer_wise=layer_wise,
            layer_index=layer_index,
        )
        
        # Set up specialized head based on task type
        hidden_size = self.model.config.hidden_size
        
        if head_hidden_size is None:
            head_hidden_size = hidden_size
        
        # Store parameters for potential gradual unfreezing
        self.gradual_unfreezing = gradual_unfreezing
        self.use_pooled_output = use_pooled_output
        self.use_mean_pooling = use_mean_pooling
        
        # Create task-specific head
        if task_type == "classification":
            self.head = ClassificationHead(
                input_size=hidden_size,
                hidden_size=head_hidden_size,
                num_layers=head_layers,
                dropout=dropout,
                num_outputs=num_outputs
            )
            logger.info(f"Created classification head with {head_layers} layers, {head_hidden_size} hidden size")
        else:
            # Regression - use specialized regression head
            self.head = RegressionHead(
                input_size=hidden_size,
                hidden_size=head_hidden_size,
                num_layers=head_layers,
                dropout=dropout,
                num_outputs=num_outputs,
                use_residual=True
            )
            logger.info(f"Created regression head with {head_layers} layers, {head_hidden_size} hidden size, with residual connections")
        
        # Log parameter statistics
        self.log_parameter_stats()
        logger.info(f"Fine-tuning head configuration: task={task_type}, hidden_size={head_hidden_size}, layers={head_layers}, dropout={dropout}")
    
    def unfreeze_last_n_layers(self, n):
        """Unfreeze the last n transformer layers for fine-tuning."""
        # First make sure everything is frozen
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get list of transformer layers
        transformer_layers = list(self.model.encoder.layer) if hasattr(self.model, 'encoder') else []
        
        # If model doesn't have this architecture, try another common structure
        if not transformer_layers and hasattr(self.model, 'layers'):
            transformer_layers = list(self.model.layers)
        
        # If model doesn't have either structure, log warning
        if not transformer_layers:
            logger.warning("Could not identify transformer layers for gradual unfreezing")
            return
        
        # Unfreeze last n layers
        logger.info(f"Unfreezing last {n} layers of {len(transformer_layers)} total layers")
        layers_to_unfreeze = transformer_layers[-n:]
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
    
    def get_representation(self, input_ids, attention_mask, token_type_ids=None):
        """Enhanced representation extraction with pooling options."""
        if self.use_pooled_output and hasattr(self.model, 'pooler'):
            # Some models like BERT have a pooler that gives better results for fine-tuning
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if token_type_ids is not None else None,
                output_hidden_states=self.layer_wise
            )
            
            if hasattr(outputs, 'pooler_output'):
                return outputs.pooler_output
        
        # Default to base implementation if pooler not available or not requested
        hidden_states = super().get_representation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Apply mean pooling if requested (over all tokens, not just CLS)
        if self.use_mean_pooling:
            # Get the last hidden state
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if token_type_ids is not None else None,
            )
            
            last_hidden_state = outputs.last_hidden_state
            
            # Apply attention mask for proper mean calculation
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.sum(input_mask_expanded, 1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            # Mean pooling
            return sum_hidden / sum_mask
        
        return hidden_states

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        sentence_repr = self.get_representation(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        return self.head(sentence_repr)


#
# UNIFIED MODEL FACTORY
#

def create_model(model_type: str, task_type: str, **kwargs) -> nn.Module:
    """
    Unified factory function to create the appropriate model type.
    
    Args:
        model_type (str): The type of model to create - 'lm_probe', 'linear_probe', 'lm_finetune'
        task_type (str): The task type - 'classification' or 'regression'
        **kwargs: Additional model-specific parameters
    
    Returns:
        nn.Module: The created model
    """
    # Normalize inputs
    model_type = str(model_type).lower()
    task_type = str(task_type).lower() if isinstance(task_type, str) else "classification"
    lm_name = kwargs.get("lm_name", "cis-lmu/glot500-base")
    
    logger.info(f"Creating {model_type} model for {task_type} task")
    
    # Common model parameters
    common_params = {
        "model_name": lm_name,
        "task_type": task_type,
        "num_outputs": kwargs.get("num_outputs", 1),
        "layer_wise": kwargs.get("layer_wise", True),
        "layer_index": kwargs.get("layer_index", -1)
    }
    
    # Check if linear probe is requested either directly or via probe_hidden_size=0
    probe_hidden_size = kwargs.get("probe_hidden_size", None)
    use_linear_probe = kwargs.get("use_linear_probe", False) or probe_hidden_size == 0
    
    # PROBING MODELS
    if model_type == "linear_probe" or (model_type == "lm_probe" and use_linear_probe):
        return LinearProbe(
            **common_params,
            freeze_model=kwargs.get("freeze_model", True),  # Always freeze for probes
            dropout=kwargs.get("dropout", 0.0),  # Minimal dropout for linear probes
            weight_normalization=kwargs.get("weight_normalization", False),
            probe_rank=kwargs.get("probe_rank", None)
        )
    
    if model_type == "lm_probe":
        # Choose task-specific probe architecture
        if task_type == "classification":
            return ClassificationProbe(
                **common_params,
                freeze_model=kwargs.get("freeze_model", True),  # Always freeze for probes
                probe_hidden_size=probe_hidden_size if probe_hidden_size is not None else 128,
                dropout=kwargs.get("dropout", 0.3),
                activation=kwargs.get("activation", "gelu"),
                normalization=kwargs.get("normalization", "layer"),
                probe_depth=kwargs.get("probe_depth", 1),
                weight_init=kwargs.get("weight_init", "xavier"),
                use_class_weights=kwargs.get("use_class_weights", False)
            )
        else:  # regression
            return RegressionProbe(
                **common_params,
                freeze_model=kwargs.get("freeze_model", True),  # Always freeze for probes
                probe_hidden_size=probe_hidden_size if probe_hidden_size is not None else 96,
                dropout=kwargs.get("dropout", 0.1),
                activation=kwargs.get("activation", "silu"),
                normalization=kwargs.get("normalization", "layer"),
                probe_depth=kwargs.get("probe_depth", 1),
                weight_init=kwargs.get("weight_init", "small"),
                output_standardization=kwargs.get("output_standardization", False)
            )
    
    # FINE-TUNING MODELS
    if model_type == "lm_finetune":
        # Apply task-specific fine-tuning hyperparameters
        if task_type == "classification":
            # Classification fine-tuning optimized params
            return LMFineTuner(
                **common_params,
                freeze_model=kwargs.get("freeze_model", False),  # Default to unfrozen for fine-tuning
                head_hidden_size=kwargs.get("head_hidden_size", 768),
                head_layers=kwargs.get("head_layers", 2),
                dropout=kwargs.get("dropout", 0.2),  # Higher dropout for classification
                use_pooled_output=kwargs.get("use_pooled_output", True),
                use_mean_pooling=False,
                gradual_unfreezing=kwargs.get("gradual_unfreezing", False)
            )
        else:  # regression
            # Regression fine-tuning optimized params
            return LMFineTuner(
                **common_params,
                freeze_model=kwargs.get("freeze_model", False),  # Default to unfrozen for fine-tuning
                head_hidden_size=kwargs.get("head_hidden_size", 512),  # Smaller for regression
                head_layers=kwargs.get("head_layers", 3),  # More layers for regression
                dropout=kwargs.get("dropout", 0.1),  # Lower dropout for regression
                use_pooled_output=False,
                use_mean_pooling=kwargs.get("use_mean_pooling", True),  # Mean pooling often helps regression
                gradual_unfreezing=kwargs.get("gradual_unfreezing", False)
            )
    
    # Specific error for invalid model type
    supported_types = ["linear_probe", "lm_probe", "lm_finetune"]
    raise ValueError(f"Unknown model type: '{model_type}'. Supported types are: {supported_types}")


# Helper function for getting task-specific configuration
def get_model_config(task: str, model_approach: str) -> Dict[str, Any]:
    """
    Get task-specific model configuration.
    
    Args:
        task (str): 'question_type', 'complexity', or a submetric name
        model_approach (str): 'probe' or 'finetune'
    
    Returns:
        Dict[str, Any]: Configuration dictionary with optimized parameters
    """
    # Normalize inputs
    task = task.lower() if isinstance(task, str) else "question_type"
    model_approach = model_approach.lower() if isinstance(model_approach, str) else "probe"
    
    # Determine task type
    if task == "question_type":
        task_type = "classification"
    else:  # complexity or submetrics
        task_type = "regression"
    
    # Create configuration dictionary
    config = {
        "task_type": task_type,
    }
    
    # Set approach and task specific configurations
    if model_approach == "probe":
        # Base probe configuration
        config["model_type"] = "lm_probe"
        config["freeze_model"] = True
        config["layer_wise"] = True
        
        if task_type == "classification":
            # Question type classification probe
            config.update({
                "probe_hidden_size": 128,
                "dropout": 0.3,
                "activation": "gelu",
                "normalization": "layer",
                "probe_depth": 1,
                "weight_init": "xavier",
            })
        else:
            # Complexity/submetric regression probe
            config.update({
                "probe_hidden_size": 96,
                "dropout": 0.1,
                "activation": "silu",
                "normalization": "layer", 
                "probe_depth": 1,
                "weight_init": "small",
                "output_standardization": True,
            })
    
    else:  # "finetune"
        # Base fine-tuning configuration
        config["model_type"] = "lm_finetune"
        config["freeze_model"] = False
        config["layer_wise"] = False
        
        if task_type == "classification":
            # Question type classification fine-tuning
            config.update({
                "head_hidden_size": 768,
                "head_layers": 2,
                "dropout": 0.2,
                "use_pooled_output": True,
                "use_mean_pooling": False,
            })
        else:
            # Complexity/submetric regression fine-tuning
            config.update({
                "head_hidden_size": 512,
                "head_layers": 3,
                "dropout": 0.1,
                "use_pooled_output": False,
                "use_mean_pooling": True,
            })
    
    return config


def get_training_config(task: str, model_approach: str) -> Dict[str, Any]:
    """
    Get task-specific training configuration.
    
    Args:
        task (str): 'question_type', 'complexity', or a submetric name
        model_approach (str): 'probe' or 'finetune'
    
    Returns:
        Dict[str, Any]: Configuration dictionary with optimized training parameters
    """
    # Normalize inputs
    task = task.lower() if isinstance(task, str) else "question_type"
    model_approach = model_approach.lower() if isinstance(model_approach, str) else "probe"
    
    # Determine task type
    if task == "question_type":
        task_type = "classification"
    else:  # complexity or submetrics
        task_type = "regression"
    
    # Create base configuration dictionary
    config = {
        "task_type": task_type,
        "batch_size": 16,
        "weight_decay": 0.01,
        "num_epochs": 15 if model_approach == "probe" else 10,
    }
    
    # Set approach and task specific configurations
    if model_approach == "probe":
        if task_type == "classification":
            # Question type classification probe training
            config.update({
                "lr": 1e-4,
                "patience": 3,
                "scheduler_factor": 0.5,
                "scheduler_patience": 2,
                "gradient_accumulation_steps": 2,
            })
        else:
            # Complexity/submetric regression probe training
            config.update({
                "lr": 5e-5,
                "patience": 4, 
                "scheduler_factor": 0.5,
                "scheduler_patience": 2,
                "gradient_accumulation_steps": 2,
            })
    
    else:  # "finetune"
        if task_type == "classification":
            # Question type classification fine-tuning
            config.update({
                "lr": 2e-5,
                "patience": 3,
                "scheduler_factor": 0.5,
                "scheduler_patience": 2,
                "gradient_accumulation_steps": 4,  # Higher for fine-tuning
            })
        else:
            # Complexity/submetric regression fine-tuning
            config.update({
                "lr": 1e-5,
                "patience": 4,
                "scheduler_factor": 0.5,
                "scheduler_patience": 3,
                "gradient_accumulation_steps": 4,  # Higher for fine-tuning
            })
    
    return config