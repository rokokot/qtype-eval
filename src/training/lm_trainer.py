# trainer class for neural language models

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
from src.evaluation.metrics import calculate_metrics, format_metrics_for_logging
import logging
from collections import defaultdict
import json
import time
import sys

logger = logging.getLogger(__name__)


class LMTrainer:
    def __init__(
        self,
        model: nn.Module,
        task_type: str = "classification",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        patience: int = 3,
        device: Optional[str] = None,
        output_dir: Optional[str] = None,
        wandb_run: Optional[Any] = None,
        gradient_accumulation_steps: int = 1,
        debug_mode: bool = False,
    ):
        self.model = model
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.wandb_run = wandb_run
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.debug_mode = debug_mode

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if task_type == "classification":
            self.criterion = nn.BCEWithLogitsLoss() if model.num_outputs == 1 else nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        self.metrics_history = defaultdict(list)

        if hasattr(model, 'task_type') and model.task_type == "probe":
            self.learning_rate = 1e-4 
            
            self.patience = max(5, self.patience)
            
            self.weight_decay = 0.1 


    def train(self, train_loader, val_loader=None, test_loader=None) -> Dict[str, Any]:
        # Add a cleanup function for GPU memory
        def cleanup_gpu_memory():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared")
        
        try:
            self.model = self.model.to(self.device)
            
            # Debug mode logging
            if self.debug_mode:
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f'Model has {trainable_params:,} trainable parameters out of {total_params:,}')

                if hasattr(self.model, 'model') and hasattr(self.model, 'head'):
                    encoder_trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
                    head_trainable = sum(p.numel() for p in self.model.head.parameters() if p.requires_grad)
                    logger.info(f"Encoder: {encoder_trainable:,} trainable parameters, Head: {head_trainable:,} trainable parameters")
                    
                    # Check if model is correctly frozen/unfrozen
                    encoder_frozen = encoder_trainable == 0
                    logger.info(f"Encoder is {'frozen' if encoder_frozen else 'trainable'}")

            # Probe-specific logging - FIXED: removed the early-stopping check that was in the wrong place
            if hasattr(self.model, 'task_type') and self.model.task_type == "probe":
                # More detailed logging for probes
                logger.info("Running probe experiment with specialized configuration")

            optimizer = optim.AdamW(
                [
                    {"params": [p for p in self.model.parameters() if p.requires_grad], 
                    "lr": self.learning_rate}
                ], 
                weight_decay=self.weight_decay
            )
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.num_epochs, 
                eta_min=self.learning_rate * 0.1
            )

            best_val_loss = float("inf")
            best_model_state = None
            patience_counter = 0
            start_time = time.time()
            degenerate_predictions_count = 0  # Track degenerate predictions

            if self.debug_mode:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                logger.info(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
                
                # Log layer-wise trainable parameters
                for name, module in self.model.named_children():
                    module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    logger.info(f"Module {name}: {module_params:,} trainable parameters")

            # Store prediction tracking for debugging
            if self.debug_mode:
                self.all_epoch_preds = []
                self.all_epoch_labels = []

            for epoch in range(self.num_epochs):
                self.model.train()
                train_loss = 0.0
                batch_losses = []
                epoch_preds = []
                epoch_labels = []
                batch_idx = 0
                total_batches = len(train_loader)
                
                # Initialize progress tracking
                print(f"Epoch {epoch+1}/{self.num_epochs}: [", end="")
                sys.stdout.flush()

                for batch in train_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    
                    # Handle different output shapes based on task type
                    if self.task_type == "classification":
                        if outputs.size(-1) == 1:
                            batch["labels"] = batch["labels"].view(-1, 1).float()
                    else:
                        batch["labels"] = batch["labels"].view(outputs.size()).float()
                    
                    loss = self.criterion(outputs, batch["labels"])
                    # Normalize loss by accumulation steps for proper scaling
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                    
                    # Track predictions for debugging
                    if self.debug_mode:
                        with torch.no_grad():
                            if self.task_type == "classification":
                                preds = (outputs > 0.5).float().cpu().numpy()
                            else:
                                preds = outputs.cpu().numpy()
                            epoch_preds.append(preds)
                            epoch_labels.append(batch["labels"].cpu().numpy())
                    
                    batch_losses.append(loss.item() * self.gradient_accumulation_steps)

                    # Perform optimizer step after accumulating gradients
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                        # Debug gradient information
                        if self.debug_mode and batch_idx < 3:
                            grad_norms = []
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    grad_norms.append((name, param.grad.norm().item()))
                            
                            for name, norm in grad_norms:
                                if "head" in name or batch_idx == 0:  # Only print head gradients after first batch
                                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, {name} grad norm: {norm:.6f}")
                        
                        # Apply accumulated gradients
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    batch_idx += 1
                    train_loss += loss.item() * self.gradient_accumulation_steps
                    
                    # Update progress display in same line
                    progress = int(30 * batch_idx / total_batches)
                    print(f"\rEpoch {epoch+1}/{self.num_epochs}: [{'=' * progress}{' ' * (30-progress)}] {batch_idx}/{total_batches} batches, loss: {train_loss/batch_idx:.4f}", end="")
                    sys.stdout.flush()
                
                # Print newline after epoch completion
                print()
                
                avg_train_loss = train_loss / len(train_loader)
                self.metrics_history['train_loss'].append(avg_train_loss)
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_train_loss:.4f}")

                # Analyze predictions for debugging
                if self.debug_mode and epoch_preds:
                    try:
                        epoch_preds = np.vstack([p for p in epoch_preds if p.size > 0])
                        epoch_labels = np.vstack([l for l in epoch_labels if l.size > 0])
                        self.all_epoch_preds.append(epoch_preds)
                        self.all_epoch_labels.append(epoch_labels)
                        
                        # Check for prediction bias (all same class)
                        unique_preds, counts = np.unique(epoch_preds, return_counts=True)
                        logger.info(f"Epoch {epoch+1} prediction distribution: {list(zip(unique_preds, counts))}")
                        
                        # Sanity check accuracy computation
                        if self.task_type == "classification":
                            accuracy = np.mean((epoch_preds > 0.5).astype(int) == epoch_labels.astype(int))
                            logger.info(f"Epoch {epoch+1} manual accuracy check: {accuracy:.4f}")
                            
                            # Alert on degenerate predictions (predicting all one class)
                            if len(unique_preds) <= 1:
                                logger.warning(f"WARNING: Model is predicting only one class: {unique_preds[0]}")
                                degenerate_predictions_count += 1
                                
                                # NEW: Early termination for degenerate models
                                if degenerate_predictions_count >= 3:  # If 3 consecutive epochs have degenerate predictions
                                    logger.warning("Stopping training due to persistent degenerate predictions")
                                    break
                            else:
                                degenerate_predictions_count = 0  # Reset counter if predictions are not degenerate
                    except Exception as e:
                        logger.error(f"Error analyzing predictions: {e}")

                # Log to wandb if available
                if self.wandb_run:
                    self.wandb_run.log({
                        "epoch": epoch + 1, 
                        "train_loss": avg_train_loss, 
                        "learning_rate": optimizer.param_groups[0]["lr"]
                    })

                # Validation
                if val_loader:
                    val_loss, val_metrics = self._evaluate(val_loader)
                    self.metrics_history['val_loss'].append(val_loss)
                    for k, v in val_metrics.items():
                        self.metrics_history[f'val_{k}'].append(v)
                    
                    logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Val Loss: {val_loss:.4f}, Metrics: {format_metrics_for_logging(val_metrics)}")

                    scheduler.step(val_loss)

                    if self.wandb_run:
                        self.wandb_run.log({
                            "epoch": epoch + 1, 
                            "val_loss": val_loss, 
                            **{f"val_{k}": v for k, v in val_metrics.items()}
                        })

                    # Track best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                        patience_counter = 0

                        if self.wandb_run:
                            self.wandb_run.log({
                                "best_val_loss": best_val_loss, 
                                **{f"best_val_{k}": v for k, v in val_metrics.items()}
                            })
                    else:
                        patience_counter += 1
                        logger.info(f"Validation did not improve. Patience: {patience_counter}/{self.patience}")
                    
                    # NEW: Stricter early stopping for probe models
                    if hasattr(self.model, 'task_type') and self.model.task_type == "probe":
                        if val_loss > best_val_loss * 1.2:  # More strict improvement criterion for probes
                            logger.info(f"Probe performance significantly worse than best. Adding extra patience point.")
                            patience_counter += 1  # Extra penalty for probe models

                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        if self.wandb_run:
                            self.wandb_run.log({"early_stop_epoch": epoch+1})
                        break

                    # Add a more aggressive early stopping for probe models
                    if hasattr(self.model, 'task_type') and self.model.task_type == "probe":
                        if val_loss > best_val_loss * 1.5:  # More strict stopping for probes
                            logger.warning("Probe performance degrading significantly. Stopping early.")
                            break

            train_time = time.time() - start_time
            logger.info(f"Training completed in {train_time:.2f} seconds")

            # Load best model for final evaluation
            if best_model_state:
                logger.info("Loading best model for final evaluation")
                self.model.load_state_dict(best_model_state)
                self.model = self.model.to(self.device)
            else:
                logger.warning("No best model saved, using final model state")

            # Final evaluation
            train_loss, train_metrics = self._evaluate(train_loader)
            val_loss, val_metrics = self._evaluate(val_loader) if val_loader else (None, None)
            test_loss, test_metrics = self._evaluate(test_loader) if test_loader else (None, None)

            logger.info(f"Final evaluation - Train metrics: {format_metrics_for_logging(train_metrics)}")
            if val_metrics:
                logger.info(f"Final evaluation - Validation metrics: {format_metrics_for_logging(val_metrics)}")
            if test_metrics:
                logger.info(f"Final evaluation - Test metrics: {format_metrics_for_logging(test_metrics)}")

            # Collect results
            results = {
                "train_time": train_time,
                "train_metrics": {"loss": train_loss, **train_metrics},
                "val_metrics": {"loss": val_loss, **val_metrics} if val_metrics else None,
                "test_metrics": {"loss": test_loss, **test_metrics} if test_metrics else None,
            }

            # Log results to wandb
            if self.wandb_run:
                # Log final metrics
                self.wandb_run.log({
                    "train_time": train_time,
                    **{f"final_train_{k}": v for k, v in train_metrics.items()},
                    **({f"final_val_{k}": v for k, v in val_metrics.items()} if val_metrics else {}),
                    **({f"final_test_{k}": v for k, v in test_metrics.items()} if test_metrics else {})
                })
                
                # Log summary metrics
                self.wandb_run.summary.update({
                    "train_time": train_time,
                    "train_loss": avg_train_loss,
                    **{f"final_train_{k}": v for k, v in train_metrics.items()},
                    **({f"final_val_{k}": v for k, v in val_metrics.items()} if val_metrics else {}),
                    **({f"final_test_{k}": v for k, v in test_metrics.items()} if test_metrics else {})
                })
        
            # Save results and model
            if self.output_dir:
                with open(os.path.join(self.output_dir, "results.json"), "w") as f:
                    json.dump(results, f, indent=2)
        
                model_path = os.path.join(self.output_dir, "model.pt")
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
        
            return results
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Traceback: {error_traceback}")
            
            # Return partial results with error information
            return {
                "error": str(e),
                "traceback": error_traceback,
                "train_metrics": None,
                "val_metrics": None,
                "test_metrics": None,
            }
        
        finally:
            # Always clean up GPU memory
            cleanup_gpu_memory()

    def _evaluate(self, data_loader):
        """Evaluate model on a data loader"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                
                if self.task_type == "classification":
                    if outputs.size(-1) == 1:  
                        batch["labels"] = batch["labels"].view(-1, 1).float()
                    
                    loss = self.criterion(outputs, batch["labels"])
                else:
                    batch["labels"] = batch["labels"].view(outputs.size()).float()
                    loss = self.criterion(outputs, batch["labels"])
                
                total_loss += loss.item()
                
                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(batch["labels"].detach().cpu().numpy())
    
        avg_loss = total_loss / len(data_loader)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        metrics = self._calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate standardized metrics based on task type."""
        try:
            # Prepare predictions based on task type
            y_true = np.squeeze(y_true)
            y_pred = np.squeeze(y_pred)
            
            if self.task_type == "classification":
                # Convert probabilities to binary predictions
                y_pred = (y_pred > 0.5).astype(int)
            
            # Flatten arrays if needed
            if y_true.ndim > 1:
                y_true = y_true.reshape(-1)
            if y_pred.ndim > 1:
                y_pred = y_pred.reshape(-1)
            
            return calculate_metrics(y_true, y_pred, self.task_type)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return fallback metrics
            if self.task_type == "classification":
                return {"primary_metric": "accuracy", "primary_value": 0.0, "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                return {"primary_metric": "mse", "primary_value": float('inf'), "mse": float('inf'), "r2": 0.0}