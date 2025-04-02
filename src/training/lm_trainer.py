# trainer class for neural language models

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import logging
import json
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LMTrainer:
    def __init__(
        self,
        model: nn.Module,
        task_type: str = "classification",
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        patience: int = 3,
        device: Optional[str] = None,
        output_dir: Optional[str] = None,
        wandb_run: Optional[Any] = None,
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

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.criterion = nn.BCELoss() if task_type == "classification" else nn.MSELoss()

    def train(self, train_loader, val_loader=None, test_loader=None) -> Dict[str, Any]:
        self.model = self.model.to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])  # fwd pass

                loss = self.criterion(outputs, batch["labels"])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # bwd pass

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_train_loss:.4f}")

            if self.wandb_run:
                self.wandb_run.log(
                    {"epoch": epoch + 1, "train_loss": avg_train_loss, "learning_rate": optimizer.param_groups[0]["lr"]}
                )

            if val_loader:
                val_loss, val_metrics = self._evaluate(val_loader)
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

                scheduler.step(val_loss)

                if self.wandb_run:
                    self.wandb_run.log(
                        {"epoch": epoch + 1, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    patience_counter = 0

                    if self.wandb_run:
                        self.wandb_run.log(
                            {"best_val_loss": best_val_loss, **{f"best_val_{k}": v for k, v in val_metrics.items()}}
                        )

                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        train_time = time.time() - start_time

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.model = self.model.to(self.device)

        train_loss, train_metrics = self._evaluate(train_loader)
        val_loss, val_metrics = self._evaluate(val_loader) if val_loader else (None, None)
        test_loss, test_metrics = self._evaluate(test_loader) if test_loader else (None, None)

        results = {
            "train_time": train_time,
            "train_metrics": {"loss": train_loss, **train_metrics},
            "val_metrics": {"loss": val_loss, **val_metrics} if val_metrics else None,
            "test_metrics": {"loss": test_loss, **test_metrics} if test_metrics else None,
        }

        if self.wandb_run:
            self.wandb_run.log(
                {
                    "train_time": train_time,
                    **{f"final_train_{k}": v for k, v in train_metrics.items()},
                    **({f"final_val_{k}": v for k, v in val_metrics.items()} if val_metrics else {}),
                    **({f"final_test_{k}": v for k, v in test_metrics.items()} if test_metrics else {}),
                }
            )

        if self.output_dir:
            with open(os.path.join(self.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

            torch.save(self.model.state_dict(), os.path.join(self.output_dir, "model.pt"))

        return results

    def _evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

                loss = self.criterion(outputs, batch["labels"])
                total_loss += loss.item()

                all_preds.append(outputs.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        metrics = self._calculate_metrics(all_labels, all_preds)

        return avg_loss, metrics

    def _calculate_metrics(self, y_true, y_pred):
        if self.task_type == "classification":
            y_pred_binary = (y_pred > 0.5).astype(int)
            return {
                "accuracy": float(accuracy_score(y_true, y_pred_binary)),
                "f1": float(f1_score(y_true, y_pred_binary, average="binary")),
            }
        else:  # regression
            return {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
            }
