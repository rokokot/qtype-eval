import pytest
import numpy as np
import torch
import os
import tempfile
from src.training.sklearn_trainer import SklearnTrainer
from src.training.lm_trainer import LMTrainer
from src.models.model_factory import create_model


@pytest.mark.unit
def test_sklearn_trainer():
    X_train = np.random.rand(10, 5)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    X_val = np.random.rand(5, 5)
    y_val = np.array([0, 1, 0, 1, 0])
    
    model = create_model("dummy", "classification")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = SklearnTrainer(model=model, task_type="classification", output_dir=tmpdir)
        
        results = trainer.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val)
        )
        
        assert "train_metrics" in results
        assert "val_metrics" in results
        assert "accuracy" in results["train_metrics"]
        assert "f1" in results["train_metrics"]
        
        assert os.path.exists(os.path.join(tmpdir, "results.json"))
        assert os.path.exists(os.path.join(tmpdir, "model.joblib"))


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10, task_type="classification"):
        self.num_samples = num_samples
        self.task_type = task_type
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, 1000, (20,))
        attention_mask = torch.ones(20)
        
        if self.task_type == "classification":
            labels = torch.tensor(idx % 2, dtype=torch.long)
        else:
            labels = torch.tensor([float(idx) / self.num_samples], dtype=torch.float)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


@pytest.mark.unit
def test_lm_trainer():
    """Test the LM trainer."""
    train_dataset = DummyDataset(20, "classification")
    val_dataset = DummyDataset(10, "classification")
    test_dataset = DummyDataset(10, "classification")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
    
    class DummyLMProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(20, 1)
            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, input_ids, attention_mask, **kwargs):
            features = input_ids.sum(dim=1, keepdim=True).float()
            return self.sigmoid(self.linear(features.expand(-1, 20)))
    
    model = DummyLMProbe()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = LMTrainer(
            model=model,
            task_type="classification",
            num_epochs=2,
            patience=1,
            output_dir=tmpdir
        )
        
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        assert "train_metrics" in results
        assert "val_metrics" in results
        assert "test_metrics" in results
        
        assert os.path.exists(os.path.join(tmpdir, "results.json"))
        assert os.path.exists(os.path.join(tmpdir, "model.pt"))