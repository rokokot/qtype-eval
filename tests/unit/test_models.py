import pytest
import torch
from src.models.model_factory import create_model, LMProbe


def test_create_dummy_classifier():
    model = create_model("dummy", "classification")
    assert model.__class__.__name__ == "DummyClassifier"
    assert model.strategy == "most_frequent"


def test_create_dummy_regressor():
    model = create_model("dummy", "regression", strategy="mean")
    assert model.__class__.__name__ == "DummyRegressor"
    assert model.strategy == "mean"


def test_logistic_regression():
    model = create_model("logistic", "classification")
    assert model.__class__.__name__ == "LogisticRegression"
    assert model.C == 1.0


def test_create_ridge_regressor():
    model = create_model("ridge", "regression", alpha=2.0)
    assert model.__class__.__name__ == "Ridge"
    assert model.alpha == 2.0


def test_create_xgboost():
    model = create_model("xgboost", "classification")
    assert "XGBClassifier" in model.__class__.__name__
    assert model.n_estimators == 100

    model = create_model("xgboost", "regression", n_estimators=50, max_depth=8)
    assert "XGBRegressor" in model.__class__.__name__
    assert model.n_estimators == 50
    assert model.max_depth == 8


def test_lm_probe_initialization():
    model = LMProbe(model_name="distilbert-base-uncased", task_type="classification")
    assert model.task_type == "classification"

    model = LMProbe(
        model_name="distilbert-base-uncased",
        task_type="regression",
        num_outputs=2,
        dropout=0.2,
        freeze_model=True,
    )
    assert model.task_type == "regression"
    assert model.num_outputs == 2

    params_frozen = all(not param.requires_grad for param in model.model.parameters())
    assert params_frozen


def test_lm_probe_forward():
    model = LMProbe(model_name="distilbert-base-uncased", task_type="classification")

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.shape == (batch_size, 1)
    assert torch.all(outputs >= 0) and torch.all(outputs <= 1)

    model_reg = LMProbe(model_name="distilbert-base-uncased", task_type="regression")
    with torch.no_grad():
        outputs_reg = model_reg(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs_reg.shape == (batch_size, 1)


def test_invalid_model_type():
    with pytest.raises(ValueError):
        create_model("invalid_model", "classification")
