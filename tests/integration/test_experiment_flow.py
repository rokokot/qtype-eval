import pytest
import tempfile
import os
from src.data.datasets import load_sklearn_data
from src.models.model_factory import create_model
from src.training.sklearn_trainer import SklearnTrainer


@pytest.mark.integration
def test_sklearn_experiment_flow():
    task = "question_type"
    languages = ["en"]
    model_type = "dummy"

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(languages=languages, task=task)

    X_train, y_train = X_train[:10], y_train[:10]
    X_val, y_val = X_val[:5], y_val[:5]
    X_test, y_test = X_test[:5], y_test[:5]

    model = create_model(model_type, "classification")

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = SklearnTrainer(model=model, task_type="classification", output_dir=tmpdir)

        results = trainer.train(train_data=(X_train, y_train), val_data=(X_val, y_val), test_data=(X_test, y_test))

        assert os.path.exists(os.path.join(tmpdir, "results.json"))

        assert "train_metrics" in results
        assert "val_metrics" in results
        assert "test_metrics" in results
