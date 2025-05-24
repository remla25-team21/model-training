import os
import pytest
import pickle
import joblib
import json
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model
from sklearn.metrics import accuracy_score

DATA_PATH = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"

@pytest.fixture(scope="module")
def preprocessed():
    return preprocess_data(DATA_PATH)

@pytest.fixture(scope="module")
def train_test_data(preprocessed):
    with open(preprocessed, "rb") as f:
        data = pickle.load(f)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

def test_integration_pipeline(preprocessed):
    """Run the full training and evaluation pipeline"""
    model_path = train_model(preprocessed, random_state=0)
    assert os.path.exists(model_path), "Trained model file not created"

    metrics_path = evaluate_model(model_path, preprocessed)
    assert os.path.exists(metrics_path), "Metrics file not created"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics, "Accuracy not found in metrics file"
    assert 0.7 <= metrics["accuracy"] <= 1.0, f"Unrealistic accuracy: {metrics['accuracy']}"

def test_model_rollback(train_test_data, preprocessed):
    """Test loading a saved model and re-evaluating"""
    X_train, X_test, y_train, y_test = train_test_data

    model_path = train_model(preprocessed, random_state=0)
    model = joblib.load(model_path)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    assert 0.7 <= acc <= 1.0, f"Reloaded model accuracy out of range: {acc:.2f}"
