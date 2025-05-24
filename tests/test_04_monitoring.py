import pytest
import pickle
from preprocess import preprocess_data
from train import train_model
import numpy as np
from scipy.stats import ks_2samp

DATA_PATH = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"

@pytest.fixture(scope="module")
def preprocessed():
    return preprocess_data(DATA_PATH)

@pytest.fixture(scope="module")
def train_test_data(preprocessed):
    with open(preprocessed, "rb") as f:
        data = pickle.load(f)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

def test_feature_distribution_drift(train_test_data):
    """Compare feature distributions in train and test via Kolmogorov–Smirnov test"""
    X_train, X_test, _, _ = train_test_data

    drift_scores = []
    for i in range(X_train.shape[1]):
        train_feat = X_train[:, i].ravel()
        test_feat = X_test[:, i].ravel()
        stat, pval = ks_2samp(train_feat, test_feat)
        drift_scores.append(pval)

    # If many p-values are very low, feature drift exists
    drift_detected = np.sum(np.array(drift_scores) < 0.01)
    ratio = drift_detected / len(drift_scores)
    assert ratio < 0.1, f"Feature drift detected in {ratio:.2%} of features"

def test_prediction_distribution_stability(train_test_data, preprocessed):
    """Check for dramatic changes in predicted label distribution"""
    _, X_test, _, _ = train_test_data
    model_path = train_model(preprocessed, random_state=0)

    import joblib
    model = joblib.load(model_path)
    preds = model.predict(X_test)

    # Count proportion of each predicted label
    unique, counts = np.unique(preds, return_counts=True)
    ratios = dict(zip(unique, counts / len(preds)))

    for label, ratio in ratios.items():
        assert 0.1 <= ratio <= 0.9, f"Prediction ratio for class {label} is unrealistic: {ratio:.2f}"
