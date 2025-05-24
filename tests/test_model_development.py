import pytest
import pickle
import joblib
import json
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

DATA_PATH = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"

@pytest.fixture(scope="module")
def preprocessed():
    return preprocess_data(DATA_PATH)

@pytest.fixture(scope="module")
def train_test_data(preprocessed):
    with open(preprocessed, "rb") as f:
        data = pickle.load(f)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

def test_nondeterminism_robustness(preprocessed, train_test_data):
    accs = []
    for seed in [1, 42, 123]:
        model_path = train_model(preprocessed, random_state=seed)
        metrics_path = evaluate_model(model_path, preprocessed)
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            acc = metrics.get("accuracy")
            assert acc is not None, "Accuracy not found in metrics.json"
            accs.append(acc)

    variability = max(accs) - min(accs)
    assert variability <= 0.05, f"Accuracy variance too high: {accs}"

def test_data_slice_performance(preprocessed, train_test_data):
    _, X_test, _, y_test = train_test_data
    model_path = train_model(preprocessed, random_state=0)
    model = joblib.load(model_path)

    short_idx = [i for i, x in enumerate(X_test) if x.sum() <= 5]
    long_idx = [i for i, x in enumerate(X_test) if x.sum() >= 15]

    if not short_idx or not long_idx:
        pytest.skip("Insufficient short/long samples for slice test")

    short_X = X_test[short_idx]
    short_y = [y_test[i] for i in short_idx]
    long_X = X_test[long_idx]
    long_y = [y_test[i] for i in long_idx]

    short_preds = model.predict(short_X)
    long_preds = model.predict(long_X)

    acc_short = accuracy_score(short_y, short_preds)
    acc_long = accuracy_score(long_y, long_preds)

    diff = abs(acc_short - acc_long)
    assert diff <= 0.25, f"Accuracy gap on slices too large: short={acc_short:.2f}, long={acc_long:.2f}"

def test_baseline_comparison(train_test_data, preprocessed):
    X_train, X_test, y_train, y_test = train_test_data

    dummy = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy.fit(X_train, y_train)
    baseline_preds = dummy.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_preds)

    model_path = train_model(preprocessed, random_state=0)
    model = joblib.load(model_path)
    model_preds = model.predict(X_test)
    model_acc = accuracy_score(y_test, model_preds)

    assert model_acc > baseline_acc, (
        f"Trained model does not outperform baseline: model={model_acc:.2f}, baseline={baseline_acc:.2f}"
    )
