import os
import pickle
import json
import shutil
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_model


# A dedicated dir for test inputs
INPUT_DIR = "test_eval_artifacts"
# The dir created by evaluate_model function
ARTIFACTS_TEST_DIR = "artifacts_test"


# Create dummy model and preprocessed data which will be an input to the evaluate_model function
@pytest.fixture(scope="function")
def setup_and_cleanup_test_environment():
    os.makedirs(INPUT_DIR, exist_ok=True)

    dummy_trained_model_path = os.path.join(INPUT_DIR, "dummy_trained_model.pkl")
    dummy_preprocessed_data_path = os.path.join(
        INPUT_DIR, "dummy_preprocessed_data.pkl"
    )

    # craeating a simple dummy model
    model = LogisticRegression()
    X_train_dummy = np.array([[1, 1], [2, 2], [1, 0], [0, 1]])
    y_train_dummy = np.array([1, 1, 0, 0])
    model.fit(X_train_dummy, y_train_dummy)
    with open(dummy_trained_model_path, "wb") as f:
        pickle.dump(model, f)

    # Create dummy preprocessed data
    X_test_dummy = np.array([[1, 1], [2, 1], [0, 0], [1, 0]])
    y_test_dummy = np.array([1, 1, 0, 0])
    preprocessed_data = {"X_test": X_test_dummy, "y_test": y_test_dummy}
    with open(dummy_preprocessed_data_path, "wb") as f:
        pickle.dump(preprocessed_data, f)

    yield dummy_trained_model_path, dummy_preprocessed_data_path

    # Cleanup the test artifacts directory after the test
    if os.path.exists(INPUT_DIR):
        shutil.rmtree(INPUT_DIR)
    # Cleanup the artifacts directory created by evaluate_model function
    if os.path.exists(ARTIFACTS_TEST_DIR):
        shutil.rmtree(ARTIFACTS_TEST_DIR)


# Successful test case for evaluate_model function
def test_evaluate_model_success(setup_and_cleanup_test_environment):
    dummy_model_path, dummy_data_path = setup_and_cleanup_test_environment

    # Check matrix path
    expected_metrics_path = os.path.join(ARTIFACTS_TEST_DIR, "metrics.json")

    returned_metrics_path = evaluate_model(
        trained_model_path=dummy_model_path,
        preprocessed_data_path=dummy_data_path,
        output_dir=ARTIFACTS_TEST_DIR,
    )

    assert os.path.normpath(returned_metrics_path) == os.path.normpath(
        expected_metrics_path
    ), "Returned metrics path is incorrect."

    # check that the model was copied properly
    expected_final_model_path = os.path.join(ARTIFACTS_TEST_DIR, "sentiment_model.pkl")
    assert os.path.exists(
        expected_final_model_path
    ), "Final model (copied) was not created."
    assert os.path.getsize(dummy_model_path) == os.path.getsize(
        expected_final_model_path
    ), "Copied model size does not match original."

    # investigate the content of metrics.json
    with open(expected_metrics_path, "r") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1-score" in metrics
    assert "classification_report" in metrics
    assert isinstance(metrics["accuracy"], float)
    # In the dumm data that was created, the training data is the same as the test data so accuracy should be 1.0
    assert metrics["accuracy"] == 1.0


# invalid trained model path test
def test_evaluate_model_trained_model_not_found(setup_and_cleanup_test_environment):
    _, dummy_data_path = setup_and_cleanup_test_environment

    with pytest.raises(FileNotFoundError):
        evaluate_model(
            trained_model_path="non_existent_model.pkl",
            preprocessed_data_path=dummy_data_path,
            output_dir=ARTIFACTS_TEST_DIR,
        )


# Testing invalid preprocessed data path
def test_evaluate_model_preprocessed_data_not_found(setup_and_cleanup_test_environment):
    dummy_model_path, _ = setup_and_cleanup_test_environment

    with pytest.raises(FileNotFoundError):
        evaluate_model(
            trained_model_path=dummy_model_path,
            preprocessed_data_path="non_existent_data.pkl",
            output_dir=ARTIFACTS_TEST_DIR,
        )


# Testing invalid preprocessed data format
def test_evaluate_model_data_key_error(setup_and_cleanup_test_environment):
    dummy_model_path, _ = setup_and_cleanup_test_environment

    os.makedirs(ARTIFACTS_TEST_DIR, exist_ok=True)

    # Create a faulty preprocessed data file (no x_test)
    faulty_data_path = os.path.join(ARTIFACTS_TEST_DIR, "faulty_data.pkl")
    faulty_data = {
        "X_test_WRONG_KEY": np.array([[0, 0]]),
        "y_test": np.array([0]),
    }
    with open(faulty_data_path, "wb") as f:
        pickle.dump(faulty_data, f)

    with pytest.raises(KeyError):
        evaluate_model(
            trained_model_path=dummy_model_path,
            preprocessed_data_path=faulty_data_path,
            output_dir=ARTIFACTS_TEST_DIR,
        )

    # Testing invalid preprocessed data format (no y_test)
    faulty_data_y = {
        "X_test": np.array([[0, 0]]),
        "y_test_WRONG_KEY": np.array([0]),
    }
    with open(faulty_data_path, "wb") as f:
        pickle.dump(faulty_data_y, f)

    with pytest.raises(KeyError):
        evaluate_model(
            trained_model_path=dummy_model_path,
            preprocessed_data_path=faulty_data_path,
            output_dir=ARTIFACTS_TEST_DIR,
        )
