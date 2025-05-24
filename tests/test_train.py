import os
import pickle
import yaml
import shutil
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train import train_model

# direcotry for storing artifacts created by train_model function
TEST_ARTIFACTS_DIR = "artifacts_test"
# dummy params.yaml file for the test
DUMMY_PARAMS_FILE = os.path.join(TEST_ARTIFACTS_DIR, "params.yaml")
# direcotry for storing dummy preprocessed data for tests
TEST_INPUT_ARTIFACTS_DIR = "test_train_input_artifacts"


# create dummy files and directories for the test
@pytest.fixture(scope="function")
def setup_and_cleanup_train_environment():
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)

    # 1Create dummy params.yaml in the project root
    dummy_params_content = {
        "train": {
            "random_state": 42,
            "grid_search": {
                "n_estimators": [10],
                "max_depth": [3],
            },
            "cross_validation": 2,
        }
    }
    with open(DUMMY_PARAMS_FILE, "w") as f:
        yaml.dump(dummy_params_content, f)

    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(TEST_INPUT_ARTIFACTS_DIR, exist_ok=True)

    # Create dummy preprocessed_data.pkl in a test-specific input directory
    dummy_preprocessed_data_path = os.path.join(
        TEST_INPUT_ARTIFACTS_DIR, "dummy_train_data.pkl"
    )

    # Create simple X_train, y_train
    X_train_dummy = np.array([[1, 1], [2, 2], [1, 0], [0, 1], [1, 2], [2, 0]])
    y_train_dummy = np.array([0, 1, 0, 1, 0, 1])  # Balanced classes for simplicity
    preprocessed_data = {"X_train": X_train_dummy, "y_train": y_train_dummy}
    with open(dummy_preprocessed_data_path, "wb") as f:
        pickle.dump(preprocessed_data, f)

    yield dummy_preprocessed_data_path

    if os.path.exists(DUMMY_PARAMS_FILE):
        os.remove(DUMMY_PARAMS_FILE)

    if os.path.exists(TEST_INPUT_ARTIFACTS_DIR):
        shutil.rmtree(TEST_INPUT_ARTIFACTS_DIR)

    if os.path.exists(TEST_ARTIFACTS_DIR):
        shutil.rmtree(TEST_ARTIFACTS_DIR)


# Test successful execution of train_model function
def test_train_model_success(setup_and_cleanup_train_environment):
    dummy_data_path = setup_and_cleanup_train_environment
    test_output_path = os.path.join(TEST_ARTIFACTS_DIR, "trained_model.pkl")
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)

    returned_model_path = train_model(
        preprocessed_data_path=dummy_data_path,
        random_state=42,
        output_model_directory=TEST_ARTIFACTS_DIR,
        params_filepath=DUMMY_PARAMS_FILE,
    )

    assert os.path.normpath(returned_model_path) == os.path.normpath(
        test_output_path
    ), "Returned model path is incorrect."

    with open(test_output_path, "rb") as f:
        model = pickle.load(f)

    assert isinstance(
        model, RandomForestClassifier
    ), "Saved model is not a RandomForestClassifier instance."


# Check for the case when params.yaml is missing
def test_train_model_params_file_not_found(setup_and_cleanup_train_environment):
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)

    dummy_data_path = setup_and_cleanup_train_environment
    if os.path.exists(DUMMY_PARAMS_FILE):
        os.remove(DUMMY_PARAMS_FILE)

    with pytest.raises(FileNotFoundError):
        train_model(
            preprocessed_data_path=dummy_data_path,
            output_model_directory=TEST_ARTIFACTS_DIR,
            params_filepath=DUMMY_PARAMS_FILE,
        )


# Check for the case when preprocessed data file is missing
def test_train_model_preprocessed_data_not_found():
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)

    dummy_params_content = {
        "train": {
            "random_state": 0,
            "grid_search": {"n_estimators": [10], "max_depth": [3]},
            "cross_validation": 2,
        }
    }
    with open(DUMMY_PARAMS_FILE, "w") as f:
        yaml.dump(dummy_params_content, f)

    with pytest.raises(FileNotFoundError):
        train_model(
            preprocessed_data_path="non_existent.pkl",
            output_model_directory=TEST_ARTIFACTS_DIR,
            params_filepath=DUMMY_PARAMS_FILE,
        )

    os.remove(DUMMY_PARAMS_FILE)


# Check for the case when processing data is missing expected keys
def test_train_model_data_key_error(setup_and_cleanup_train_environment):
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)

    dummy_data_path_valid = setup_and_cleanup_train_environment

    faulty_data_path = os.path.join(TEST_INPUT_ARTIFACTS_DIR, "dummy_train_data.pkl")

    # Test for missing X_train
    faulty_data_x = {"X_train_WRONG_KEY": np.array([[0, 0]]), "y_train": np.array([0])}
    with open(faulty_data_path, "wb") as f:
        pickle.dump(faulty_data_x, f)

    with pytest.raises(KeyError):
        train_model(
            preprocessed_data_path=faulty_data_path,
            output_model_directory=TEST_ARTIFACTS_DIR,
            params_filepath=DUMMY_PARAMS_FILE,
        )

    # Test for missing y_train
    faulty_data_y = {"X_train": np.array([[0, 0]]), "y_train_WRONG_KEY": np.array([0])}
    with open(faulty_data_path, "wb") as f:
        pickle.dump(faulty_data_y, f)

    with pytest.raises(KeyError):
        train_model(preprocessed_data_path=faulty_data_path)


# Test for robustness of train_model function to missing or faulty params.yaml keys
def test_train_model_params_structure_robustness(setup_and_cleanup_train_environment):
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)

    dummy_data_path = setup_and_cleanup_train_environment

    # Overwrite params.yaml with a version missing some grid_search keys
    faulty_params_content = {
        "train": {
            "random_state": 10,
            # "grid_search": {} # Missing grid_search entirely, or some sub-keys
            "cross_validation": 2,
        }
    }

    with open(DUMMY_PARAMS_FILE, "w") as f:
        yaml.dump(faulty_params_content, f)

    # Expect the function to run using default values specified in train_model.py
    # and not raise a KeyError for missing param keys.
    try:
        test_output_path = os.path.join(TEST_ARTIFACTS_DIR, "trained_model.pkl")
        returned_model_path = train_model(
            preprocessed_data_path=dummy_data_path,
            random_state=10,
            output_model_directory=TEST_ARTIFACTS_DIR,
            params_filepath=DUMMY_PARAMS_FILE,
        )
        assert os.path.exists(test_output_path)
        assert os.path.exists(
            test_output_path
        ), "Model not created with faulty (but handled) params."
        with open(test_output_path, "rb") as f:
            model = pickle.load(f)
        assert isinstance(model, RandomForestClassifier)
    except Exception as e:
        pytest.fail(f"train_model failed with faulty but handled params: {e}")
