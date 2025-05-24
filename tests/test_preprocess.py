import os
import pickle
import pytest
from preprocess import preprocess_data


# Dummy data file creation
@pytest.fixture(scope="module")
def dummy_data_file():
    dummy_file_path = "dummy_test_data.tsv"
    dummy_data_content = """Review\tLiked
    Wow... Loved this place.\t1
    Crust is not good.\t0
    Not tasty and the texture was just nasty.\t0
    """
    with open(dummy_file_path, "w") as f:
        f.write(dummy_data_content)
    # Make this dummy file path available for the test
    yield dummy_file_path
    os.remove(dummy_file_path)


# Clean up the artifacts directory and files after each test
@pytest.fixture(scope="function", autouse=True)
def cleanup_artifacts():
    artifacts_dir = "artifacts"
    preprocessed_file = os.path.join(artifacts_dir, "preprocessed_data.pkl")
    vectorizer_file = os.path.join(artifacts_dir, "c1_BoW_Sentiment_Model.pkl")

    # Cleanup before test
    if os.path.exists(preprocessed_file):
        os.remove(preprocessed_file)
    if os.path.exists(vectorizer_file):
        os.remove(vectorizer_file)
    if os.path.exists(artifacts_dir) and not os.listdir(artifacts_dir):
        os.rmdir(artifacts_dir)

    # runthe test
    yield

    # Cleanup after test
    if os.path.exists(preprocessed_file):
        os.remove(preprocessed_file)
    if os.path.exists(vectorizer_file):
        os.remove(vectorizer_file)
    if os.path.exists(artifacts_dir) and not os.listdir(artifacts_dir):
        os.rmdir(artifacts_dir)


def test_preprocess_data(dummy_data_file):
    # Run the preprocess function with the dummy data file path
    returned_path = preprocess_data(
        data_path=dummy_data_file, test_size=0.5, random_state=42
    )

    # Check if the function returns the expected path
    assert returned_path == "artifacts/preprocessed_data.pkl"

    # Check if the output files were created
    preprocessed_path = "artifacts/preprocessed_data.pkl"
    vectorizer_path = "artifacts/c1_BoW_Sentiment_Model.pkl"
    assert os.path.exists(preprocessed_path), "Preprocessed data file was not created."
    assert os.path.exists(vectorizer_path), "Vectorizer file was not created."

    # Check the content of preprocessed_data.pkl
    with open(preprocessed_path, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, dict), "Preprocessed data should be a dictionary."
    expected_keys = ["X_train", "X_test", "y_train", "y_test"]
    for key in expected_keys:
        assert key in data, f"Key '{key}' missing in preprocessed data."

    # Check that the data was split in such a way that the total number of samples is correct
    assert hasattr(data["X_train"], "__len__")
    assert hasattr(data["X_test"], "__len__")
    assert hasattr(data["y_train"], "__len__")
    assert hasattr(data["y_test"], "__len__")
    assert len(data["X_train"]) + len(data["X_test"]) == 3
    assert len(data["y_train"]) + len(data["y_test"]) == 3

    # Check if the vectorizer file is not empty
    assert os.path.getsize(vectorizer_path) > 0, "Vectorizer file is empty."

    # 5. Check if the artifacts directory was created
    assert os.path.exists("artifacts"), "Artifacts directory was not created."


# Test that if the data file does not exist, it thorws an error
def test_preprocess_data_no_file():
    non_existent_file = "path/to/non_existent_file.tsv"
    with pytest.raises((FileNotFoundError, Exception)) as excinfo:
        preprocess_data(data_path=non_existent_file)
    print(f"Caught expected error: {excinfo.value}")
