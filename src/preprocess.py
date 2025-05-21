import os
import pickle
from libml.preprocessing import preprocess_train

def preprocess_data(data_path, version_tag, test_size=0.2, random_state=0):
    """
    Preprocess the data and save the intermediate results.
    
    Args:
        data_path: Path to the raw data file
        version_tag: Version tag for the artifacts
        test_size: Test split size
        random_state: Random state for reproducibility
    
    Returns:
        Path to the preprocessed data
    """
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    
    # Define paths for saving preprocessing results
    preprocessed_path = f"artifacts/preprocessed_data_{version_tag}.pkl"
    vectorizer_path = f"artifacts/c1_BoW_Sentiment_Model_{version_tag}.pkl"
    
    # Preprocess and save vectorizer using lib-ml
    X_train, X_test, y_train, y_test = preprocess_train(
        dataset_filepath=data_path,
        test_size=test_size,
        random_state=random_state,
        vectorizer_output=vectorizer_path
    )
    
    # Save preprocessed data
    with open(preprocessed_path, "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }, f)
    
    print(f"Preprocessed data saved to {preprocessed_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    
    return preprocessed_path

if __name__ == "__main__":
    tag = os.getenv("GITHUB_REF_NAME", "local")
    preprocess_data(
        data_path="data/raw/a1_RestaurantReviews_HistoricDump.tsv",
        version_tag=tag
    )