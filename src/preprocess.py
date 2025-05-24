"""
Module for preprocessing text data for sentiment analysis.

This module handles the preprocessing pipeline for restaurant review data,
including text cleaning, feature extraction, and train-test splitting.
It integrates with the libml preprocessing utilities to create vectorized
representations of text data suitable for machine learning models.
"""

import os
import pickle
from libml.preprocessing import preprocess_train


def preprocess_data(data_path, test_size=0.2, random_state=0, output_dir="artifacts"):
    """
    Preprocess the data and save the intermediate results.

    Args:
        data_path: Path to the raw data file
        test_size: Test split size
        random_state: Random state for reproducibility
        output_dir: Directory to save the preprocessed data and vectorizer

    Returns:
        Path to the preprocessed data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define paths for saving preprocessing results
    preprocessed_path = os.path.join(output_dir, "preprocessed_data.pkl")
    vectorizer_path = os.path.join(output_dir, "c1_BoW_Sentiment_Model.pkl")

    # Preprocess and save vectorizer using lib-ml
    X_train, X_test, y_train, y_test = preprocess_train(
        dataset_filepath=data_path,
        test_size=test_size,
        random_state=random_state,
        vectorizer_output=vectorizer_path,
    )

    # Save preprocessed data
    with open(preprocessed_path, "wb") as f:
        pickle.dump(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            },
            f,
        )

    print(f"Preprocessed data saved to {preprocessed_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

    return preprocessed_path


if __name__ == "__main__":
    preprocess_data(data_path="data/raw/a1_RestaurantReviews_HistoricDump.tsv")
