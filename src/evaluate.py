"""
Module for evaluating trained machine learning models.

This module provides functionality to evaluate model performance,
calculate metrics, and save evaluation results for sentiment analysis models.
"""

import os
import pickle
import json
import shutil
from sklearn.metrics import classification_report, accuracy_score


def evaluate_model(trained_model_path, preprocessed_data_path):
    """
    Evaluate the model performance and save metrics.

    Args:
        trained_model_path: Path to the trained model
        preprocessed_data_path: Path to the preprocessed data

    Returns:
        Path to the evaluation metrics file
    """
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)

    # Define path for saving evaluation metrics
    metrics_path = "artifacts/metrics.json"

    # Load trained model
    with open(trained_model_path, "rb") as f:
        model = pickle.load(f)  # nosec B301

    # Load preprocessed data
    with open(preprocessed_data_path, "rb") as f:
        data = pickle.load(f)  # nosec B301

    X_test = data["X_test"]
    y_test = data["y_test"]

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Create metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1-score": report["weighted avg"]["f1-score"],
        "support": report["weighted avg"]["support"],
        "classification_report": report
    }

    # Save metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Create model artifact for final release
    final_model_path = "artifacts/sentiment_model.pkl"

    # Copy the trained model to the final model path
    shutil.copyfile(trained_model_path, final_model_path)

    print(f"Model accuracy on test set: {accuracy:.2f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print(f"Evaluation metrics saved to {metrics_path}")
    print(f"Final model saved to {final_model_path}")

    return metrics_path


if __name__ == "__main__":
    evaluate_model(
        trained_model_path="artifacts/trained_model.pkl",
        preprocessed_data_path="artifacts/preprocessed_data.pkl"
    )
