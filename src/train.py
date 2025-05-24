"""
Module for training machine learning models for sentiment analysis.

This module handles the complete training pipeline including parameter loading,
hyperparameter tuning through grid search, and model persistence. It integrates
with configuration files to make the training process flexible and reproducible.
"""

import os
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def prepare_training_config(preprocessed_data_path, params_filepath, random_state=None):
    """
    Load training configuration, data, and grid search parameters.

    Args:
        preprocessed_data_path (str): Path to preprocessed training data.
        params_filepath (str): Path to the parameter YAML file.
        random_state (int or None): Seed for reproducibility.

    Returns:
        tuple: X_train, y_train, param_grid, cv_folds, final_random_state
    """
    # Load parameters
    with open(params_filepath, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    train_params = params.get("train", {})
    final_random_state = (
        random_state
        if random_state is not None
        else train_params.get("random_state", 0)
    )

    # Load preprocessed data
    with open(preprocessed_data_path, "rb") as file:
        data = pickle.load(file)  # nosec B301

    X_train, y_train = data["X_train"], data["y_train"]

    # Construct parameter grid
    param_grid = {
        "n_estimators": train_params.get("grid_search", {}).get(
            "n_estimators", [50, 100, 200]
        ),
        "max_depth": train_params.get("grid_search", {}).get(
            "max_depth", [None, 10, 20]
        ),
        "random_state": [final_random_state],
    }

    cv_folds = train_params.get("cross_validation", 5)

    return X_train, y_train, param_grid, cv_folds, final_random_state


def train_model(
    preprocessed_data_path,
    random_state=None,
    output_model_directory="artifacts",
    params_filepath="params.yaml",
):
    """
    Train the model using the preprocessed data.

    Args:
        preprocessed_data_path: Path to the preprocessed data
        random_state: Random state for reproducibility
        output_model_directory (str): Directory to save the trained model.
        params_filepath (str): Path to training configuration parameters.

    Returns:
        Dictionary containing training parameters
    """
    X_train, y_train, param_grid, cv_folds, random_state = prepare_training_config(
        preprocessed_data_path, params_filepath, random_state
    )

    # Create artifacts directory if it doesn't exist
    os.makedirs(output_model_directory, exist_ok=True)
    model_path = os.path.join(output_model_directory, "trained_model.pkl")

    print("Starting model training with grid search...")
    grid_search = GridSearchCV(
        RandomForestClassifier(), param_grid, cv=cv_folds, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        pickle.dump(grid_search.best_estimator_, f)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Trained model saved to {model_path}")

    return model_path


if __name__ == "__main__":
    train_model(preprocessed_data_path="artifacts/preprocessed_data.pkl")
