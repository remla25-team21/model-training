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


def load_training_parameters(params_file_path="params.yaml"):
    """
    Load training parameters from configuration file.

    Args:
        params_file_path: Path to the parameters YAML file

    Returns:
        Dictionary containing training parameters
    """
    with open(params_file_path, "r", encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)
    return params.get("train", {})


def create_parameter_grid(train_params, random_state):
    """
    Create hyperparameter grid for model tuning.

    Args:
        train_params: Dictionary of training parameters
        random_state: Random state for reproducibility

    Returns:
        Dictionary containing parameter grid for GridSearchCV
    """
    # Extract grid search parameters with sensible defaults
    grid_params = train_params.get("grid_search", {})
    n_estimators = grid_params.get("n_estimators", [50, 100, 200])
    max_depth = grid_params.get("max_depth", [None, 10, 20])

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": [random_state]
    }


def train_model(preprocessed_data_path, random_state=None):
    """
    Train the model using the preprocessed data with hyperparameter tuning.

    This function loads configuration parameters, sets up a grid search for
    hyperparameter optimization, trains a Random Forest classifier, and
    saves the best model for later use.

    Args:
        preprocessed_data_path: Path to the preprocessed data
        random_state: Random state for reproducibility

    Returns:
        Path to the trained model
    """
    # Load training parameters from configuration
    train_params = load_training_parameters()

    # Determine random state (use parameter if provided, otherwise use config)
    if random_state is None:
        random_state = train_params.get("random_state", 0)

    # Get cross-validation parameter
    cv_folds = train_params.get("cross_validation", 5)

    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)

    # Define path for saving the trained model
    model_path = "artifacts/trained_model.pkl"

    # Load preprocessed training data
    with open(preprocessed_data_path, "rb") as f:
        data = pickle.load(f)  # nosec B301

    X_train = data["X_train"]
    y_train = data["y_train"]

    # Create hyperparameter grid for tuning
    param_grid = create_parameter_grid(train_params, random_state)

    # Perform grid search with cross-validation
    print("Starting model training with grid search...")
    print(f"Parameter grid: {param_grid}")
    print(f"Cross-validation folds: {cv_folds}")

    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=cv_folds,
        n_jobs=-1,
        verbose=1
    )

    # Fit the grid search to find the best hyperparameters
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Save the trained model to disk
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    # Display training results
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Trained model saved to {model_path}")

    return model_path


if __name__ == "__main__":
    train_model(
        preprocessed_data_path="artifacts/preprocessed_data.pkl"
    )
