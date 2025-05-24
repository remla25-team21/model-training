import os
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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

    Returns:
        Path to the trained model
    """
    # Load parameters from params.yaml
    with open(params_filepath, "r") as params_file:
        params = yaml.safe_load(params_file)

    # Get training parameters
    train_params = params.get("train", {})
    random_state = (
        random_state
        if random_state is not None
        else train_params.get("random_state", 0)
    )
    n_estimators = train_params.get("grid_search", {}).get(
        "n_estimators", [50, 100, 200]
    )
    max_depth = train_params.get("grid_search", {}).get("max_depth", [None, 10, 20])
    cv = train_params.get("cross_validation", 5)

    # Create artifacts directory if it doesn't exist
    os.makedirs(output_model_directory, exist_ok=True)

    # Load preprocessed data
    with open(preprocessed_data_path, "rb") as f:
        data = pickle.load(f)

    X_train = data["X_train"]
    y_train = data["y_train"]

    # Hyperparameter grid for Random Forest
    param_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": [random_state],
    }

    # Grid search
    print("Starting model training with grid search...")
    grid = GridSearchCV(
        RandomForestClassifier(), param_grid, cv=cv, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    output_model_path = os.path.join(output_model_directory, "trained_model.pkl")

    # Save trained model
    with open(output_model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Trained model saved to {output_model_path}")

    return output_model_path


if __name__ == "__main__":
    train_model(preprocessed_data_path="artifacts/preprocessed_data.pkl")
