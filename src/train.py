import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_model(preprocessed_data_path, version_tag, random_state=0):
    """
    Train the model using the preprocessed data.
    
    Args:
        preprocessed_data_path: Path to the preprocessed data
        version_tag: Version tag for the artifacts
        random_state: Random state for reproducibility
        
    Returns:
        Path to the trained model
    """
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    
    # Define path for saving model
    model_path = f"artifacts/trained_model_{version_tag}.pkl"
    
    # Load preprocessed data
    with open(preprocessed_data_path, "rb") as f:
        data = pickle.load(f)
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    # Hyperparameter grid for Random Forest
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "random_state": [random_state]
    }
    
    # Grid search
    print("Starting model training with grid search...")
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    
    # Save trained model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Best parameters: {grid.best_params_}")
    print(f"Trained model saved to {model_path}")
    
    return model_path

if __name__ == "__main__":
    tag = os.getenv("GITHUB_REF_NAME", "local")
    train_model(
        preprocessed_data_path=f"artifacts/preprocessed_data_{tag}.pkl",
        version_tag=tag
    )