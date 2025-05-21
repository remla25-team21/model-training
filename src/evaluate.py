import os
import pickle
import json
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(trained_model_path, preprocessed_data_path, version_tag):
    """
    Evaluate the model performance and save metrics.
    
    Args:
        trained_model_path: Path to the trained model
        preprocessed_data_path: Path to the preprocessed data
        version_tag: Version tag for the artifacts
        
    Returns:
        Path to the evaluation metrics file
    """
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    
    # Define path for saving evaluation metrics
    metrics_path = f"artifacts/metrics_{version_tag}.json"
    
    # Load trained model
    with open(trained_model_path, "rb") as f:
        model = pickle.load(f)
    
    # Load preprocessed data
    with open(preprocessed_data_path, "rb") as f:
        data = pickle.load(f)
    
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
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Create model artifact for final release
    final_model_path = f"artifacts/sentiment_model_{version_tag}.pkl"
    
    # Copy the trained model to the final model path
    shutil.copyfile(trained_model_path, final_model_path)
    
    print(f"Model accuracy on test set: {accuracy:.2f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print(f"Evaluation metrics saved to {metrics_path}")
    print(f"Final model saved to {final_model_path}")
    
    return metrics_path

if __name__ == "__main__":
    tag = os.getenv("GITHUB_REF_NAME", "local")
    evaluate_model(
        trained_model_path=f"artifacts/trained_model_{tag}.pkl",
        preprocessed_data_path=f"artifacts/preprocessed_data_{tag}.pkl",
        version_tag=tag
    )