import os
import pickle
from libml.preprocessing import preprocess_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def train_and_save_model(data_path, version_tag, test_size=0.2, random_state=0):
    os.makedirs("artifacts", exist_ok=True)

    model_path = f"artifacts/sentiment_model_{version_tag}.pkl"
    vectorizer_path = f"artifacts/c1_BoW_Sentiment_Model_{version_tag}.pkl"

    # Preprocess and save vectorizer using lib-ml
    X_train, X_test, y_train, y_test = preprocess_train(
        dataset_filepath=data_path,
        test_size=test_size,
        random_state=random_state,
        vectorizer_output=vectorizer_path
    )

    # Hyperparameter grid for Random Forest
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "random_state": [random_state]
    }

    # Grid search
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2f}")
    print("Classification report:")
    print(classification_report(y_test, model.predict(X_test)))

    # Save
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    tag = os.getenv("GITHUB_REF_NAME", "local")
    train_and_save_model(
        data_path="data/a1_RestaurantReviews_HistoricDump.tsv",
        version_tag=tag
    )
