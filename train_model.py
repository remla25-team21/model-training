import os
import pickle
from libml.preprocessing import preprocess_train
from sklearn.naive_bayes import GaussianNB

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

    # Train
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2f}")

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
