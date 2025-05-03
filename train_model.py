import pickle
from libml.preprocessing import preprocess_train
from sklearn.naive_bayes import GaussianNB
import shutil

def train_and_save_model(data_path, model_path, test_size=0.2, random_state=0):
    # Preprocess using `lib-ml``
    X_train, X_test, y_train, y_test = preprocess_train(data_path, test_size, random_state)

    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)
    shutil.copy("../c1_BoW_Sentiment_Model.pkl", "c1_BoW_Sentiment_Model.pkl")

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2f}")

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model(
        data_path="data/a1_RestaurantReviews_HistoricDump.tsv",
        model_path="sentiment_model.pkl",
    )
