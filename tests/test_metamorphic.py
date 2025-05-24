import os
import pickle
import pytest
import numpy as np


@pytest.fixture(scope="module")
def trained_sentiment_model():
    model_path = "../artifacts/trained_model.pkl"
    model_path = os.path.abspath("artifacts/trained_model.pkl")

    if not os.path.exists(model_path):
        pytest.fail(
            f"ERROR: Model file not found at {model_path}. "
            f"Ensure the DVC 'train' stage has been run."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


@pytest.fixture(scope="module")
def sentiment_vectorizer():
    vectorizer_path = "artifacts/c1_BoW_Sentiment_Model.pkl"
    if not os.path.exists(vectorizer_path):
        pytest.fail(
            f"ERROR: Vectorizer file not found at {vectorizer_path}. "
            f"Ensure the DVC 'preprocess' stage has been run."
        )
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer


def replace_with_synonym(text, original_word, synonym):
    return text.replace(original_word, synonym)


# Metamorphic tests for sentiment analysis model
def test_metamorphic_synonym_positive_review(
    trained_sentiment_model, sentiment_vectorizer
):
    model = trained_sentiment_model

    original_review = "The food was great and service excellent."
    original_review_vectorized = sentiment_vectorizer.transform([original_review])
    original_prediction = model.predict(original_review_vectorized)[0]

    # context similar alternative 1
    transformed_review_1_text = replace_with_synonym(original_review, "great", "good")
    transformed_review_1_vectorized = sentiment_vectorizer.transform(
        [transformed_review_1_text]
    )
    transformed_prediction_1 = model.predict(transformed_review_1_vectorized)[0]

    assert (
        transformed_prediction_1 == original_prediction
    ), f"Sentiment changed from '{original_prediction}' to '{transformed_prediction_1}' after synonym replacement (great -> good)."

    # context similar alternative 2
    transformed_review_2_text = replace_with_synonym(
        original_review, "excellent", "fine"
    )
    transformed_review_2_vectorized = sentiment_vectorizer.transform(
        [transformed_review_2_text]
    )
    transformed_prediction_2 = model.predict(transformed_review_2_vectorized)[0]
    assert (
        transformed_prediction_2 == original_prediction
    ), f"Sentiment changed from '{original_prediction}' to '{transformed_prediction_2}' after synonym replacement (excellent -> fine)."


# Metamorphic tests for sentiment analysis model
def test_metamorphic_synonym_negative_review(
    trained_sentiment_model, sentiment_vectorizer
):
    model = trained_sentiment_model
    original_review = "The food was terrible and the service awful."
    original_review_vectorized = sentiment_vectorizer.transform([original_review])
    original_prediction = model.predict(original_review_vectorized)[0]

    # context similar alternative 1
    transformed_review_1_text = replace_with_synonym(
        original_review, "terrible", "horrible"
    )
    transformed_review_1_vectorized = sentiment_vectorizer.transform(
        [transformed_review_1_text]
    )
    transformed_prediction_1 = model.predict(transformed_review_1_vectorized)[0]

    assert (
        transformed_prediction_1 == original_prediction
    ), f"Sentiment changed from '{original_prediction}' to '{transformed_prediction_1}' after synonym replacement (terrible -> horrible)."

    # context similar alternative 2
    transformed_review_2_text = replace_with_synonym(
        original_review, "awful", "dreadful"
    )
    transformed_review_2_vectorized = sentiment_vectorizer.transform(
        [transformed_review_2_text]
    )
    transformed_prediction_2 = model.predict(transformed_review_2_vectorized)[0]

    assert (
        transformed_prediction_2 == original_prediction
    ), f"Sentiment changed from '{original_prediction}' to '{transformed_prediction_2}' after synonym replacement (awful -> dreadful)."


def test_metamorphic_add_neutral_phrase_negative_review(
    trained_sentiment_model, sentiment_vectorizer
):
    model = trained_sentiment_model
    original_review = "The experience was terrible."
    original_review_vectorized = sentiment_vectorizer.transform([original_review])
    original_prediction = model.predict(original_review_vectorized)[0]

    transformed_review_text = "To be honest, " + original_review
    transformed_review_vectorized = sentiment_vectorizer.transform(
        [transformed_review_text]
    )
    transformed_prediction = model.predict(transformed_review_vectorized)[0]

    # assert original_prediction == "Negative" # Base assumption
    assert (
        transformed_prediction == original_prediction
    ), f"Sentiment changed from '{original_prediction}' to '{transformed_prediction}' after adding a neutral phrase."
