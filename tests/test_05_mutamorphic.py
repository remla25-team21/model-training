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

def replace_with_synonym(text, original, synonym):
    return text.replace(original, synonym)

@pytest.mark.parametrize("original_review, replacements", [
    ("The food was great and service excellent.", [("great", "good"), ("excellent", "fine")]),
    ("The food was terrible and the service awful.", [("terrible", "horrible"), ("awful", "dreadful")])
])

def test_mutamorphic_synonym_consistency(trained_sentiment_model, sentiment_vectorizer, original_review, replacements):
    model = trained_sentiment_model
    vectorizer = sentiment_vectorizer
    original_vec = vectorizer.transform([original_review])
    original_pred = model.predict(original_vec)[0]

    for original, synonym in replacements:
        mutated = replace_with_synonym(original_review, original, synonym)
        mutated_vec = vectorizer.transform([mutated])
        mutated_pred = model.predict(mutated_vec)[0]
        assert mutated_pred == original_pred, (
            f"Prediction inconsistency:\n"
            f"Original: {original_review} -> {original_pred}\n"
            f"Mutated:  {mutated} -> {mutated_pred}"
        )

def test_mutamorphic_add_neutral_phrase(trained_sentiment_model, sentiment_vectorizer):
    model = trained_sentiment_model
    vectorizer = sentiment_vectorizer
    review = "The experience was terrible."
    neutralized = "To be honest, " + review

    vec_orig = vectorizer.transform([review])
    vec_neutral = vectorizer.transform([neutralized])
    pred_orig = model.predict(vec_orig)[0]
    pred_neutral = model.predict(vec_neutral)[0]

    assert pred_orig == pred_neutral, (
        f"Prediction changed after neutral phrase: '{pred_orig}' -> '{pred_neutral}'"
    )

def test_mutamorphic_repair_placeholder(trained_sentiment_model, sentiment_vectorizer):
    """
    Placeholder test to suggest the idea of automatic inconsistency repair.
    Currently does not perform real repair, just simulates detection.
    """
    model = trained_sentiment_model
    vectorizer = sentiment_vectorizer
    sentence = "The dessert was delightful."
    mutated = replace_with_synonym(sentence, "delightful", "amazing")

    orig_vec = vectorizer.transform([sentence])
    mutated_vec = vectorizer.transform([mutated])
    pred_orig = model.predict(orig_vec)[0]
    pred_mutated = model.predict(mutated_vec)[0]

    if pred_orig != pred_mutated:
        # placeholder "repair": fallback to original
        repaired = sentence
        repaired_vec = vectorizer.transform([repaired])
        repaired_pred = model.predict(repaired_vec)[0]
        assert repaired_pred == pred_orig, (
            f"Repair step failed: original='{pred_orig}', mutated='{pred_mutated}', repaired='{repaired_pred}'"
        )
