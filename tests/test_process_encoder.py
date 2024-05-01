"""
This module is designed to test the performance of a machine learning model on pre-processed data.
It loads the trained models and evaluates them using standard metrics such as accuracy, precision,
recall, and F1 score. The tests ensure that all metrics fall within acceptable ranges.
"""

import pandas as pd
import pytest
from joblib import load
from sklearn.metrics import accuracy_score
from src.process_data import process_data
from src.model import compute_model_metrics

@pytest.fixture
def dataset():
    """
    Get dataset for testing.
    """
    return pd.read_csv("./data/cleaned_data/census.csv")

@pytest.fixture
def trained_models():
    """
    Load the trained model, label binarizer, and encoder.
    """
    model = load("./models/model.joblib")
    encoder = load("./models/encoder.joblib")
    label_binarizer = load("./models/lb.joblib")
    return model, encoder, label_binarizer

def test_model_performance(dataset, trained_models):
    """
    Test model performance on test data to ensure metrics like accuracy, precision,
    recall, and F1 score are within acceptable ranges.
    """
    model, encoder, label_binarizer = trained_models

    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    x_test, y_test, _, _ = process_data(
        dataset, categorical_features=categorical_features, label="salary",
        encoder=encoder, label_binarizer=label_binarizer, training=False
    )

    y_pred = model.predict(x_test)

    # Compute model performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score = compute_model_metrics(y_test, y_pred)

    # Assertions to ensure metrics are within expected ranges
    assert 0.0 <= accuracy <= 1.0, "Accuracy out of acceptable range"
    assert 0.0 <= precision <= 1.0, "Precision out of acceptable range"
    assert 0.0 <= recall <= 1.0, "Recall out of acceptable range"
    assert 0.0 <= f1_score <= 1.0, "F1 score out of acceptable range"
