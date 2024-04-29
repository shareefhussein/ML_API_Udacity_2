import os
import pytest
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score
from src.process_data import process_data
from src.model import compute_model_metrics

@pytest.fixture
def data():
    """
    Get dataset
    """
    data = pd.read_csv("./data/cleaned_data/census.csv")
    return data

@pytest.fixture
def saved_models():
    """
    Load the trained model, label binarizer, and encoder
    """
    model = load("./models/model.joblib")
    encoder = load("./models/encoder.joblib")
    lb = load("./models/lb.joblib")
    
    return model, encoder, lb


def test_model_performance(data, saved_models):
    """
    Test model performance on test data
    """
    model, encoder, lb = saved_models

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_test, y_test, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        training=False
    )

    y_pred = model.predict(X_test)

    # Compute model performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1 = compute_model_metrics(y_test, y_pred)

    # Ensure metrics are within acceptable ranges
    assert accuracy >= 0.0 and accuracy <= 1.0
    assert precision >= 0.0 and precision <= 1.0
    assert recall >= 0.0 and recall <= 1.0
    assert f1 >= 0.0 and f1 <= 1.0


    
    