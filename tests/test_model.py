"""
This module is designed to test the
functionality of the machine learning model training process
for the census income prediction project.
It includes tests to ensure that models are trained correctly
and are instances of the expected class.
"""

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from src.process_data import process_data
from src.model import train_model


@pytest.fixture
def dataset():
    """
    Fixture to load the dataset used for testing.
    """
    return pd.read_csv("./data/cleaned_data/census.csv")


def test_trained_model(dataset):
    """
    Test that the trained model is an instance of GradientBoostingClassifier.
    """
    train_data, _ = train_test_split(dataset, test_size=0.20, random_state=42)

    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    x_train, y_train, _, _ = process_data(
        train_data, categorical_features=categorical_features, label="salary",
        training=True
    )

    model = train_model(x_train, y_train)
    assert isinstance(model, GradientBoostingClassifier), \
        "Model is not an instance of GradientBoostingClassifier"
