"""
This module is designed to test
the performance of a machine learning model on pre-processed data.
It loads the trained models and
evaluates them using standard metrics such as accuracy, precision,
recall, and F1 score.
The tests ensure that all metrics fall within acceptable ranges.
"""

import pandas as pd
import pytest
import warnings
from joblib import load
from sklearn.metrics import accuracy_score
from src.process_data import process_data
from src.model import compute_model_metrics


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture
def dataset():
    """
    Load and return the dataset used for testing.
    """
    return pd.read_csv("./data/cleaned_data/census.csv")


@pytest.fixture
def trained_models():
    """
    Load and return the trained model, label binarizer, and encoder.
    """
    model = load("./models/model.joblib")
    encoder = load("./models/encoder.joblib")
    label_binarizer = load("./models/lb.joblib")
    return model, encoder, label_binarizer


