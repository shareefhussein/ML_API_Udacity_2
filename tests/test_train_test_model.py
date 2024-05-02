"""
This module is designed to perform tests on
the training and testing phases of a machine learning model,
including verifying the creation of
 files and the proper execution of data splitting.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.train_test_model import train_test_model
from src.process_data import process_data


def test_model_files_generated():
    """
    Test to ensure that all necessary model files are generated.
    """
    train_test_model()

    assert os.path.isfile("./models/model.joblib"), \
        "Model file is missing"
    assert os.path.isfile("./models/lb.joblib"), \
        "Label binarizer file is missing"
    assert os.path.isfile("./models/encoder.joblib"), \
        "Encoder file is missing"


def test_data_split():
    """
    Test the data splitting process to ensure it
    correctly splits the data and processes it.
    """
    # Load the cleaned data
    data_frame = pd.read_csv('./data/cleaned_data/census.csv')
    train_data, _ = train_test_split(data_frame,
                                     test_size=0.20, random_state=42)

    categorical_features = [
        "workclass", "education",
        "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    # Process the training data
    x_train, y_train, _, _ = process_data(
        train_data, categorical_features=categorical_features,
        label="salary", training=True
    )

    # Validate that processed training data
    assert x_train.shape[0] > 0, "X_train is empty"
    assert y_train.shape[0] > 0, "Y_train is empty"
    assert x_train.shape[0] == y_train.shape[0], \
        "X_train and Y_train sample sizes do not match"
