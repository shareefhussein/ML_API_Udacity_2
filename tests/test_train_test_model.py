"""
train test split test

"""
import os
import pytest
import pandas as pd
from joblib import load
from src.train_test_model import train_test_model
from src.process_data import process_data
from sklearn.model_selection import train_test_split


def test_model_files_generated():
    train_test_model()

    assert os.path.isfile("./models/model.joblib")
    assert os.path.isfile("./models/lb.joblib")
    assert os.path.isfile("./models/encoder.joblib")


def test_data_split():
    # Run the train_test_model function
    train_test_model()
    
    df = pd.read_csv('./data/cleaned_data/census.csv')
    train, _ = train_test_split(df, test_size=0.20)  


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
    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )


    # Check if X_train and y_train are not empty
    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0

    # Check if X_train and y_train have the same number of samples
    assert X_train.shape[0] == y_train.shape[0]


    