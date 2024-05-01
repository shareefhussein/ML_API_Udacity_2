"""
Code to train and evaluate the model
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.process_data import process_data
from src.model import train_model

def train_test_model():
    """
    This function trains the model and prints performance metrics.
    """
    # Load the dataset
    data_frame = pd.read_csv('./data/cleaned_data/census.csv')

    # Define categorical and continuous features as they appear in the test.py file
    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    # Splitting into training and testing datasets for model training and evaluation
    train_data, test_data = train_test_split(data_frame, test_size=0.20, random_state=42)

    # Process training data
    x_train, y_train, encoder, label_binarizer = process_data(
        train_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=True
    )

    # Process testing data
    x_test, y_test, _, _ = process_data(
        test_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=False, 
        encoder=encoder, 
        label_binarizer=label_binarizer
    )

    # Train the model
    model = train_model(x_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(x_test)

    # Compute performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    # Print performance metrics
    print(f"Model Performance Metrics:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

    # Dumping model and preprocessors to files for later use
    dump(model, "./models/model.joblib")
    dump(encoder, "./models/encoder.joblib")
    dump(label_binarizer, "./models/lb.joblib")

# Ensure this script is only executed when run as a main script
if __name__ == "__main__":
    train_test_model()
