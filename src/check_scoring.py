"""
Check scoring code
"""

import logging
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from src.process_data import process_data
from src.model import compute_model_metrics

logging.basicConfig(level=logging.INFO)


def scoring_check():
    """
    Perform a scoring check by evaluating the model across different slices
    of the dataset based on categorical features.
    """
    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    data_frame = pd.read_csv("./data/cleaned_data/census.csv")
    _, test_data = train_test_split(data_frame, test_size=0.20)

    trained_model = load("./models/model.joblib")
    label_binarizer = load("./models/lb.joblib")
    encoder = load("./models/encoder.joblib")

    metrics = []

    for category in categorical_features:
        for category_value in test_data[category].unique():
            slice_data_frame = test_data[test_data[category] == category_value]
            x_test, y_test, _, _ = process_data(
                slice_data_frame, categorical_features=categorical_features,
                label='salary', training=False, encoder=encoder,
                label_binarizer=label_binarizer
            )

            y_pred = trained_model.predict(x_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

            variation = (f"[{category}->{category_value}] Precision: {precision} "
                         f"Recall: {recall} FBeta: {fbeta}")
            logging.info(variation)
            metrics.append(variation)

    with open("./models/slice_output.txt", 'w', encoding='utf-8') as slice_output:
        slice_output.writelines(metric + '\n' for metric in metrics)


if __name__ == "__main__":
    scoring_check()
