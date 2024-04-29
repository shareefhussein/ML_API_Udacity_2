"""
check scoring code
"""

import pandas as pd
import numpy as np
import logging
from joblib import load
from sklearn.model_selection import train_test_split
from src.process_data import process_data
from src.model import compute_model_metrics

def scoring_check():
    """
    scoring check function
    """
    
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
        
    df = pd.read_csv("./data/cleaned_data/census.csv")
    _, test_data = train_test_split(df, test_size=0.20)

    trained_model = load("./models/model.joblib")
    lb = load("./models/lb.joblib")
    encoder = load("./models/encoder.joblib")
    
    metrics = []
    
    for cat in cat_features:
        
        for cat_value in test_data[cat].unique():
            slice_df = test_data[test_data[cat] == cat_value]
            X_test, y_test, _, _ = process_data(
            
                slice_df, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb
            )
            
            y_pred = trained_model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test,
                                                              y_pred)
            
            variations = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, cat_value, precision, recall, fbeta)
            logging.info(variations)
            metrics.append(variations)
    
    with open("./models/slice_output.txt", 'w') as slice_output:
        for slice_ in metrics:
            slice_output.write(slice_ + '\n')






