"""
Modeling code

"""

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from src.process_data import process_data

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    GB_model = GradientBoostingClassifier(n_estimators=100)
    
    print("Training the model")
    GB_model.fit(X_train, y_train)
    
    scores = cross_val_score(
        GB_model, X_train, y_train, scoring='accuracy', cv=kfold, n_jobs=-1
    )
    
    return GB_model
    


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    
    return predictions


