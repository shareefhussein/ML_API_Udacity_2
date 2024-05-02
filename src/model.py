"""
Modeling code
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


def train_model(x_train, y_train):
    """
    Trains a machine learning model using gradient boosting and returns it.

    Parameters:
    - x_train (np.array): Training data.
    - y_train (np.array): Labels.

    Returns:
    - GradientBoostingClassifier: Trained machine learning model.
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    model = GradientBoostingClassifier(n_estimators=100)

    print("Training the model")
    model.fit(x_train, y_train)

    # Optionally return the cross-validation
    # scores to evaluate model performance
    scores = cross_val_score(model, x_train,
                             y_train, scoring='accuracy', cv=kfold, n_jobs=-1)
    print("Cross-validation scores:", scores)

    return model


def compute_model_metrics(labels, preds):
    """
    Validates the trained machine
    learning model using precision, recall, and F1-score.

    Parameters:
    - labels (np.array): Known labels, binarized.
    - preds (np.array): Predicted labels, binarized.

    Returns:
    - tuple: precision, recall, and fbeta-score of the model.
    """
    fbeta = fbeta_score(labels, preds, beta=1, zero_division=1)
    precision = precision_score(labels, preds, zero_division=1)
    recall = recall_score(labels, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, data):
    """
    Runs model inference and returns the predictions.

    Parameters:
    - model (GradientBoostingClassifier): Trained machine learning model.
    - data (np.array): Data used for prediction.

    Returns:
    - np.array: Predictions from the model.
    """
    predictions = model.predict(data)
    return predictions
