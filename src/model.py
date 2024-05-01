"""
Modeling code
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def train_model(x_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    x_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    gradient_boosting_model = GradientBoostingClassifier(n_estimators=100)

    print("Training the model")
    gradient_boosting_model.fit(x_train, y_train)

    # The scores variable is being calculated but not used, consider storing or returning it if needed.
    scores = cross_val_score(
        gradient_boosting_model, x_train, y_train, scoring='accuracy', cv=kfold, n_jobs=-1
    )

    return gradient_boosting_model

def compute_model_metrics(labels, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    labels : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(labels, preds, beta=1, zero_division=1)
    precision = precision_score(labels, preds, zero_division=1)
    recall = recall_score(labels, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, data):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Trained machine learning model.
    data : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(data)

    return predictions
