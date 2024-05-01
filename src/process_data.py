"""
Prepare data code
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def process_data(
        data_frame,
        categorical_features=None,
        continuous_features=None,
        label=None,
        training=True,
        encoder=None,
        label_binarizer=None):
    """
    Process the data used in the machine learning pipeline.

    Inputs
    ------
    data_frame : pd.DataFrame
        Dataframe containing the features and label. Columns specified in `categorical_features` and `continuous_features`
    categorical_features : list[str]
        List containing the names of the categorical features (default=None)
    continuous_features : list[str]
        List containing the names of the continuous features (default=None)
    label : str
        Name of the label column in `data_frame`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    label_binarizer : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    data_frame : np.array
        Processed data.
    labels : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    label_binarizer : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    if categorical_features is None:
        categorical_features = []

    if continuous_features is None:
        continuous_features = []

    if label is not None:
        labels = data_frame[label].values
        data_frame = data_frame.drop(label, axis=1)
    else:
        labels = np.array([])

    if training:
        if encoder is None:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        if label_binarizer is None:
            label_binarizer = LabelBinarizer()
        encoder.fit(data_frame[categorical_features])
        if label is not None:
            label_binarizer.fit(labels.reshape(-1, 1))

    categorical_data = encoder.transform(data_frame[categorical_features])
    continuous_data = data_frame[continuous_features].values
    processed_data = np.concatenate([continuous_data, categorical_data], axis=1)

    if label is not None:
        labels = label_binarizer.transform(labels.reshape(-1, 1)).ravel()

    return processed_data, labels, encoder, label_binarizer

