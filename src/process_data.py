"""
Prepare data code
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def process_data(data_frame, categorical_features=None, continuous_features=None,
                 label=None, training=True, encoder=None, label_binarizer=None):
    """
    Process the data used in the machine learning pipeline.

    Parameters:
    - data_frame (pd.DataFrame): Dataframe containing the features and label.
    - categorical_features (list[str], optional): Names of the categorical features.
    - continuous_features (list[str], optional): Names of the continuous features.
    - label (str, optional): Name of the label column in `data_frame`. If None, returns an empty array for y.
    - training (bool): Indicates if training mode or inference/validation mode.
    - encoder (OneHotEncoder, optional): Trained sklearn OneHotEncoder, used if training=False.
    - label_binarizer (LabelBinarizer, optional): Trained sklearn LabelBinarizer, used if training=False.

    Returns:
    - np.array: Processed data.
    - np.array: Processed labels if labeled=True, otherwise empty array.
    - OneHotEncoder: Trained OneHotEncoder if training is True, otherwise returns the encoder passed in.
    - LabelBinarizer: Trained LabelBinarizer if training is True, otherwise returns the binarizer passed in.
    """
    if categorical_features is None:
        categorical_features = []

    if continuous_features is None:
        continuous_features = []

    labels = np.array([])
    if label is not None:
        labels = data_frame[label].values
        data_frame = data_frame.drop(label, axis=1)

    if training:
        if encoder is None:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(data_frame[categorical_features])
        if label_binarizer is None:
            label_binarizer = LabelBinarizer()
        if label is not None:
            label_binarizer.fit(labels.reshape(-1, 1))

    categorical_data = encoder.transform(data_frame[categorical_features])
    continuous_data = data_frame[continuous_features].values
    processed_data = np.concatenate([continuous_data, categorical_data], axis=1)

    if label is not None:
        labels = label_binarizer.transform(labels.reshape(-1, 1)).ravel()

    return processed_data, labels, encoder, label_binarizer
