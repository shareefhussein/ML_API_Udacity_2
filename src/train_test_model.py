"""
Code to train the model

"""

from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump
from src.process_data import process_data
from src.model import train_model


def train_test_model():
    """
    this function trains the model
  
    """
  
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
    
    model = train_model(X_train, y_train)
    
    dump(model, "./models/model.joblib")
    dump(lb,"./models/lb.joblib")
    dump(encoder,"./models/encoder.joblib")
    
    

    
