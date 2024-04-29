import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from src.process_data import process_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from src.model import train_model, inference

@pytest.fixture
def data():
    """
    Get dataset
    """
    data = pd.read_csv("./data/cleaned_data/census.csv")
    return data

def test_trained_model(data):
    train, _ = train_test_split(data, test_size=0.20)
    
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
    
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary",
        training=True
    )
    
    model = train_model(X_train, y_train)
    assert isinstance(model, GradientBoostingClassifier)

    
    
