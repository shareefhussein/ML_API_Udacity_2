"""
cleaning data test

"""

import numpy as np
import pandas as pd
import pytest
from src.cleaning_data import clean_data
import os


@pytest.fixture
def test_cleaned_csv_generated():
    
    clean_data()    
    
    # Check if the cleaned CSV file exists
    assert os.path.isfile("./data/cleaned_data/census.csv")

    
def test_columns_stripped():
    
    clean_data()
    df = pd.read_csv("./data/cleaned_data/census.csv")    
    assert all(col.strip() == col for col in df.columns)

    
def test_unwanted_columns_dropped():
    clean_data()
    
    df = pd.read_csv("./data/cleaned_data/census.csv")
    
    assert "fnlgt" not in df.columns
    assert "education-num" not in df.columns
    assert "capital-gain" not in df.columns
    assert "capital-loss" not in df.columns

    