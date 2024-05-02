"""
Module for testing data cleaning functions.
It ensures that the cleaning process outputs files
as expected, correctly strips and formats columns,
 and drops unwanted columns from the dataset.
"""

import os
import pandas as pd
import pytest
from src.cleaning_data import clean_data


@pytest.fixture
def test_cleaned_csv_generated():
    """
    Test that the clean_data function
    generates the expected cleaned CSV file.
    """
    clean_data()
    assert os.path.isfile("./data/cleaned_data/census.csv"), \
        "Cleaned data CSV file is missing"


def test_columns_stripped():
    """
    Test that all columns in the cleaned
     CSV file have no leading or trailing whitespace.
    """
    clean_data()
    data_frame = pd.read_csv("./data/cleaned_data/census.csv")
    assert all(column.strip() == column for column in data_frame.columns), \
        "Columns have trailing whitespace"


def test_unwanted_columns_dropped():
    """
    Test that specific unwanted columns
    are dropped from the cleaned CSV file.
    """
    clean_data()
    data_frame = pd.read_csv("./data/cleaned_data/census.csv")

    assert "fnlgt" not in data_frame.columns, "'fnlgt' column was not dropped"
    assert "education-num" not in data_frame.columns, \
        "'education-num' column was not dropped"
    assert "capital-gain" not in data_frame.columns, \
        "'capital-gain' column was not dropped"
    assert "capital-loss" not in data_frame.columns, \
        "'capital-loss' column was not dropped"
