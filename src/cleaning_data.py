"""
Cleaning data code
"""

import pandas as pd


def clean_data():
    """
    Clean the dataset for use in model training.
    This includes stripping whitespace,
    replacing problematic values, and removing unwanted columns.
    """
    data_frame = pd.read_csv("./data/raw_data/census.csv")
    # Strip whitespace from column names
    data_frame.columns = data_frame.columns.str.strip()
    # Strip whitespace from string
    data_frame = data_frame.applymap(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    data_frame.replace({'?': None}, inplace=True)
    data_frame.dropna(inplace=True)

    # Drop unwanted columns
    columns_to_drop = ["fnlgt", "education-num",
                       "capital-gain", "capital-loss"]
    data_frame.drop(columns=columns_to_drop, inplace=True)

    print("Export cleaned data to CSV file.")
    data_frame.to_csv("./data/cleaned_data/census.csv", index=False)
