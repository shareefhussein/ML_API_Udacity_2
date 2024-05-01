"""
Cleaning data code
"""

import pandas as pd

def clean_data():
    """
    Clean data function to use it in model training.
    """
    data_frame = pd.read_csv("./data/raw_data/census.csv")
    data_frame.columns = data_frame.columns.str.strip()
    data_frame = data_frame.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    data_frame.replace({'?': None}, inplace=True)
    data_frame.dropna(inplace=True)

    # Drop unwanted columns
    columns_to_drop = ["fnlgt", "education-num", "capital-gain", "capital-loss"]
    data_frame.drop(columns=columns_to_drop, axis="columns", inplace=True)

    print("Export cleaned data to CSV file.")
    data_frame.to_csv("./data/cleaned_data/census.csv", index=False)
