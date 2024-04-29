"""
cleaning data code

"""

import numpy as np
import pandas as pd

def clean_data():
    """
    clean data function to use it in model training 
    """
    
    df = pd.read_csv("./data/raw_data/census.csv")
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)
    
    # drop unwanted columns
    
    df.drop("fnlgt", axis="columns", inplace=True)
    df.drop("education-num", axis="columns", inplace=True)
    df.drop("capital-gain", axis="columns", inplace=True)
    df.drop("capital-loss", axis="columns", inplace=True)
    
    print("Export cleaned data to CSV file.")
    df.to_csv("./data/cleaned_data/census.csv",index=False)
    



