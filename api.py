import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from src.process_data import process_data
from src.model import inference
import os

class Client(BaseModel):
    age: int
    workclass: str
    relationship: str
    education: str
    native_country: str
    race: str
    sex: str
    marital_status: str 
    occupation: str
    fnlgt: int
    capital_gain: int
    capital_loss: int 
    hours_per_week: int
        
    class Config:
        schema_extra = {
            "example": {
                'age': 45,
                'workclass': 'State-gov',
                'fnlgt': 2334,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Never-married',
                'occupation': 'Prof-speciality',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 2174,
                'capital-loss': 0,
                'hours-per-week': 60,
                'native-country': 'Jordan'
            }
        }

        
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
    
    
app = FastAPI()

@app.get("/")
async def get_data():
    return {"message": "Greetings from Shareef!!"}

@app.post("/")
async def model_inference(client_info: Client):
    model = load("./models/model.joblib")
    encoder = load("./models/encoder.joblib")
    lb = load("./models/lb.joblib")

    # Define categorical features
    cat_features = [
        "workclass",
        "relationship",
        "education",
        "native_country",
        "race",
        "sex",
        "marital_status",
        "occupation"
    ]

    client_df = pd.DataFrame(
        data=[    
            [
                client_info.age,
                client_info.workclass,
                client_info.relationship,
                client_info.education,
                client_info.native_country,
                client_info.race,
                client_info.sex,
                client_info.marital_status, 
                client_info.occupation,
                client_info.fnlgt,
                client_info.capital_gain,
                client_info.capital_loss, 
                client_info.hours_per_week
            ]
        ],
        columns=[
            'age',
            'workclass',
            'relationship',
            'education',
            'native_country',
            'race',
            'sex',
            'marital_status', 
            'occupation',
            'fnlgt',
            'capital_gain',
            'capital_loss',
            'hours_per_week'
        ]
    )
    
    X, _, _, _ = process_data(
        client_df,
        categorical_features=cat_features,
        encoder=encoder, lb=lb, training=False
    )
    
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    return {"Predictions": y}


