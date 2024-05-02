import os
import sys
import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from src.model import inference
from src.process_data import process_data

app = FastAPI()


class Client(BaseModel):
    age: int
    workclass: str
    relationship: str
    education: str
    education_num: int
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
                'education_num': 13,
                'marital_status': 'Never-married',
                'occupation': 'Prof-speciality',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital_gain': 2174,
                'capital_loss': 0,
                'hours_per_week': 60,
                'native_country': 'United States'
            }
        }


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        sys.exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


@app.get("/")
async def get_data():
    return {"message": "Greetings from Shareef!!"}


@app.post("/")
async def model_inference(client_info: Client):
    try:
        model = load("./models/model.joblib")
        encoder = load("./models/encoder.joblib")
        label_binarizer = load("./models/lb.joblib")

        categorical_features = [
            "workclass", "relationship", "education",
            "native_country", "race", "sex",
            "marital_status", "occupation"
        ]

        client_df = pd.DataFrame([client_info.dict().values()],
                                 columns=client_info.dict().keys())

        preprocessed_data, _, _, _ = process_data(
            client_df,
            categorical_features=categorical_features,
            encoder=encoder, label_binarizer=label_binarizer, training=False
        )

        prediction = inference(model, preprocessed_data)
        prediction_label = label_binarizer.inverse_transform(prediction)[0]
        return {"Predictions": prediction_label}
    except Exception as e:
        # Enhanced error message
        return JSONResponse(status_code=500, content={"message": str(e)})
