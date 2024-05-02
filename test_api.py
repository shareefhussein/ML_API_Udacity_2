from fastapi.testclient import TestClient
from api import app
import numpy as np
from pandas.core.frame import DataFrame


client = TestClient(app)


def test_get_data():
    """
    Test GET request to root endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Greetings from Shareef!!"}


def test_model_inference_valid_input():
    """
    Test POST request with valid
    input data to ensure the API returns correct status and output.
    """
    array = np.array([[
                     32,
                     "Private",
                     "Some-college",
                     "Married-civ-spouse",
                     "Exec-managerial",
                     "Husband",
                     "Black",
                     "Male",
                     80,
                     "United-States"
                     ]])
    client_info = DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    # Convert DataFrame to dictionary
    client_info_dict = client_info.to_dict(orient='records')

    response = client.post("/", json=client_info_dict)
    print(response)


def test_model_inference_invalid_input():
    """
    Test POST request with missing fields
    to ensure the API handles invalid input gracefully.
    """
    client_info = {
        "age": 35,
        "workclass": "Private",
        "relationship": "Husband",
        "education": "Bachelors",
        "native_country": "United-States",
        "race": "White",
        "sex": "Male",
        "occupation": "Exec-managerial",
        "fnlgt": 182238,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 41
    }
    response = client.post("/", json=client_info)
    assert response.status_code == 422  # Expect 422 Unprocessable Entity
