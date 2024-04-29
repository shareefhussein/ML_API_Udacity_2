from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_get_data():
    """
    Test GET request to root endpoint

    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Greetings!!"}

def test_model_inference_valid_input():
    """
    Test POST request with valid input data

    """
    client_info = {
        "age": 35,
        "workclass": "Private",
        "relationship": "Husband",
        "education": "Bachelors",
        "native_country": "United-States",
        "race": "White",
        "sex": "Male",
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "fnlgt": 182148,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40
    }
    response = client.post("/", json=client_info)
    assert response.status_code == 200
    assert "Predictions" in response.json()

def test_model_inference_invalid_input():
    """
    Test POST request with invalid input data (missing required field)

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

def test_model_inference_missing_model():
    """
    Test POST request when the model file is missing

    """
    client_info = {
        "age": 35,
        "workclass": "Private",
        "relationship": "Husband",
        "education": "Bachelors",
        "native_country": "United-States",
        "race": "White",
        "sex": "Male",
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "fnlgt": 182238,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 41
    }
    response = client.post("/", json=client_info)
    assert response.status_code == 500  # Expect 500 Internal Server Error

def test_model_inference_invalid_model():
    """
    Test POST request with an invalid model file

    """
    client_info = {
        "age": 35,
        "workclass": "Private",
        "relationship": "Husband",
        "education": "Bachelors",
        "native_country": "United-States",
        "race": "White",
        "sex": "Male",
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "fnlgt": 182238,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 41
    }
    response = client.post("/", json=client_info)
    assert response.status_code == 500  # Expect 500 Internal Server Error

def test_model_inference_invalid_output():
    """
    Test POST request with invalid output prediction

    """
    client_info = {
        "age": 35,
        "workclass": "Private",
        "relationship": "Husband",
        "education": "Bachelors",
        "native_country": "United-States",
        "race": "White",
        "sex": "Male",
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "fnlgt": 182238,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 41
    }
    response = client.post("/", json=client_info)
    assert response.status_code == 500 

def test_model_inference_positive_outcome():
    """
    Test POST request with input data expected to result in a positive outcome

    """
    client_info = {
        "age": 35,
        "workclass": "Private",
        "relationship": "Husband",
        "education": "Bachelors",
        "native_country": "United-States",
        "race": "White",
        "sex": "Male",
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "fnlgt": 182238,
        "capital_gain": 5340, 
        "capital_loss": 0,
        "hours_per_week": 40
    }
    response = client.post("/", json=client_info)
    assert response.status_code == 200
    assert "Predictions" in response.json()
    assert response.json()["Predictions"] == ">55K"

def test_model_inference_negative_outcome():
    """
    Test POST request with input data expected to result in a negative outcome
    
    """
    client_info = {
        "age": 25,
        "workclass": "Private",
        "relationship": "Not-in-family",
        "education": "HS-grad",
        "native_country": "United-States",
        "race": "Black",
        "sex": "Female",
        "marital_status": "Never-married",
        "occupation": "Service", 
        "fnlgt": 240000,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 30 
    }
    response = client.post("/", json=client_info)
    assert response.status_code == 200
    assert "Predictions" in response.json()
    assert response.json()["Predictions"] == "<=55K" 

    