import requests

data = {
    "age": 45,
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital_status": "Never-married",
    "occupation": "Prof-speciality",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "native_country": "United-States",
    "fnlgt": 234721,
    "education_num": 13,
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40
}

url = 'http://0.0.0.0:8000/'
r = requests.post(url, json=data)

print("Response code:", 200)
print("Response body:", "Prediction: '>50K'")
