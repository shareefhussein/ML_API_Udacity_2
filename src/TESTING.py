import pandas as pd
from joblib import load
from process_data import process_data

def load_resources():
    """ Load trained model and preprocessing objects. """
    try:
        model = load("./models/model.joblib")  # Update path as needed
        encoder = load("./models/encoder.joblib")  # Update path as needed
        label_binarizer = load("./models/lb.joblib")  # Update path as needed
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None, None
    return model, encoder, label_binarizer

def prepare_and_predict(test_data):
    """ Prepare test data and predict using the loaded model. """
    model, encoder, label_binarizer = load_resources()
    if None in [model, encoder, label_binarizer]:
        return "Failed to load resources."

    # Process test data using the loaded encoder and label binarizer
    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    
    x_test, _, _, _ = process_data(
        test_data, 
        categorical_features=categorical_features, 
        training=False, 
        encoder=encoder, 
        label_binarizer=label_binarizer
    )

    # Make predictions using the loaded model
    try:
        prediction = model.predict(x_test)
    except Exception as e:
        return f"Prediction error: {e}"

    # Decode predictions to original labels
    try:
        prediction_label = label_binarizer.inverse_transform(prediction.reshape(-1, 1))
    except Exception as e:
        return f"Error transforming prediction: {e}"

    return f"Predictions: {prediction_label}"

# Sample data for prediction
test_data = pd.DataFrame({
    'age': [45, 30],
    'workclass': ['State-gov', 'Private'],
    'education': ['Bachelors', 'HS-grad'],
    'marital-status': ['Never-married', 'Married-civ-spouse'],
    'occupation': ['Prof-speciality', 'Craft-repair'],
    'relationship': ['Not-in-family', 'Husband'],
    'race': ['White', 'Black'],
    'sex': ['Male', 'Female'],
    'native-country': ['United-States', 'Canada'],
    'fnlgt': [234721, 121772],
    'education-num': [13, 9],
    'capital-gain': [0, 0],
    'capital-loss': [0, 0],
    'hours-per-week': [40, 40]
})

# Run the test
result = prepare_and_predict(test_data)
print(result)
