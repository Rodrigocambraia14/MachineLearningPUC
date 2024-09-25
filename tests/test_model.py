import pytest
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

# Load the model and scaler
@pytest.fixture(scope="module")
def load_model_and_scaler():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Test if the model and scaler are loaded correctly
def test_model_loading(load_model_and_scaler):
    model, scaler = load_model_and_scaler
    assert model is not None, "Model not loaded properly"
    assert scaler is not None, "Scaler not loaded properly"

# Test the model performance by evaluating it on a test set
def test_model_performance(load_model_and_scaler):
    model, scaler = load_model_and_scaler
    
    # Here, we use a fixed test set that was saved with known expected behavior
    X_test = np.array([
        [7.4, 0.7, 0.00, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4],  # Example test data
        [7.8, 0.88, 0.00, 2.6, 0.098, 25, 67, 0.9968, 3.20, 0.68, 9.8],
        [11.2, 0.28, 0.56, 1.9, 0.075, 17, 60, 0.9980, 3.16, 0.58, 9.8],
    ])
    y_test = np.array([5, 5, 6])  # Expected labels for the above test data
    
    # Preprocess the test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Ensure the accuracy is above a threshold (e.g., 70%)
    assert accuracy >= 0.70, f"Model accuracy is too low: {accuracy * 100}%"

# Test if the preprocessing (scaling) is applied properly
def test_scaler_behavior(load_model_and_scaler):
    _, scaler = load_model_and_scaler
    
    # Test with a small example to check if scaling is consistent
    X = np.array([
        [8.3, 0.58, 0.20, 2.3, 0.076, 9, 23, 0.9973, 3.40, 0.55, 9.7]
    ])
    
    # Ensure the scaler applies transformation without errors
    X_scaled = scaler.transform(X)
    
    # The scaler should scale values properly (check if the output is transformed correctly)
    assert X_scaled is not None, "Scaler did not transform values properly"
    assert X_scaled.shape == (1, 11), "Scaler output shape is incorrect"
