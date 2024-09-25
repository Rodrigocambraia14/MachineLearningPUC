from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from flasgger import Swagger, swag_from

app = Flask(__name__)

# Enable CORS for the entire Flask app
CORS(app)

# Initialize Swagger
swagger = Swagger(app)

# Load the trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
@swag_from({
    'tags': ['Prediction'],
    'description': 'Make a prediction based on features provided',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'description': 'Array of features to be predicted',
            'schema': {
                'type': 'object',
                'properties': {
                    'features': {
                        'type': 'array',
                        'items': {
                            'type': 'number'
                        },
                        'example': [7.88, 1.44, 0.81, 9.81, 0.47, 29, 44, 0.99208, 2.83, 1.31, 8.55]
                    }
                },
                'required': ['features']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Prediction result',
            'schema': {
                'type': 'object',
                'properties': {
                    'prediction': {
                        'type': 'integer',
                        'example': 5
                    }
                }
            }
        },
        '400': {
            'description': 'Bad request or prediction error',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string',
                        'example': 'Error message explaining what went wrong'
                    }
                }
            }
        }
    }
})
def predict():
    """Endpoint for making predictions.
    ---
    """
    try:
        # Receive JSON data
        data = request.get_json()

        # Convert features to NumPy array
        features = np.array([data['features']])

        # Scale features using pre-trained scaler
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)

        # Return prediction as JSON
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
