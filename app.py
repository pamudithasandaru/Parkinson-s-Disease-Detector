import joblib
import numpy as np
from flask import Flask, request, jsonify
# Flask-CORS is necessary to allow your index.html (running in a browser) 
# to talk to this Python server (running on a different port/origin).
from flask_cors import CORS 

# --- Initialization ---
app = Flask(__name__)
# Enable CORS for all routes
CORS(app) 

# --- Load Model and Scaler ---
# These variables will hold the trained model and the fitted StandardScaler
model = None
scaler = None

try:
    # Load the trained model (SVM) and the fitted scaler
    # NOTE: Ensure these file names match the output of your notebook's saving step!
    model = joblib.load('parkinsons_model.pkl')
    scaler = joblib.load('parkinsons_scaler.pkl')
    print("Model and Scaler loaded successfully. Server ready.")
except FileNotFoundError:
    print("FATAL ERROR: Model files (parkinsons_model.pkl or parkinsons_scaler.pkl) not found.")
    print("Please ensure you ran the model saving step in your Jupyter notebook.")
    # In a real app, you might stop the server here, but we let it run 
    # for debugging so you see the API endpoint.

# --- API Endpoint ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests from the web UI.
    1. Reads 22 features from the JSON request body.
    2. Uses the loaded scaler to standardize the data.
    3. Uses the loaded model to make a prediction.
    4. Returns the result (0 or 1) and a message to the frontend.
    """
    # Check if model/scaler loaded successfully
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded on server. Check server console for file errors.'
        }), 500
        
    try:
        # 1. Get data from POST request (force=True handles cases where content-type might be slightly off)
        data = request.get_json(force=True)
        features = data.get('features')
        
        # Input validation
        if not features or not isinstance(features, list) or len(features) != 22:
            return jsonify({
                'error': 'Input array must contain exactly 22 numerical features under the "features" key.'
            }), 400

        # 2. Convert to numpy array and reshape (1 sample, 22 features)
        # This matches the shape (1, 22) required by the scaler and model
        input_data = np.asarray(features).reshape(1, -1)
        
        # 3. Standardize the data using the loaded scaler
        std_data = scaler.transform(input_data)
        
        # 4. Make prediction
        prediction = model.predict(std_data)
        
        # 5. Prepare result
        # The prediction result is an array, we take the first (and only) element
        result = int(prediction[0])
        
        # Determine the user-friendly message
        if result == 0:
            message = "The model predicts a Healthy result based on the acoustic features."
        else:
            message = "The model predicts the presence of Parkinson's Disease based on the acoustic features."
            
        # Return the final response to the frontend
        response = {
            'prediction': result,
            'message': message
        }
        
        return jsonify(response)

    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"--- API Execution Error ---")
        print(f"Error details: {e}")
        print(f"Received data: {request.get_data()}")
        print(f"---------------------------")
        return jsonify({
            'error': 'An internal server error occurred during prediction.',
            'details': str(e)
        }), 500

# --- Running the Server ---
if __name__ == '__main__':
    # Flask will start a web server listening on port 5000 on all interfaces (0.0.0.0)
    print("=" * 40)
    print("Flask Prediction Server Initialized.")
    print("STATUS: Waiting for connections on http://127.0.0.1:5000/")
    print("PRESS Ctrl+C TO SHUT DOWN THE SERVER.")
    print("=" * 40)
    app.run(host='0.0.0.0', port=5000)
