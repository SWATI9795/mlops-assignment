from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the model (make sure it's in the same directory or update the path)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Debugging: print the data received
        print("Received data:", data)
        
        # Extract features from the input data
        features = data['features']
        
        # Convert features to numpy array for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        
        # Debugging: print the prediction result
        print("Prediction:", prediction)
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        # If there's an error, return a message
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

