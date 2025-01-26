from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load('iris_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True) 

