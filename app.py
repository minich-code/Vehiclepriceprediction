import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np 
import pandas as pd  



app = Flask(__name__)

# Load the trained machine learning model 
model = pickle.load(open('best_xgb_model.pkl', 'rb'))

# Load the scaler used to preprocess data
scalar = pickle.load(open('scaler.pkl', 'rb'))

# Define the route for the home page
@app.route('/')
def home():
    # Render the home.html template
    return render_template('home.html')

# Define the route to handle predictions
@app.route('/predict_api', methods=['POST'])

def predict_api():
    # Extract data from the JSON request
    data = request.json['data']

    # Print the received data and its shape(optional)
    print(data)
    
    # Transform the input data using the scaler(optional)
    print(np.array(list(data.values())).reshape(1, -1))

    # Transform the input data using the scaler
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))

    # Make predictions using the loaded model
    prediction = model.predict(new_data)
    # print predictions (optional)
    print(prediction[0])

    # Return the prediction as JSON
    return jsonify(prediction[0])

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)