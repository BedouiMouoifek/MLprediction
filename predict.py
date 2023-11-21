import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify ,render_template
import joblib

# Define features and their ranges
features = ['Machine Type', 'Operating Hours', 'Operating Temperature', 'Operating Pressure', 'Vibration Level', 'Current Consumption']

# Generate synthetic data
def generate_synthetic_data(num_rows):
    data = []
    for _ in range(num_rows):
        # Generate random values for each feature
        machine_type = np.random.choice(['Pump', 'Turbine', 'Compressor'])
        machine_type_value = machine_type_mapping[machine_type]  # Convert 'Machine Type' to numerical value
        operating_hours = np.random.randint(500, 3000)
        operating_temperature = np.random.randint(60, 90)
        operating_pressure = np.random.randint(80, 120)
        vibration_level = np.random.random() * (0.7 - 0.3) + 0.3
        current_consumption = np.random.randint(6, 14)

        # Generate failure indicator based on machine type and operating hours
        failure_probability = 0.2
        failure = np.random.choice([0, 1], p=[1 - failure_probability, failure_probability])

        # Append data to list
        data.append([machine_type_value, operating_hours, operating_temperature, operating_pressure, vibration_level, current_consumption, failure])

    # Create DataFrame from generated data
    df = pd.DataFrame(data, columns=features + ['Failure'])

    return df

# Create a dictionary to map machine type values to numerical values
machine_type_mapping = {
    'Pump': 0,
    'Turbine': 1,
    'Compressor': 2
}

# Train and evaluate the model
def train_and_evaluate_model(df):
    X = df[features]
    y = df['Failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    return model

# Generate synthetic data and train the model
df = generate_synthetic_data(10000)
trained_model = train_and_evaluate_model(df)

# Save the entire model using joblib
joblib.dump(trained_model, 'model.pkl')

# Create the Flask application
app = Flask(__name__)
# Define the route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Define the API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
     # Load the saved model
    loaded_model = joblib.load('model.pkl')

    # Extract data from the request body
    data = request.get_json()
    machine_type = data['machineType']
    machine_type_value = machine_type_mapping[machine_type]  # Convert 'Machine Type' to numerical value
    operating_hours = float(data['operatingHours'])
    operating_temperature = float(data['operatingTemperature'])
    operating_pressure = float(data['operatingPressure'])
    vibration_level = float(data['vibrationLevel'])
    current_consumption = float(data['currentConsumption'])

    # Prepare input data for the model
    input_data = np.array([machine_type_value, operating_hours, operating_temperature, operating_pressure, vibration_level, current_consumption])
    input_data = np.array(input_data).reshape(1, -1)
    # Predict the failure rate
    predicted_failure_rate = loaded_model.predict_proba(input_data)[0, 1]

    # Return the predicted failure rate
    return jsonify({'failureRate': predicted_failure_rate})

if __name__ == '__main__':
    app.run(debug=True)

