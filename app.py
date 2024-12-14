import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = load_model('lstm_model.h5')  # Replace with your model file
scaler = joblib.load('scaler.pkl')


# Load historical data (2000-01 to 2020-12)
historical_data = pd.read_csv('history.csv')
historical_data['Date'] = pd.to_datetime(historical_data['Date'])
historical_data.set_index('Date', inplace=True)

# Define look_back
look_back = 12

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        try:
            year = int(data.get('year'))
            month = int(data.get('month'))
        except (ValueError, TypeError):
            return jsonify({"error": "Year and month must be integers."}), 400

        # Validate year and month
        if year < 2000:
            return jsonify({"error": "Data before 2000-01 is not available."}), 400
        if month < 1 or month > 12:
            return jsonify({"error": "Invalid month. Month must be between 1 and 12."}), 400

        query_date = pd.Timestamp(year=year, month=month, day=1)

        # Case 1: Query for historical data
        if query_date <= pd.Timestamp('2020-12-01'):
            if query_date in historical_data.index:
                value = historical_data.loc[query_date]["Alkoholunfälle"]
                return jsonify({"prediction": int(value)})
            else:
                return jsonify({"error": "Historical data for the given date is not available."}), 400

        # Case 2: Query for future prediction (post-2020-12)
        else:
            # Prepare recent data for prediction
            recent_data = historical_data.tail(look_back).values  # Get the last `look_back` rows
            scaled_recent_data = scaler.transform(recent_data)

            # Iteratively predict until the required month
            forecast_date = pd.Timestamp('2021-01-01')
            predictions = []

            while forecast_date <= query_date:
                # Prepare input
                input_data = scaled_recent_data[-look_back:].reshape((1, look_back, recent_data.shape[1]))

                # Predict next step
                prediction_scaled = model.predict(input_data)
                prediction = scaler.inverse_transform(prediction_scaled)[0]

                # Update predictions and recent data
                predictions.append({"prediction": int(prediction[0])})
                scaled_recent_data = np.vstack([scaled_recent_data, prediction_scaled])
                forecast_date += pd.DateOffset(months=1)

            # Return the requested prediction
            return jsonify(predictions[-1])

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Get the PORT from environment variables or default to 8080
    port = int(os.environ.get("PORT", 8080))
    # Listen on all available network interfaces
    app.run(host="0.0.0.0", port=port)