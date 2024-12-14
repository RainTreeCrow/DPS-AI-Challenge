### DPS Classic AI Challenge - Accident Forecasting

This project was developed as part of the DPS AI Challenge to forecast the number of accidents per month. The solution processes historical traffic accident data to predict future values for various accident categories.

#### **Files Overview**

- **`AI_Model.ipynb`**: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- **`Dockerfile`**: Defines the Docker image to deploy the app on Google Cloud Run.
- **`MV_Stats.csv`**: Raw dataset containing traffic accident statistics.
- **`history.csv`**: Processed historical data (2000-01 to 2020-12) used for deployed model input.
- **`app.py`**: Flask application for serving predictions via an API.
- **`log.txt`**: Training logs containing loss values for train and validation.
- **`lstm_model.h5`**: Saved LSTM model used for inference.
- **`requirements.txt`**: List of required Python packages and their versions.
- **`scaler.pkl`**: Saved scaler used for preprocessing input data.
- **`visualise.png`**: Visualizations of predictions and actual data for accident category Alkoholunfälle.

#### **Solution Overview**

The project predicts monthly traffic accidents for three categories:
1. **Alkoholunfälle** (Alcohol-related accidents)
2. **Fluchtunfälle** (Hit-and-run accidents)
3. **Verkehrsunfälle** (Total traffic accidents)

The project experimented with various models (e.g., ARIMA, Prophet, VAR). The **LSTM (Long Short-Term Memory)** with multiple accident categories was chosen as the final model, it was trained on historical data and deployed using **Google Cloud Run**.

## **How to Use**

### **1. Setup Environment**
- Install required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

### **2. Run the Flask App Locally**
- Start the Flask server:
  ```bash
  python app.py
  ```
- The app will listen on `http://127.0.0.1:8080`.

### **3. Test the API**
You can make POST requests to the `/predict` endpoint. For example:

```bash
curl -X POST http://127.0.0.1:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"year": 2021, "month": 1}'
```

### **4. Deployment**
The application has been deployed using **Google Cloud Run**. You can test the deployed version with the following request:

```bash
curl -X POST https://dps-ai-challenge-309063161338.europe-west2.run.app/predict \
     -H "Content-Type: application/json" \
     -d '{"year": 2021, "month": 1}'
```

#### **Model Details**

- **Input Data**: Monthly traffic accident statistics (2000-01 to 2020-12).
- **Model**: Multi-variable LSTM model trained with a `look_back` window of 12 months.
- **Loss Function**: Mean Absolute Error (MAE).
- **Validation Performance**:
  - **Alkoholunfälle**: MAE = 6.27
  - **Fluchtunfälle**: MAE = 65.85
  - **Verkehrsunfälle**: MAE = 284.61
- **Test Performance**:
  - **Alkoholunfälle**: MAE = 7.71
  - **Fluchtunfälle**: MAE = 86.09
  - **Verkehrsunfälle**: MAE = 287.08
