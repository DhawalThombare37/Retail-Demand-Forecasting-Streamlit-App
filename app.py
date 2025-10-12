import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# --- Load saved artifacts ---
transformer_model = load_model("transformer_model.keras")
xgb_model = joblib.load("xgb_model.pkl")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("training_info.pkl", "rb") as f:
    training_info = pickle.load(f)
    training_columns = training_info["training_columns"]
    sequence_length = training_info["sequence_length"]

# --- Wrapper class for preprocessing & prediction ---
class TransformerPredictor:
    def __init__(self, model, xgb_model, scaler, training_columns, sequence_length):
        self.model = model
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.training_columns = training_columns
        self.sequence_length = sequence_length

    def preprocess(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)

        # Time-based features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

        # Lag features
        lag_period = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)

        # Rolling features
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
            df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)

        df = df.fillna(0)

        # Select features
        features_to_process = [col for col in df.columns if col not in ['Date','Demand Forecast','Store ID','Product ID','Category','Region','Weather Condition','Seasonality']]
        df_processed = pd.get_dummies(df[features_to_process], columns=['Discount','Holiday/Promotion'])

        # Align columns
        for col in self.training_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.training_columns]

        # Scale
        X_scaled = self.scaler.transform(df_processed)

        # Create sequences
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:(i+self.sequence_length)])
        return np.array(X_seq), df.iloc[self.sequence_length-1:].reset_index(drop=True)

    def predict(self, df):
        X_seq, df_aligned = self.preprocess(df)
        if X_seq.size == 0:
            return np.array([]), df_aligned

        # Transformer predictions
        transformer_preds = self.model.predict(X_seq).flatten()

        # Prepare for XGBoost
        df_xgb = df_aligned.copy()
        df_xgb['transformer_preds'] = transformer_preds
        final_preds = self.xgb_model.predict(df_xgb)

        return final_preds, df_aligned

# --- Streamlit UI ---
st.title("Retail Demand Forecasting App (Exact Colab MAPE)")

uploaded_file = st.file_uploader("Upload retail_store_inventory CSV", type=["csv"])
if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    predictor = TransformerPredictor(transformer_model, xgb_model, scaler, training_columns, sequence_length)
    preds, df_aligned = predictor.predict(df_input)

    if preds.size > 0:
        df_result = df_aligned.copy()
        df_result['Predicted Demand'] = preds
        st.write("Predictions (aligned with Colab pipeline):")
        st.dataframe(df_result.head(20))

        # Compute MAPE on uploaded data
        if 'Demand Forecast' in df_result.columns:
            y_true = df_result['Demand Forecast'].values
            mape = mean_absolute_percentage_error(y_true, preds)
            st.success(f"MAPE on uploaded data: {mape:.2%}")
        else:
            st.warning("No 'Demand Forecast' column found in uploaded CSV for MAPE calculation.")
    else:
        st.warning("Not enough data to form sequences for prediction.")
