import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import tempfile
import requests

from tensorflow.keras.models import load_model
import xgboost as xgb

# --------------------------
# Helper: Download files from GitHub raw URLs
# --------------------------
def load_file_from_github(url, suffix=""):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        r = requests.get(url)
        r.raise_for_status()  # Ensure download success
        tmp_file.write(r.content)
        tmp_path = tmp_file.name
    return tmp_path

# --------------------------
# Load all artifacts
# --------------------------
@st.cache_resource
def load_artifacts():
    # Replace these with your GitHub raw URLs
    TRANSFORMER_URL = "https://raw.githubusercontent.com/DhawalThombare37/Retail-Demand-Forecasting-Streamlit-App/main/transformer_model.keras"
    SCALER_URL = "https://raw.githubusercontent.com/DhawalThombare37/Retail-Demand-Forecasting-Streamlit-App/main/scaler.pkl"
    XGB_URL = "https://raw.githubusercontent.com/DhawalThombare37/Retail-Demand-Forecasting-Streamlit-App/main/xgb_model.pkl"
    INFO_URL = "https://raw.githubusercontent.com/DhawalThombare37/Retail-Demand-Forecasting-Streamlit-App/main/training_info.pkl"

    # Transformer model
    transformer_path = load_file_from_github(TRANSFORMER_URL)
    transformer_model = load_model(transformer_path)
    os.remove(transformer_path)

    # Scaler
    scaler_path = load_file_from_github(SCALER_URL)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    os.remove(scaler_path)

    # XGBoost
    xgb_path = load_file_from_github(XGB_URL)
    with open(xgb_path, "rb") as f:
        xgb_model = pickle.load(f)
    os.remove(xgb_path)

    # Training info
    info_path = load_file_from_github(INFO_URL)
    with open(info_path, "rb") as f:
        info = pickle.load(f)
    os.remove(info_path)

    return transformer_model, scaler, xgb_model, info["training_columns"], info["sequence_length"]

transformer_model, scaler, xgb_model, training_columns, sequence_length = load_artifacts()

# --------------------------
# Preprocessing functions
# --------------------------
def create_sequences(X, seq_len):
    sequences = []
    for i in range(len(X) - seq_len):
        sequences.append(X[i:(i + seq_len)])
    return np.array(sequences)

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Feature engineering
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

    lag_period = 7
    rolling_window = 7
    for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
        df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)
        df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(window=rolling_window).mean()
        df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(window=rolling_window).std()

    df = df.fillna(0)

    features_to_use = [c for c in df.columns if c not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
    df_processed = pd.get_dummies(df[features_to_use])

    # Ensure training columns exist and in exact order
    for col in training_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    df_processed = df_processed[training_columns]

    X_scaled = scaler.transform(df_processed)
    X_seq = create_sequences(X_scaled, sequence_length)
    df_seq = df.iloc[sequence_length:].reset_index(drop=True)
    return X_seq, df_seq

def predict(df):
    X_seq, df_seq = preprocess(df)
    if X_seq.size == 0:
        return np.array([]), df_seq

    # Transformer predictions
    transformer_preds = transformer_model.predict(X_seq, verbose=0)

    # XGBoost input
    df_xgb = df_seq.copy()
    df_xgb['transformer_preds'] = transformer_preds

    # Ensure numeric and align columns
    for col in training_columns:
        if col not in df_xgb.columns:
            df_xgb[col] = 0
    df_xgb = df_xgb[training_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    final_preds = xgb_model.predict(df_xgb)
    return final_preds, df_seq

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("🧠 Retail Demand Forecasting Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload CSV for prediction", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset Preview")
    st.dataframe(df_input.head())

    st.info("Running preprocessing and prediction...")
    preds, df_processed = predict(df_input)

    if preds.size > 0:
        df_processed["Predicted_Demand"] = preds
        st.success("✅ Predictions generated successfully!")

        st.write("### 📊 Predictions (Top 20 Rows)")
        st.dataframe(df_processed[["Date", "Predicted_Demand"]].head(20))

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df_processed["Date"], df_processed["Predicted_Demand"], label="Predicted Demand")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Demand")
        ax.grid(True)
        st.pyplot(fig)

        csv = df_processed.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
    else:
        st.warning("⚠️ Not enough data to create valid sequences. Please upload a larger file.")
else:
    st.info("Upload a CSV file with the same structure as training data to generate predictions.")
