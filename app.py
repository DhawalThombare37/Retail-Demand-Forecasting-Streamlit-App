import pandas as pd
import numpy as np
import streamlit as st
import pickle
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# --- Load artifacts ---
@st.cache_data
def load_artifacts():
    transformer_model = load_model("transformer_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("training_info.pkl", "rb") as f:
        info = pickle.load(f)
    training_columns = info["training_columns"]
    sequence_length = info["sequence_length"]
    return transformer_model, scaler, xgb_model, training_columns, sequence_length

transformer_model, scaler, xgb_model, training_columns, sequence_length = load_artifacts()

# --- Helper functions ---
def create_lags_rolls(df):
    df = df.sort_values("Date").reset_index(drop=True)
    lag_period = 7
    rolling_window = 7
    for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
        df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)
        df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
        df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
    df = df.fillna(0)
    return df

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = create_lags_rolls(df)
    # Select features and one-hot encode
    features_to_use = [col for col in df.columns if col not in ['Date','Demand Forecast','Store ID','Product ID','Category','Region','Weather Condition','Seasonality']]
    df_proc = pd.get_dummies(df[features_to_use], columns=['Discount','Holiday/Promotion'])
    # Align columns with training
    for col in training_columns:
        if col not in df_proc.columns:
            df_proc[col] = 0
    df_proc = df_proc[training_columns]
    X_scaled = scaler.transform(df_proc)
    # Create sequences
    X_seq = []
    for i in range(len(X_scaled) - sequence_length + 1):
        X_seq.append(X_scaled[i:i+sequence_length])
    return np.array(X_seq), df.iloc[sequence_length-1:].reset_index(drop=True)

# --- Streamlit UI ---
st.title("Retail Demand Forecasting (Transformer + XGBoost)")
uploaded_file = st.file_uploader("Upload CSV with required columns", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    if df_input.empty:
        st.warning("Uploaded file is empty!")
    else:
        X_seq, df_aligned = preprocess(df_input)
        if X_seq.size == 0:
            st.warning("Not enough rows to form sequences for prediction.")
        else:
            # Transformer predictions
            transformer_preds = transformer_model.predict(X_seq).flatten()
            # Combine with XGBoost
            df_xgb_input = df_aligned.copy()
            df_xgb_input['transformer_pred'] = transformer_preds
            final_preds = xgb_model.predict(df_xgb_input)
            df_aligned['Predicted Demand'] = final_preds
            st.write(df_aligned[['Date','Store ID','Product ID','Category','Region','Predicted Demand']])
            
            # Calculate MAPE if Demand Forecast exists
            if 'Demand Forecast' in df_aligned.columns:
                epsilon = 1e-8
                y_true = df_aligned['Demand Forecast'].replace(0, epsilon)
                mape = mean_absolute_percentage_error(y_true, final_preds)
                st.success(f"MAPE on uploaded data: {mape:.2%}")
