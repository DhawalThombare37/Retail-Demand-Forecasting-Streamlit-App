import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error

# -------------------------
# Load Models & Config
# -------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    sequence_length = joblib.load("sequence_length.pkl")
    transformer_model = load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    return scaler, training_columns, sequence_length, transformer_model, xgb_model

scaler, training_columns, sequence_length, transformer_model, xgb_model = load_models()

# -------------------------
# Helper Functions
# -------------------------
def preprocess_and_encode(df):
    # Ensure correct column order
    df = df.copy()
    
    # Extract date features
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['weekofyear'] = df['Date'].dt.isocalendar().week

    # One-hot encoding for categorical columns
    categorical_cols = ['Discount', 'Holiday/Promotion']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Ensure all training columns exist (add missing with zeros)
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[training_columns]  # keep only training columns
    return df

def create_sequences(df, seq_length, features):
    data = df[features].values
    X_seq = []
    for i in range(len(data) - seq_length + 1):
        X_seq.append(data[i:i+seq_length])
    return np.array(X_seq)

# -------------------------
# Streamlit App
# -------------------------
st.title("Retail Demand Forecasting")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.success(f"✅ File loaded successfully — Shape: {df_input.shape}")

    try:
        # Preprocess
        df_full = preprocess_and_encode(df_input)
        
        # Scale features
        df_full_scaled = pd.DataFrame(scaler.transform(df_full), columns=df_full.columns)
        
        # Create sequences for Transformer
        X_seq = create_sequences(df_full_scaled, sequence_length, training_columns)
        transformer_preds_seq = transformer_model.predict(X_seq, verbose=0)
        
        # Map predictions back to dataframe
        transformer_preds_full = np.concatenate(
            [np.full((sequence_length-1,), transformer_preds_seq[0]), transformer_preds_seq.flatten()]
        )
        df_full_scaled['Transformer_Pred'] = transformer_preds_full

        # XGBoost prediction
        X_xgb = df_full_scaled.copy()
        final_preds = xgb_model.predict(X_xgb)

        df_input['Predicted_Demand'] = final_preds
        st.subheader("Predictions")
        st.dataframe(df_input)

        # Compute MAPE
        if 'Demand Forecast' in df_input.columns:
            mape = mean_absolute_percentage_error(df_input['Demand Forecast'], df_input['Predicted_Demand'])
            st.success(f"MAPE on uploaded data: {mape*100:.2f}%")
        else:
            st.warning("Demand Forecast column not found — cannot compute MAPE.")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
