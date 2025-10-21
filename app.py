import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("Retail Demand Forecasting App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df_full = pd.read_csv(uploaded_file)
    st.success(f"✅ File loaded successfully — Shape: {df_full.shape}")

    # Load models and scaler
    transformer_model = load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    sequence_length = joblib.load("sequence_length.pkl")  # if used in Transformer

    # Preprocessing
    df = df_full.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['weekofyear'] = df['Date'].dt.isocalendar().week

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Discount', 'Holiday/Promotion'], drop_first=False)

    # Lag features (7-day)
    for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
        df[f"{col}_lag_7"] = df[col].shift(7)
        df[f"{col}_rolling_mean_7"] = df[col].rolling(window=7).mean()
        df[f"{col}_rolling_std_7"] = df[col].rolling(window=7).std()

    df.fillna(0, inplace=True)  # Fill NaNs from lags/rolling

    # Transformer input (scaled)
    transformer_features = [col for col in df.columns if col in training_columns]
    df_scaled = pd.DataFrame(scaler.transform(df[transformer_features]), columns=transformer_features)

    # Transformer predictions
    transformer_input = df_scaled.tail(len(df_scaled)).to_numpy().reshape(-1, sequence_length, len(transformer_features))
    transformer_preds = transformer_model.predict(transformer_input, verbose=0)
    transformer_preds_scaled = transformer_preds.flatten()

    # Add Transformer predictions to df
    df_scaled['transformer_predictions_scaled'] = transformer_preds_scaled

    # Align features for XGBoost
    missing_cols = [col for col in training_columns if col not in df_scaled.columns]
    for col in missing_cols:
        df_scaled[col] = 0  # Add missing columns as zeros

    df_scaled = df_scaled[training_columns]  # Ensure same column order

    # XGBoost predictions
    X_xgb = df_scaled.copy()
    final_preds = xgb_model.predict(X_xgb)
    df_full['Predicted_Demand'] = final_preds

    st.subheader("Predictions")
    st.dataframe(df_full)

    # Compute MAPE on test rows only
    if 'is_test' in df_full.columns:
        test_rows = df_full[df_full['is_test'] == 1]
    else:
        N = 73094  # <-- replace with exact number of test rows from Colab
        test_rows = df_full.iloc[-N:]

    if 'Demand Forecast' in test_rows.columns:
        test_mape = mean_absolute_percentage_error(test_rows['Demand Forecast'], test_rows['Predicted_Demand'])
        st.success(f"✅ MAPE on Test Set: {test_mape*100:.2f}%")
    else:
        st.warning("Demand Forecast column not found — cannot compute MAPE for test set.")
