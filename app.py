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

    # Load models and supporting files
    transformer_model = load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    sequence_length = joblib.load("sequence_length.pkl")
    test_indices = joblib.load("test_indices.pkl")  # indices of test rows used in Colab

    # Make a copy for processing
    df = df_full.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['weekofyear'] = df['Date'].dt.isocalendar().week

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Discount', 'Holiday/Promotion'], drop_first=False)

    # Lag and rolling features
    for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
        df[f"{col}_lag_7"] = df[col].shift(7)
        df[f"{col}_rolling_mean_7"] = df[col].rolling(window=7).mean()
        df[f"{col}_rolling_std_7"] = df[col].rolling(window=7).std()
    df.fillna(0, inplace=True)

    # Ensure all training columns exist in df
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only training columns and correct order
    df_scaled = pd.DataFrame(scaler.transform(df[training_columns]), columns=training_columns)

    # Prepare transformer input
    transformer_input = df_scaled.to_numpy().reshape(-1, sequence_length, len(training_columns))
    transformer_preds = transformer_model.predict(transformer_input, verbose=0)
    df_scaled['transformer_predictions_scaled'] = transformer_preds.flatten()

    # Prepare XGBoost features
    for col in training_columns:
        if col not in df_scaled.columns:
            df_scaled[col] = 0
    X_xgb = df_scaled[training_columns]

    # Predict with XGBoost
    final_preds = xgb_model.predict(X_xgb)
    df_full['Predicted_Demand'] = final_preds

    st.subheader("Predictions on Uploaded Data")
    st.dataframe(df_full)

    # Compute MAPE only on test rows (from Colab)
    if 'Demand Forecast' in df_full.columns:
        test_df = df_full.iloc[test_indices]  # use the same test rows as Colab
        test_mape = mean_absolute_percentage_error(test_df['Demand Forecast'], test_df['Predicted_Demand'])
        st.success(f"✅ Test Set MAPE: {test_mape*100:.2f}%")
    else:
        st.warning("Demand Forecast column not found — cannot compute test MAPE.")
