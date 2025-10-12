import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("Retail Store Demand Forecasting")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df_input.head())

    # --- Load saved artifacts ---
    scaler = joblib.load("scaler.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    transformer_model = load_model("transformer_model.keras")
    training_info = joblib.load("training_info.pkl")
    training_columns = training_info["training_columns"]
    sequence_length = training_info["sequence_length"]

    # --- Preprocessing function ---
    def preprocess(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)

        # Time-based features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

        # Lag and rolling features
        lag_period = 7
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            if 'Store ID' in df.columns and 'Product ID' in df.columns:
                df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)
                df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
            else:
                df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)
                df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(window=rolling_window).std().reset_index(drop=True)

        df = df.fillna(0)

        # Select features
        features_to_process = [col for col in df.columns if col not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        df_processed = pd.get_dummies(df[features_to_process], columns=['Discount', 'Holiday/Promotion'])

        # Ensure training columns exist
        for col in training_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[training_columns]

        # Scale
        X_scaled = scaler.transform(df_processed)

        # Create sequences
        X_seq = []
        for i in range(len(X_scaled) - sequence_length + 1):
            X_seq.append(X_scaled[i:i+sequence_length])
        return np.array(X_seq), df.iloc[sequence_length-1:].reset_index(drop=True)

    # --- Run preprocessing ---
    X_seq, df_aligned = preprocess(df_input)

    if X_seq.size == 0:
        st.warning("Not enough data to form sequences. Please upload more rows.")
    else:
        # --- Transformer predictions ---
        transformer_preds = transformer_model.predict(X_seq).flatten()

        # --- XGBoost combination ---
        df_xgb_input = df_aligned.copy()
        df_xgb_input['transformer_pred'] = transformer_preds

        # Ensure XGBoost input columns match
        xgb_features = xgb_model.get_booster().feature_names
        for col in xgb_features:
            if col not in df_xgb_input.columns:
                df_xgb_input[col] = 0
        df_xgb_input = df_xgb_input[xgb_features]

        final_preds = xgb_model.predict(df_xgb_input)

        # --- Display results ---
        df_result = df_aligned[['Date','Store ID','Product ID','Category','Region']].copy()
        df_result['Predicted Demand'] = final_preds
        st.write("Forecasted Demand:", df_result.head(20))

        # --- If Demand Forecast exists, compute MAPE ---
        if 'Demand Forecast' in df_aligned.columns:
            y_true = df_aligned['Demand Forecast'].values
            y_true_safe = np.where(y_true==0, 1e-8, y_true)
            mape = mean_absolute_percentage_error(y_true_safe, final_preds)
            st.success(f"MAPE on uploaded data: {mape:.2%}")
