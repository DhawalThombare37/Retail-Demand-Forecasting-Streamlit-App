import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error

st.title("Retail Demand Forecasting App")

# --- UPLOAD CSV ---
uploaded_file = st.file_uploader("Upload your full CSV file", type=["csv"])
if uploaded_file:
    df_full = pd.read_csv(uploaded_file)
    st.success(f"✅ File loaded successfully — Shape: {df_full.shape}")

    # --- LOAD MODELS AND ARTIFACTS ---
    transformer_model = load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    sequence_length = joblib.load("sequence_length.pkl")  # if used in transformer

    # --- FEATURE ENGINEERING (same as Colab) ---
    df_full['Date'] = pd.to_datetime(df_full['Date'])
    df_full['year'] = df_full['Date'].dt.year
    df_full['month'] = df_full['Date'].dt.month
    df_full['day'] = df_full['Date'].dt.day
    df_full['dayofweek'] = df_full['Date'].dt.dayofweek
    df_full['weekofyear'] = df_full['Date'].dt.isocalendar().week.astype(int)

    # Lag features
    for col in ['Inventory Level','Units Sold','Units Ordered','Demand Forecast','Price']:
        df_full[f'{col}_lag_7'] = df_full[col].shift(7)
        df_full[f'{col}_rolling_mean_7'] = df_full[col].rolling(7).mean()
        df_full[f'{col}_rolling_std_7'] = df_full[col].rolling(7).std()

    # One-hot encoding for categorical
    df_full = pd.get_dummies(df_full, columns=['Discount', 'Holiday/Promotion'], drop_first=False)

    # Fill any NaNs from lag/rolling
    df_full.fillna(0, inplace=True)

    # --- SCALE TRANSFORMER INPUT ---
    transformer_features = [c for c in df_full.columns if c in training_columns and c not in ['Transformer_Pred']]
    df_scaled = pd.DataFrame(scaler.transform(df_full[transformer_features]), columns=transformer_features)

    # --- TRANSFORMER PREDICTIONS ---
    transformer_input = []
    for i in range(len(df_scaled) - sequence_length + 1):
        transformer_input.append(df_scaled.iloc[i:i+sequence_length].values)
    transformer_input = np.array(transformer_input)
    
    transformer_preds = transformer_model.predict(transformer_input, verbose=0)
    
    # Align transformer predictions with original df
    transformer_preds_aligned = np.zeros(len(df_full))
    transformer_preds_aligned[sequence_length-1:] = transformer_preds.flatten()
    df_full['Transformer_Pred'] = transformer_preds_aligned

    # --- ALIGN FEATURES FOR XGBOOST ---
    # Ensure columns match training columns
    df_aligned = df_full.copy()
    for col in training_columns:
        if col not in df_aligned.columns:
            df_aligned[col] = 0  # add missing column
    df_aligned = df_aligned[training_columns]

    # --- SELECT TEST ROWS SAME AS COLAB ---
    test_size = 7300  # replace with your Colab test size
    train_size = len(df_full) - test_size
    test_indices = list(range(train_size, len(df_full)))
    df_test = df_aligned.iloc[test_indices]
    y_true = df_full['Demand Forecast'].iloc[test_indices].values

    # --- PREDICT WITH XGBOOST ---
    final_preds = xgb_model.predict(df_test)

    # --- CALCULATE TEST MAPE ---
    mape_test = mean_absolute_percentage_error(y_true, final_preds) * 100

    # --- DISPLAY RESULTS ---
    df_results = df_full.copy()
    df_results['Predicted_Demand'] = 0
    df_results.iloc[test_indices, df_results.columns.get_loc('Predicted_Demand')] = final_preds

    st.subheader("Predictions on Test Rows")
    st.dataframe(df_results.iloc[test_indices][['Date', 'Store ID', 'Product ID', 'Demand Forecast', 'Predicted_Demand']])
    st.markdown(f"### ✅ MAPE on Test Data: {mape_test:.2f}%")
