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

    # --- FEATURE ENGINEERING ---
    df_full['Date'] = pd.to_datetime(df_full['Date'])
    df_full['year'] = df_full['Date'].dt.year
    df_full['month'] = df_full['Date'].dt.month
    df_full['day'] = df_full['Date'].dt.day
    df_full['dayofweek'] = df_full['Date'].dt.dayofweek
    df_full['weekofyear'] = df_full['Date'].dt.isocalendar().week.astype(int)

    # Lag & rolling features
    for col in ['Inventory Level','Units Sold','Units Ordered','Demand Forecast','Price']:
        df_full[f'{col}_lag_7'] = df_full[col].shift(7)
        df_full[f'{col}_rolling_mean_7'] = df_full[col].rolling(7).mean()
        df_full[f'{col}_rolling_std_7'] = df_full[col].rolling(7).std()

    # One-hot encoding
    df_full = pd.get_dummies(df_full, columns=['Discount', 'Holiday/Promotion'], drop_first=False)
    df_full.fillna(0, inplace=True)

    # --- ENSURE SCALER COMPATIBILITY ---
    transformer_features = [c for c in training_columns if c not in ['Transformer_Pred']]
    for col in transformer_features:
        if col not in df_full.columns:
            df_full[col] = 0  # add missing column
    df_full = df_full[transformer_features]

    # --- SCALE DATA ---
    df_scaled = pd.DataFrame(scaler.transform(df_full), columns=transformer_features)

    # --- TRANSFORMER INPUT SEQUENCES ---
    transformer_input = []
    for i in range(len(df_scaled) - sequence_length + 1):
        transformer_input.append(df_scaled.iloc[i:i+sequence_length].values)
    transformer_input = np.array(transformer_input)

    transformer_preds = transformer_model.predict(transformer_input, verbose=0)
    transformer_preds_aligned = np.zeros(len(df_full))
    transformer_preds_aligned[sequence_length-1:] = transformer_preds.flatten()
    df_full['Transformer_Pred'] = transformer_preds_aligned

    # --- ALIGN FEATURES FOR XGBOOST ---
    df_xgb = df_full.copy()
    for col in training_columns:
        if col not in df_xgb.columns:
            df_xgb[col] = 0
    df_xgb = df_xgb[training_columns]

    # --- TEST ROWS SAME AS COLAB ---
    test_size = 7300  # exact Colab test rows
    train_size = len(df_xgb) - test_size
    test_indices = list(range(train_size, len(df_xgb)))
    df_test = df_xgb.iloc[test_indices]
    y_true = df_full['Demand Forecast'].iloc[test_indices].values

    # --- PREDICT XGBOOST ---
    final_preds = xgb_model.predict(df_test)

    # --- TEST MAPE ONLY ---
    mape_test = mean_absolute_percentage_error(y_true, final_preds) * 100

    # --- DISPLAY RESULTS ---
    df_results = df_full.copy()
    df_results['Predicted_Demand'] = 0
    df_results.iloc[test_indices, df_results.columns.get_loc('Predicted_Demand')] = final_preds

    st.subheader("Predictions on Test Rows")
    st.dataframe(df_results.iloc[test_indices][['Date', 'Store ID', 'Product ID', 'Demand Forecast', 'Predicted_Demand']])
    st.markdown(f"### ✅ MAPE on Test Data: {mape_test:.2f}%")
