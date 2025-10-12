import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error

# ------------------- LOAD MODELS AND ARTIFACTS -------------------
@st.cache_resource
def load_models():
    transformer_model = load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    with open("scaler.pkl", "rb") as f:
        scaler = joblib.load(f)
    with open("training_info.pkl", "rb") as f:
        info = joblib.load(f)
    training_columns = info["training_columns"]
    sequence_length = info["sequence_length"]
    return transformer_model, xgb_model, scaler, training_columns, sequence_length

transformer_model, xgb_model, scaler, training_columns, sequence_length = load_models()

# ------------------- PREPROCESSING -------------------
def preprocess(df, training_columns, scaler, sequence_length):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Time features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

    # Lag features
    lag_period = 7
    for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
        df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID','Product ID'])[col].shift(lag_period).fillna(0)

    # Rolling window features
    rolling_window = 7
    for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
        df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True).fillna(0)
        df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True).fillna(0)

    # Feature selection
    features_to_use = [c for c in df.columns if c not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
    
    # One-hot encoding (exact as in training)
    df_processed = pd.get_dummies(df[features_to_use], columns=['Discount', 'Holiday/Promotion'])

    # Ensure all training columns exist
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
    X_seq = np.array(X_seq)
    df_aligned = df.iloc[sequence_length-1:].reset_index(drop=True)
    
    return X_seq, df_aligned

# ------------------- PREDICTION -------------------
def predict(df_input):
    X_seq, df_aligned = preprocess(df_input, training_columns, scaler, sequence_length)
    if X_seq.size == 0:
        return np.array([]), df_aligned

    # Transformer predictions
    transformer_preds = transformer_model.predict(X_seq, verbose=0).flatten()

    # XGBoost input
    df_xgb = df_aligned.copy()
    df_xgb['transformer_pred'] = transformer_preds
    xgb_features = xgb_model.get_booster().feature_names
    for col in xgb_features:
        if col not in df_xgb.columns:
            df_xgb[col] = 0
    df_xgb = df_xgb[xgb_features]

    # Final predictions
    final_preds = xgb_model.predict(df_xgb)
    return final_preds, df_aligned

# ------------------- STREAMLIT APP -------------------
st.title("Retail Demand Forecasting")

uploaded_file = st.file_uploader("Upload the original CSV used for training/testing", type=["csv"])
if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.write("Uploaded file shape:", df_input.shape)

    preds, df_aligned = predict(df_input)
    if preds.size == 0:
        st.error("Not enough rows to create sequences for Transformer model.")
    else:
        df_output = df_aligned[['Date','Store ID','Product ID','Category','Region']].copy()
        df_output['Predicted Demand'] = preds
        st.dataframe(df_output.head(20))

        if 'Demand Forecast' in df_aligned.columns:
            mape = mean_absolute_percentage_error(df_aligned['Demand Forecast'].values, preds)
            st.success(f"MAPE on uploaded data: {mape:.2%}")
