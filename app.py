import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# ------------------- LOAD SAVED ARTIFACTS -------------------
st.title("Retail Demand Forecasting App")

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

# ------------------- HELPER FUNCTIONS -------------------
def preprocess(df, training_columns, scaler, sequence_length):
    df = df.copy()
    
    # Ensure Date
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
        df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period).fillna(0)
    
    # Rolling window features
    rolling_window = 7
    for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
        df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True).fillna(0)
        df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True).fillna(0)
    
    # Select features to process
    features_to_process = [c for c in df.columns if c not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
    
    # One-hot encoding for categorical
    df_processed = pd.get_dummies(df[features_to_process], columns=['Discount', 'Holiday/Promotion'])
    
    # Align columns with training columns
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

def predict(df_input):
    X_seq, df_aligned = preprocess(df_input, training_columns, scaler, sequence_length)
    
    if X_seq.size == 0:
        return np.array([]), df_aligned
    
    # Transformer predictions
    transformer_preds = transformer_model.predict(X_seq).flatten()
    
    # Prepare XGBoost input
    df_xgb_input = df_aligned.copy()
    df_xgb_input['transformer_pred'] = transformer_preds
    
    xgb_features = xgb_model.get_booster().feature_names
    for col in xgb_features:
        if col not in df_xgb_input.columns:
            df_xgb_input[col] = 0
    df_xgb_input = df_xgb_input[xgb_features]
    
    # XGBoost predictions
    final_preds = xgb_model.predict(df_xgb_input)
    
    return final_preds, df_aligned

# ------------------- STREAMLIT APP -------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write("Uploaded data shape:", df_input.shape)
    
    predictions, df_aligned = predict(df_input)
    
    if predictions.size == 0:
        st.error("Not enough rows to form sequences for Transformer model.")
    else:
        df_output = df_aligned[['Date', 'Store ID', 'Product ID', 'Category', 'Region']].copy()
        df_output['Predicted Demand'] = predictions
        st.write("Predictions for your data:")
        st.dataframe(df_output.head(20))
        
        # Optional: Compute MAPE if 'Demand Forecast' exists
        if 'Demand Forecast' in df_aligned.columns:
            y_true = df_aligned['Demand Forecast'].values
            mape = mean_absolute_percentage_error(y_true, predictions)
            st.success(f"MAPE on uploaded data: {mape:.2%}")
