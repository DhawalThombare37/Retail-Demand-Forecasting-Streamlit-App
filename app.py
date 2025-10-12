import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error

# --- Load saved models and objects ---
@st.cache_resource
def load_models():
    transformer_model = load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    sequence_length = joblib.load("sequence_length.pkl")
    return transformer_model, xgb_model, scaler, training_columns, sequence_length

transformer_model, xgb_model, scaler, training_columns, sequence_length = load_models()

# --- Wrapper class for preprocessing + prediction ---
class TransformerPredictor:
    def __init__(self, transformer_model, xgb_model, scaler, training_columns, sequence_length):
        self.transformer_model = transformer_model
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.training_columns = training_columns
        self.sequence_length = sequence_length

    def preprocess(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Time-based features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

        # Lag features
        lag_period = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            if 'Store ID' in df.columns and 'Product ID' in df.columns:
                df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID','Product ID'])[col].shift(lag_period)
            else:
                df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)

        # Rolling features
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            if 'Store ID' in df.columns and 'Product ID' in df.columns:
                df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
            else:
                df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(window=rolling_window).std().reset_index(drop=True)

        df = df.fillna(0)

        # One-hot encoding
        features_to_encode = ['Discount', 'Holiday/Promotion']
        df_processed = pd.get_dummies(df.drop(columns=['Date', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality', 'Demand Forecast'], errors='ignore'), columns=features_to_encode)

        # Align columns with training columns
        for col in self.training_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.training_columns]

        # Scale
        X_scaled = self.scaler.transform(df_processed)

        # Create sequences for transformer
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:i+self.sequence_length])
        X_seq = np.array(X_seq)

        df_aligned = df.iloc[self.sequence_length-1:].reset_index(drop=True)
        return X_seq, df_aligned

   def predict(self, df):
    X_seq, df_aligned = self.preprocess(df)
    if X_seq.size == 0:
        return pd.DataFrame(), 0.0

    # Transformer predictions
    transformer_preds = self.transformer_model.predict(X_seq).flatten()

    # Prepare XGB input
    X_xgb = df_aligned.copy()
    X_xgb['transformer_preds'] = transformer_preds

    # Ensure all training columns exist
    for col in self.training_columns:
        if col not in X_xgb.columns:
            X_xgb[col] = 0

    # Reorder columns exactly as in training_columns
    X_xgb = X_xgb[self.training_columns]

    # Final XGB predictions
    final_preds = self.xgb_model.predict(X_xgb)
    df_results = df_aligned.copy()
    df_results['Predicted Demand'] = final_preds

    # Compute MAPE
    y_true = df_results['Demand Forecast'].values
    epsilon = 1e-8
    y_true_safe = np.where(y_true == 0, epsilon, y_true)
    mape = mean_absolute_percentage_error(y_true_safe, final_preds) * 100

    return df_results, mape


# --- Streamlit UI ---
st.title("Retail Demand Forecasting (Transformer + XGBoost)")

uploaded_file = st.file_uploader("Upload retail_store_inventory CSV", type="csv")
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    predictor = TransformerPredictor(transformer_model, xgb_model, scaler, training_columns, sequence_length)
    df_results, mape = predictor.predict(df_input)
    if not df_results.empty:
        st.write(f"MAPE: {mape:.2f}%")
        st.dataframe(df_results[['Date','Store ID','Product ID','Demand Forecast','Predicted Demand']])
    else:
        st.error("Not enough data to make predictions. Please provide a larger dataset.")
