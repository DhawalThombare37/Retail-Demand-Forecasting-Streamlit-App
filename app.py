import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_percentage_error

import joblib
from tensorflow.keras.models import load_model
import streamlit as st

@st.cache_resource
def load_models():
    # Load Transformer model
    transformer_model = load_model("transformer_model.keras")
    
    # Load XGBoost model
    xgb_model = joblib.load("xgb_model.pkl")
    
    # Load scaler
    scaler = joblib.load("scaler.pkl")
    
    # Load training columns and ensure it's a list
    training_columns = joblib.load("training_info.pkl")
    training_columns = list(training_columns)
    
    # Sequence length used in training
    sequence_length = 7
    
    return transformer_model, xgb_model, scaler, training_columns, sequence_length


transformer_model, xgb_model, scaler, training_columns, sequence_length = load_models()


class TransformerPredictor:
    def __init__(self, transformer_model, xgb_model, scaler, training_columns, sequence_length):
        self.transformer_model = transformer_model
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.training_columns = training_columns
        self.sequence_length = sequence_length

    def preprocess(self, df):
        # Ensure Date is datetime
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
            if 'Store ID' in df.columns and 'Product ID' in df.columns:
                df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)
            else:
                df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)

        # Rolling window features
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            if 'Store ID' in df.columns and 'Product ID' in df.columns:
                df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
            else:
                df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(window=rolling_window).std().reset_index(drop=True)

        df = df.fillna(0)

        # Select features to process
        features_to_process = [c for c in df.columns if c not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        df_processed = pd.get_dummies(df[features_to_process], columns=['Discount', 'Holiday/Promotion'])

        # Align columns exactly as training
        for col in self.training_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.training_columns]

        # Convert all to float for scaler
        df_processed = df_processed.astype(float)

        # Scale features
        X_scaled = self.scaler.transform(df_processed)

        # Create sequences
        X_sequences = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_sequences.append(X_scaled[i:(i + self.sequence_length)])
        
        if not X_sequences:
            return np.array([]), df.iloc[self.sequence_length - 1:]
        else:
            return np.array(X_sequences), df.iloc[self.sequence_length - 1:].reset_index(drop=True)

    def predict(self, df):
        # Preprocess
        X_seq, df_aligned = self.preprocess(df)
        if X_seq.size == 0:
            return np.array([]), df_aligned

        # Transformer predictions
        transformer_preds = self.transformer_model.predict(X_seq)

        # Align Transformer predictions with XGBoost input
        X_xgb = df_aligned.copy()
        X_xgb['transformer_predictions_scaled'] = transformer_preds.flatten()

        # Ensure XGBoost sees same columns as training
        xgb_input_cols = self.xgb_model.get_booster().feature_names
        missing_cols = set(xgb_input_cols) - set(X_xgb.columns)
        for col in missing_cols:
            X_xgb[col] = 0
        X_xgb = X_xgb[xgb_input_cols]

        # XGBoost predictions
        final_preds = self.xgb_model.predict(X_xgb)

        return final_preds, df_aligned
