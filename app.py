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


# --- Predictor class ---
class TransformerXGBPredictor:
    def __init__(self, transformer_model, xgb_model, scaler, training_columns, sequence_length):
        self.transformer_model = transformer_model
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.training_columns = training_columns
        self.sequence_length = sequence_length

    def preprocess(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)

        # Time features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

        # Lag and rolling
        lag_period = 7
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            if 'Store ID' in df.columns and 'Product ID' in df.columns:
                df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID','Product ID'])[col].shift(lag_period)
                df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(rolling_window).std().reset_index(drop=True)
            else:
                df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)
                df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(rolling_window).std().reset_index(drop=True)

        df = df.fillna(0)

        # Features & one-hot
        features_to_process = [c for c in df.columns if c not in ['Date','Demand Forecast','Store ID','Product ID','Category','Region','Weather Condition','Seasonality']]
        df_processed = pd.get_dummies(df[features_to_process], columns=['Discount','Holiday/Promotion'])

        # Ensure all training columns are present
        for col in self.training_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.training_columns]

        # Scale
        X_scaled = self.scaler.transform(df_processed)

        # Sequences for Transformer
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:(i + self.sequence_length)])
        if not X_seq:
            return np.array([]), df.iloc[self.sequence_length-1:]
        return np.array(X_seq), df.iloc[self.sequence_length-1:].reset_index(drop=True)

    def predict(self, df):
        X_seq, original_df = self.preprocess(df)
        if X_seq.size == 0:
            return np.array([]), original_df

        # Transformer predictions
        transformer_preds = self.transformer_model.predict(X_seq).flatten()

        # Prepare XGBoost input
        df_xgb = original_df.copy()
        df_xgb['transformer_preds'] = transformer_preds

        # Ensure numeric
        for col in df_xgb.columns:
            df_xgb[col] = pd.to_numeric(df_xgb[col], errors='coerce').fillna(0)

        # Align columns exactly as XGBoost training
        missing_cols = [col for col in self.xgb_model.get_booster().feature_names if col not in df_xgb.columns]
        for col in missing_cols:
            df_xgb[col] = 0
        df_xgb = df_xgb[self.xgb_model.get_booster().feature_names]

        # Final XGBoost predictions
        final_preds = self.xgb_model.predict(df_xgb)

        return final_preds, original_df


# --- Streamlit UI ---
st.title("Retail Demand Forecasting App (Exact Colab MAPE)")
st.write("Upload `retail_store_inventory.csv` to get Transformer + XGBoost predictions with the same accuracy as Colab (~3% MAPE).")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.dataframe(df_input.head())

    predictor = TransformerXGBPredictor(transformer_model, xgb_model, scaler, training_columns, sequence_length)
    preds, df_aligned = predictor.predict(df_input)

    if preds.size > 0:
        epsilon = 1e-8
        mape = mean_absolute_percentage_error(df_aligned['Demand Forecast'].replace(0, epsilon), preds)
        st.write(f"MAPE: {mape:.2%}")
        st.write("Predictions preview:")
        df_result = df_aligned.copy()
        df_result['Predicted_Demand'] = preds
        st.dataframe(df_result.head())
    else:
        st.write("Could not generate predictions. Check if your CSV has enough rows for sequence length.")
