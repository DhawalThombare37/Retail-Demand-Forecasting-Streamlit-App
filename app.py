import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error

st.title("Retail Demand Forecasting App (Exact Colab MAPE)")

# --- Load models and artifacts ---
@st.cache_data
def load_models():
    transformer_model = load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("training_columns.pkl", "rb") as f:
        training_columns = pickle.load(f)
    with open("sequence_length.pkl", "rb") as f:
        sequence_length = pickle.load(f)
    return transformer_model, xgb_model, scaler, training_columns, sequence_length

transformer_model, xgb_model, scaler, training_columns, sequence_length = load_models()

# --- Preprocessing & prediction class ---
class TransformerPredictor:
    def __init__(self, transformer_model, xgb_model, scaler, training_columns, sequence_length):
        self.transformer_model = transformer_model
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.training_columns = training_columns
        self.sequence_length = sequence_length

    def preprocess(self, df):
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
        for col in ['Inventory Level','Units Sold','Units Ordered','Demand Forecast','Price']:
            if 'Store ID' in df.columns and 'Product ID' in df.columns:
                df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID','Product ID'])[col].shift(lag_period)
            else:
                df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)

        # Rolling features
        rolling_window = 7
        for col in ['Inventory Level','Units Sold','Units Ordered','Demand Forecast','Price']:
            if 'Store ID' in df.columns and 'Product ID' in df.columns:
                df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
            else:
                df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(window=rolling_window).std().reset_index(drop=True)

        df = df.fillna(0)

        # One-hot encode categorical columns
        df_processed = pd.get_dummies(df, columns=['Discount','Holiday/Promotion'], drop_first=False)

        # Ensure all training columns exist
        for col in self.training_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.training_columns]

        # Scale features
        X_scaled = self.scaler.transform(df_processed)

        # Create sequences
        X_seq = []
        for i in range(len(X_scaled)-self.sequence_length+1):
            X_seq.append(X_scaled[i:i+self.sequence_length])
        return np.array(X_seq), df.iloc[self.sequence_length-1:].reset_index(drop=True)

    def predict(self, df):
    X_seq, df_aligned = self.preprocess(df)
    if X_seq.size == 0:
        st.error("Not enough rows to form sequences. Upload more data.")
        return df, None

    transformer_preds = self.transformer_model.predict(X_seq)
    df_aligned['transformer_preds'] = transformer_preds.flatten()

    # Ensure all training columns exist
    for col in self.training_columns:
        if col not in df_aligned.columns:
            df_aligned[col] = 0
    df_aligned = df_aligned[self.training_columns]

    # Predict with XGBoost
    final_preds = self.xgb_model.predict(df_aligned)
    df_aligned['Predicted Demand'] = final_preds

    # Calculate MAPE
    if 'Demand Forecast' in df.columns:
        epsilon = 1e-8
        y_true = df_aligned['Demand Forecast'].replace(0, epsilon) if 'Demand Forecast' in df_aligned.columns else df['Demand Forecast']
        mape = mean_absolute_percentage_error(y_true, final_preds)
    else:
        mape = None

    return df_aligned, mape


predictor = TransformerPredictor(transformer_model, xgb_model, scaler, training_columns, sequence_length)

# --- Streamlit UI ---
uploaded_file = st.file_uploader("Upload retail_store_inventory CSV", type="csv")
if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    df_results, mape = predictor.predict(df_input)

    st.subheader("Predictions Preview")
    st.dataframe(df_results[['Date','Store ID','Product ID','Predicted Demand']].head(20))

    if mape is not None:
        st.success(f"MAPE on uploaded data: {mape*100:.2f}% (should be ~3% if same data as Colab)")
