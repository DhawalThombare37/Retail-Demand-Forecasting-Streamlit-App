# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

import joblib
import tensorflow as tf
import pickle

@st.cache_resource
def load_models():
    transformer_model = tf.keras.models.load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("training_columns.pkl", "rb") as f:
        training_columns = pickle.load(f)
    sequence_length = 7  # Same as used in Colab
    return transformer_model, xgb_model, scaler, training_columns, sequence_length


transformer_model, xgb_model, scaler, training_columns, sequence_length = load_models()

# --- Define Predictor Class ---
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

        # Features for model
        features_to_process = [c for c in df.columns if c not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        df_processed = pd.get_dummies(df[features_to_process], columns=['Discount', 'Holiday/Promotion'])

        # Align columns
        for col in self.training_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.training_columns]

        df_processed = df_processed.astype(float)

        # Scale
        X_scaled = self.scaler.transform(df_processed)

        # Create sequences
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:(i + self.sequence_length)])

        if not X_seq:
            return np.array([]), df.iloc[self.sequence_length - 1:]
        else:
            return np.array(X_seq), df.iloc[self.sequence_length - 1:].reset_index(drop=True)

    def predict(self, df):
        X_seq, df_aligned = self.preprocess(df)
        if X_seq.size == 0:
            return np.array([]), df_aligned

        # Transformer predictions
        transformer_preds = self.transformer_model.predict(X_seq)

        # Align with XGBoost
        X_xgb = df_aligned.copy()
        X_xgb['transformer_predictions_scaled'] = transformer_preds.flatten()

        # Ensure same columns for XGBoost
        xgb_input_cols = self.xgb_model.get_booster().feature_names
        for col in set(xgb_input_cols) - set(X_xgb.columns):
            X_xgb[col] = 0
        X_xgb = X_xgb[xgb_input_cols]

        final_preds = self.xgb_model.predict(X_xgb)
        return final_preds, df_aligned

predictor = TransformerPredictor(transformer_model, xgb_model, scaler, training_columns, sequence_length)

# --- Streamlit UI ---
st.title("Retail Demand Forecasting (Transformer + XGBoost)")

uploaded_file = st.file_uploader("Upload retail_store_inventory CSV", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df_input.head())

    with st.spinner("Predicting demand..."):
        preds, df_aligned = predictor.predict(df_input)

    if preds.size > 0:
        # Show predictions
        df_results = df_aligned.copy()
        df_results['Predicted_Demand'] = preds
        st.subheader("Predictions")
        st.dataframe(df_results.head(20))

        # Calculate MAPE if actual demand available
        if 'Demand Forecast' in df_results.columns:
            y_true = df_results['Demand Forecast'].values
            epsilon = 1e-8
            y_true_safe = np.where(y_true == 0, epsilon, y_true)
            mape = mean_absolute_percentage_error(y_true_safe, preds)
            st.success(f"MAPE: {mape:.2%}")

        # Plot Actual vs Predicted
        if 'Demand Forecast' in df_results.columns:
            st.subheader("Actual vs Predicted Demand")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df_results['Demand Forecast'], y=df_results['Predicted_Demand'], ax=ax, alpha=0.6)
            ax.set_xlabel("Actual Demand")
            ax.set_ylabel("Predicted Demand")
            ax.set_title("Actual vs Predicted Demand")
            st.pyplot(fig)
    else:
        st.warning("Not enough data to create sequences for prediction. Please upload a larger CSV.")
