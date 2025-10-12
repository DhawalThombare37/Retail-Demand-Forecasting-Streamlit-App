import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error

st.title("Retail Demand Forecasting App (Transformer + XGBoost)")

# --- Load models and scaler ---
@st.cache(allow_output_mutation=True)
def load_models():
    transformer_model = tf.keras.models.load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    sequence_length = joblib.load("sequence_length.pkl")
    return transformer_model, xgb_model, scaler, training_columns, sequence_length

transformer_model, xgb_model, scaler, training_columns, sequence_length = load_models()

# --- File uploader ---
uploaded_file = st.file_uploader("Upload retail_store_inventory CSV", type=["csv"])
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    # --- Wrapper class for preprocessing + prediction ---
    class TransformerXGBPredictor:
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

            # Lag & rolling features
            lag_period = 7
            rolling_window = 7
            for col in ['Inventory Level','Units Sold','Units Ordered','Demand Forecast','Price']:
                if 'Store ID' in df.columns and 'Product ID' in df.columns:
                    df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID','Product ID'])[col].shift(lag_period)
                    df[f'{col}_roll_mean_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(rolling_window).mean().reset_index(drop=True)
                    df[f'{col}_roll_std_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(rolling_window).std().reset_index(drop=True)
                else:
                    df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)
                    df[f'{col}_roll_mean_{rolling_window}'] = df[col].rolling(rolling_window).mean().reset_index(drop=True)
                    df[f'{col}_roll_std_{rolling_window}'] = df[col].rolling(rolling_window).std().reset_index(drop=True)

            df = df.fillna(0)

            # One-hot encoding
            categorical_cols = ['Discount','Holiday/Promotion']
            df_processed = pd.get_dummies(df, columns=categorical_cols)
            return df_processed

        def predict(self, df):
            df_processed = self.preprocess(df)

            # --- Transformer sequences ---
            features = [col for col in df_processed.columns if col not in ['Date','Store ID','Product ID','Category','Region','Weather Condition','Seasonality','Demand Forecast']]
            X_scaled = scaler.transform(df_processed[features])
            X_seq = []
            for i in range(len(X_scaled)-self.sequence_length+1):
                X_seq.append(X_scaled[i:i+self.sequence_length])
            X_seq = np.array(X_seq)

            # Transformer predictions
            transformer_preds = self.transformer_model.predict(X_seq).flatten()
            df_aligned = df_processed.iloc[self.sequence_length-1:].reset_index(drop=True)
            df_aligned['transformer_preds'] = transformer_preds

            # --- XGB preparation ---
            for col in self.training_columns:
                if col not in df_aligned.columns:
                    df_aligned[col] = 0
            X_xgb = df_aligned[self.training_columns]

            # Final XGB predictions
            final_preds = self.xgb_model.predict(X_xgb)
            df_aligned['Predicted Demand'] = final_preds

            # MAPE
            y_true = df_aligned['Demand Forecast'].values
            epsilon = 1e-8
            y_true_safe = np.where(y_true==0, epsilon, y_true)
            mape = mean_absolute_percentage_error(y_true_safe, final_preds)*100

            return df_aligned, mape

    predictor = TransformerXGBPredictor(transformer_model, xgb_model, scaler, training_columns, sequence_length)

    df_results, mape = predictor.predict(df_input)

    st.write("MAPE:", f"{mape:.2f}%")
    st.dataframe(df_results)
