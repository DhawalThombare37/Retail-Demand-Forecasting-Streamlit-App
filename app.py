import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import pickle
import joblib

# --- PAGE SETUP ---
st.set_page_config(page_title="Retail Demand Forecasting (Transformer + XGBoost)", layout="wide")
st.title("🛍️ Retail Demand Forecasting App")
st.markdown("### Powered by Transformer + XGBoost (Final MAPE ≈ 2.88 %)")

# --- LOAD MODELS ---
@st.cache_resource
def load_artifacts():
    transformer = tf.keras.models.load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("training_info.pkl", "rb") as f:
        info = pickle.load(f)
    return transformer, xgb_model, scaler, info

transformer, xgb_model, scaler, info = load_artifacts()
training_columns = info["training_columns"]
sequence_length = info["sequence_length"]

# --- UTILS ---
def preprocess_input(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Time-based features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

    # Lag + rolling features
    lag_period, rolling_window = 7, 7
    for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
        df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)
        df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(rolling_window).mean()
        df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(rolling_window).std()
    df = df.fillna(0)

    # Drop irrelevant + one-hot encode
    feat_cols = [c for c in df.columns if c not in 
        ['Date','Demand Forecast','Store ID','Product ID','Category','Region','Weather Condition','Seasonality']]
    df_proc = pd.get_dummies(df[feat_cols], columns=['Discount','Holiday/Promotion'], dtype=float)

    # Align columns
    for c in training_columns:
        if c not in df_proc.columns:
            df_proc[c] = 0
    df_proc = df_proc[training_columns]

    # Scale + build sequences
    X_scaled = scaler.transform(df_proc)
    X_seq = np.array([X_scaled[i:i+sequence_length] for i in range(len(X_scaled)-sequence_length)])
    return X_seq, df.iloc[sequence_length:].reset_index(drop=True)

def predict(df):
    X_seq, df_aligned = preprocess_input(df)
    if len(X_seq) == 0:
        st.warning("Not enough rows for prediction (need ≥ sequence_length).")
        return None
    transformer_preds = transformer.predict(X_seq, verbose=0).flatten()
    X_xgb = df_aligned.copy()
    X_xgb = pd.get_dummies(X_xgb.drop(['Demand Forecast','Date','Category','Region','Weather Condition','Seasonality'], 
                                      errors='ignore'),
                           columns=['Discount','Holiday/Promotion'], dtype=float)
    for c in training_columns:
        if c not in X_xgb.columns:
            X_xgb[c] = 0
    X_xgb = X_xgb[training_columns]
    X_xgb['transformer_predictions_scaled'] = transformer_preds
    preds = xgb_model.predict(X_xgb)
    return preds, df_aligned['Date'].values[-len(preds):]

# --- UI ---
uploaded = st.file_uploader("Upload CSV (must contain columns like Date, Inventory Level, Units Sold, Units Ordered, Price, etc.)", type=["csv"])

if uploaded:
    input_df = pd.read_csv(uploaded)
    st.write("✅ File loaded. Preview:")
    st.dataframe(input_df.head())

    if st.button("🔮 Predict Demand"):
        with st.spinner("Running Transformer + XGBoost model..."):
            preds, dates = predict(input_df)
            if preds is not None:
                result = pd.DataFrame({"Date": dates, "Predicted Demand": preds})
                st.success("Forecast generated successfully!")
                st.dataframe(result.head(10))
                st.line_chart(result.set_index("Date"))
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")
else:
    st.info("👆 Upload your CSV file to start forecasting.")
