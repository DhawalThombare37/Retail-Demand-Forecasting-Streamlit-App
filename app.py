import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error

# -----------------------
# Load models and artifacts
# -----------------------
@st.cache_data
def load_models():
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    sequence_length = joblib.load("sequence_length.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    transformer_model = tf.keras.models.load_model("transformer_model.keras")
    return scaler, training_columns, sequence_length, xgb_model, transformer_model

scaler, training_columns, sequence_length, xgb_model, transformer_model = load_models()

st.title("Retail Demand Forecasting App")

# -----------------------
# Upload CSV
# -----------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df_full = pd.read_csv(uploaded_file)
    st.success(f"✅ File loaded successfully — Shape: {df_full.shape}")

    # -----------------------
    # Align columns to training
    # -----------------------
    missing_cols = set(training_columns) - set(df_full.columns)
    for c in missing_cols:
        df_full[c] = 0  # Add missing columns as zeros
    df_full = df_full[training_columns]

    # -----------------------
    # Scale numeric features
    # -----------------------
    numeric_cols = df_full.select_dtypes(include=np.number).columns.tolist()
    df_full[numeric_cols] = scaler.transform(df_full[numeric_cols])

   # -----------------------
# Transformer predictions with sequence
# -----------------------
def create_sequences(df, seq_length, features):
    data = df[features].values
    X_seq = []
    for i in range(len(data) - seq_length + 1):
        X_seq.append(data[i:i+seq_length])
    return np.array(X_seq)

# Use the numeric/scaled features for sequences
features_for_seq = training_columns  # should match what was used in training
X_seq = create_sequences(df_full, sequence_length, features_for_seq)

# Transformer predicts only on sequences
transformer_preds_seq = transformer_model.predict(X_seq, verbose=0)

# Map transformer predictions back to dataframe
# For the first (sequence_length -1) rows, fill with NaN or replicate first prediction
transformer_preds_full = np.concatenate(
    [np.full((sequence_length-1,), transformer_preds_seq[0]), transformer_preds_seq.flatten()]
)
df_full["Transformer_Pred"] = transformer_preds_full


    # -----------------------
    # XGBoost prediction
    # -----------------------
    X_xgb = df_full.copy()
    final_preds = xgb_model.predict(X_xgb)
    df_full["Predicted_Demand"] = final_preds

    st.write("### Sample Predictions")
    st.dataframe(df_full.tail(10))

    # -----------------------
    # Use only the last 494 rows as Colab test set
    # -----------------------
    test_rows = 494
    if "Demand Forecast" in df_full.columns:
        df_test = df_full.iloc[-test_rows:]
        y_true = df_test["Demand Forecast"].values
        y_pred = df_test["Predicted_Demand"].values
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        st.markdown(f"### ✅ MAPE on Colab test set (last {test_rows} rows): **{mape:.2f}%**")
    else:
        st.info("No 'Demand Forecast' column found — cannot compute MAPE")
