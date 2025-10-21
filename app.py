import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Retail Demand Forecasting - XGB + Transformer Fusion", layout="wide")

# ----------------------------------------------------------------------
# 1️⃣ Load all saved model artifacts (exact from Colab)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load("xgb_model.pkl")
    transformer_model = tf.keras.models.load_model("transformer_model.keras", compile=False)
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    xgb_columns = joblib.load("xgb_columns.pkl")
    seq_len = joblib.load("sequence_length.pkl")
    return xgb_model, transformer_model, scaler, training_columns, xgb_columns, seq_len

xgb_model, transformer_model, scaler, training_columns, xgb_columns, seq_len = load_artifacts()

st.title("📊 Retail Demand Forecasting (3% MAPE – Colab Synced)")

# ----------------------------------------------------------------------
# 2️⃣ Upload CSV
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload full train/test CSV file", type=["csv"])

if uploaded_file:
    df_full = pd.read_csv(uploaded_file)
    st.success(f"✅ File loaded successfully — Shape: {df_full.shape}")

    # ----------------------------------------------------------------------
    # 3️⃣ Ensure same preprocessing pipeline (match training columns exactly)
    # ----------------------------------------------------------------------
    df_full.columns = df_full.columns.str.strip()
    df_full = df_full[training_columns]  # keep only the same columns as in training

    # Scale using saved scaler
    df_scaled = pd.DataFrame(scaler.transform(df_full), columns=training_columns)

    # ----------------------------------------------------------------------
    # 4️⃣ Sequence preparation (same as Colab Transformer input)
    # ----------------------------------------------------------------------
    def create_sequences(data, seq_len):
        X_seq = []
        for i in range(len(data) - seq_len):
            X_seq.append(data.iloc[i:i+seq_len].values)
        return np.array(X_seq)

    X_seq = create_sequences(df_scaled, seq_len)

    # Predict Transformer embeddings
    transformer_preds = transformer_model.predict(X_seq, verbose=0)
    transformer_preds = np.concatenate([[0]*seq_len, transformer_preds.flatten()])  # align lengths

    # Append transformer predictions as new feature
    df_full["transformer_predictions_scaled"] = transformer_preds

    # ----------------------------------------------------------------------
    # 5️⃣ Align columns for XGBoost model
    # ----------------------------------------------------------------------
    for col in xgb_columns:
        if col not in df_full.columns:
            df_full[col] = 0  # add missing if any

    X_xgb = df_full[xgb_columns]

    # ----------------------------------------------------------------------
    # 6️⃣ Make final predictions
    # ----------------------------------------------------------------------
    preds = xgb_model.predict(X_xgb)
    df_full["Predicted_Demand"] = preds

    # Compute MAPE if actuals exist
    if "Demand Forecast" in df_full.columns:
        mape = mean_absolute_percentage_error(df_full["Demand Forecast"], preds) * 100
        st.success(f"✅ Predictions complete — MAPE: {mape:.2f}%")
    else:
        mape = None
        st.info("Predictions complete (no actual labels found).")

    # ----------------------------------------------------------------------
    # 7️⃣ Display results
    # ----------------------------------------------------------------------
    st.write("### 🔍 Sample Predictions")
    st.dataframe(df_full[["Predicted_Demand"]].head(10))

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df_full["Predicted_Demand"], label="Predicted", color="orange")
    if "Demand Forecast" in df_full.columns:
        ax.plot(df_full["Demand Forecast"], label="Actual", color="green")
    ax.legend()
    st.pyplot(fig)

    # Download option
    csv = df_full.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predicted Results", data=csv, file_name="predictions.csv", mime="text/csv")

else:
    st.info("Please upload your full dataset to begin.")
