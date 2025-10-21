import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Retail Demand Forecasting (XGB + Transformer)", layout="wide")

# ----------------------------------------------------------------------
# 1️⃣ Load saved models and preprocessing objects
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load("xgb_model.pkl")  # ✅ your filename
    transformer_model = tf.keras.models.load_model("transformer_model.keras", compile=False)
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    xgb_columns = joblib.load("xgb_columns.pkl")
    seq_len = joblib.load("sequence_length.pkl")
    return xgb_model, transformer_model, scaler, training_columns, xgb_columns, seq_len

xgb_model, transformer_model, scaler, training_columns, xgb_columns, seq_len = load_artifacts()

st.title("📊 Retail Demand Forecasting – Colab Synced (≈3% MAPE)")

# ----------------------------------------------------------------------
# 2️⃣ Upload user dataset
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload your full train/test CSV file", type=["csv"])

if uploaded_file:
    df_full = pd.read_csv(uploaded_file)
    df_full.columns = df_full.columns.str.strip()  # clean spaces

    st.success(f"✅ File loaded successfully — Shape: {df_full.shape}")

    # ----------------------------------------------------------------------
    # 3️⃣ Align columns with training data
    # ----------------------------------------------------------------------
    missing_cols = [c for c in training_columns if c not in df_full.columns]
    for col in missing_cols:
        df_full[col] = 0  # fill missing with zeros
    df_full = df_full[training_columns]

    # ----------------------------------------------------------------------
    # 4️⃣ Apply saved scaler
    # ----------------------------------------------------------------------
    df_scaled = pd.DataFrame(scaler.transform(df_full), columns=training_columns)

    # ----------------------------------------------------------------------
    # 5️⃣ Prepare Transformer sequence input
    # ----------------------------------------------------------------------
    def create_sequences(data, seq_len):
        X_seq = []
        for i in range(len(data) - seq_len):
            X_seq.append(data.iloc[i:i + seq_len].values)
        return np.array(X_seq)

    X_seq = create_sequences(df_scaled, seq_len)
    transformer_preds = transformer_model.predict(X_seq, verbose=0)
    transformer_preds = np.concatenate([[0] * seq_len, transformer_preds.flatten()])
    df_full["transformer_predictions_scaled"] = transformer_preds

    # ----------------------------------------------------------------------
    # 6️⃣ Align for XGBoost
    # ----------------------------------------------------------------------
    for col in xgb_columns:
        if col not in df_full.columns:
            df_full[col] = 0

    df_full = df_full[xgb_columns]

   # ----------------------------------------------------------------------
# 7️⃣ Predict with XGBoost
# ----------------------------------------------------------------------
preds = xgb_model.predict(df_full)
df_full["Predicted_Demand"] = preds

# ----------------------------------------------------------------------
# 8️⃣ Compute MAPE on original test rows
# ----------------------------------------------------------------------
# Assuming Colab used last N rows as test (replace N with your exact number)
test_rows = 494  # <-- use the same as Colab
if "Demand Forecast" in df_full.columns:
    y_true = df_full["Demand Forecast"].iloc[-test_rows:]
    y_pred = df_full["Predicted_Demand"].iloc[-test_rows:]
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    st.markdown(f"### ✅ Prediction complete — **MAPE on test rows: {mape:.2f}%**")
else:
    mape = None
    st.info("Predictions generated, but no 'Demand Forecast' column found to compute MAPE.")


    # ----------------------------------------------------------------------
    # 9️⃣ Show results
    # ----------------------------------------------------------------------
    st.write("### 🔍 Sample Predictions")
    st.dataframe(df_full.head(10))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_full["Predicted_Demand"], label="Predicted", color="orange")
    if "Demand Forecast" in df_full.columns:
        ax.plot(df_full["Demand Forecast"], label="Actual", color="green")
    ax.legend()
    st.pyplot(fig)

    # Download results
    csv = df_full.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predicted Results", data=csv, file_name="predictions.csv", mime="text/csv")

else:
    st.info("Please upload your full dataset to begin.")
