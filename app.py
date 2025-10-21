import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error

# -----------------------------
# Load models and preprocessing
# -----------------------------
@st.cache_resource
def load_models():
    transformer_model = tf.keras.models.load_model("transformer_model.keras")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    sequence_length = joblib.load("sequence_length.pkl")
    return transformer_model, xgb_model, scaler, training_columns, sequence_length


transformer_model, xgb_model, scaler, training_columns, sequence_length = load_models()

st.title("🧠 Retail Demand Forecasting (Transformer + XGBoost)")
st.caption("Upload your retail dataset — same structure as training CSV — to reproduce Colab 3% MAPE performance.")

uploaded_file = st.file_uploader("📂 Upload your retail_store_inventory.csv", type=["csv"])

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_input(df: pd.DataFrame, training_columns):
    df = df.copy()

    # Fill or encode categorical columns if present
    cat_cols = ["Category", "Region", "Weather Condition", "Holiday/Promotion", "Seasonality"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
        else:
            df[c] = "Unknown"

    # Convert Date to datetime and numeric if applicable
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Weekday"] = df["Date"].dt.weekday
        df.drop(columns=["Date"], inplace=True, errors="ignore")

    # One-hot encode categoricals
    df_processed = pd.get_dummies(df, drop_first=True)

    # Align columns with training columns
    for col in training_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Drop unknown extras
    df_processed = df_processed[training_columns]

    return df_processed


# -----------------------------
# Prediction function
# -----------------------------
def predict_mape(df_input):
    # Preprocess uploaded data
    df_processed = preprocess_input(df_input, training_columns)

    # Apply scaler
    X_scaled = scaler.transform(df_processed)

    # Use only last N rows as per test size (to match Colab)
    N = len(X_scaled)
    if N > 500:
        X_scaled = X_scaled[-500:]  # assuming test size similar to Colab

    # Create sequential inputs for Transformer
    X_seq = []
    for i in range(len(X_scaled) - sequence_length + 1):
        X_seq.append(X_scaled[i:i + sequence_length])
    X_seq = np.array(X_seq)

    # Transformer inference
    transformer_preds = transformer_model.predict(X_seq, verbose=0)
    transformer_preds = transformer_preds.flatten()

    # Align with main dataframe for XGB
    df_aligned = df_processed.iloc[sequence_length - 1:].copy()
    df_aligned["Transformer_Pred"] = transformer_preds

    # XGBoost inference
    xgb_preds = xgb_model.predict(df_aligned)

    # Final blended prediction (same ratio used in Colab)
    final_preds = 0.6 * transformer_preds[-len(xgb_preds):] + 0.4 * xgb_preds

    # Calculate MAPE (assume actuals available)
    if "Units Sold" in df_input.columns:
        y_true = df_input["Units Sold"].iloc[-len(final_preds):].values
        mape = mean_absolute_percentage_error(y_true, final_preds) * 100
    else:
        mape = None

    results = df_input.iloc[-len(final_preds):].copy()
    results["Predicted_Units_Sold"] = final_preds

    return results, mape


# -----------------------------
# Run app
# -----------------------------
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded data preview:", df_input.head())

    try:
        df_results, mape = predict_mape(df_input)
        st.success("✅ Prediction complete!")

        if mape is not None:
            st.metric(label="MAPE (Model Accuracy)", value=f"{mape:.2f}%")
            if mape < 5:
                st.balloons()
        else:
            st.info("Actual 'Units Sold' not found — MAPE cannot be computed.")

        st.dataframe(df_results.tail(20))

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("👆 Please upload your CSV to begin.")
