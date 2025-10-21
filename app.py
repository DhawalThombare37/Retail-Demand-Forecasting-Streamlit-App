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
st.caption("Upload your retail dataset (same as Colab structure) — model automatically matches the correct test portion.")

uploaded_file = st.file_uploader("📂 Upload your retail_store_inventory.csv", type=["csv"])

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_input(df: pd.DataFrame, training_columns):
    df = df.copy()

    cat_cols = ["Category", "Region", "Weather Condition", "Holiday/Promotion", "Seasonality"]
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "Unknown"
        df[c] = df[c].astype(str)

    # Date processing
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Weekday"] = df["Date"].dt.weekday
        df.drop(columns=["Date"], inplace=True, errors="ignore")

    df_processed = pd.get_dummies(df, drop_first=True)

    # Align with training columns
    for col in training_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0

    df_processed = df_processed[training_columns]

    return df_processed


# -----------------------------
# Prediction pipeline
# -----------------------------
def predict_mape(df_input):
    df_processed = preprocess_input(df_input, training_columns)
    X_scaled = scaler.transform(df_processed)

    # --- Handle full dataset automatically ---
    total_rows = len(df_processed)
    test_size = 500  # same as Colab test length
    if total_rows > test_size + sequence_length:
        X_scaled = X_scaled[-(test_size + sequence_length):]
        df_input = df_input.iloc[-(test_size + sequence_length):].reset_index(drop=True)

    # Build sequential input for Transformer
    X_seq = []
    for i in range(len(X_scaled) - sequence_length + 1):
        X_seq.append(X_scaled[i:i + sequence_length])
    X_seq = np.array(X_seq)

    transformer_preds = transformer_model.predict(X_seq, verbose=0).flatten()

    # Align lengths
    df_aligned = df_input.iloc[sequence_length - 1:].copy()
    df_aligned["Transformer_Pred"] = transformer_preds

    # XGB Prediction
    X_xgb = preprocess_input(df_aligned, training_columns)
    X_xgb["Transformer_Pred"] = transformer_preds
    xgb_preds = xgb_model.predict(X_xgb)

    final_preds = 0.6 * transformer_preds[-len(xgb_preds):] + 0.4 * xgb_preds

    # Match lengths properly
    results = df_aligned.iloc[-len(final_preds):].copy()
    results["Predicted_Units_Sold"] = final_preds

    if "Units Sold" in results.columns:
        y_true = results["Units Sold"].values
        mape = mean_absolute_percentage_error(y_true, final_preds) * 100
    else:
        mape = None

    return results, mape


# -----------------------------
# Streamlit UI
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
            st.info("Actual 'Units Sold' not found — MAPE not computed.")

        st.dataframe(df_results.tail(20))

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("👆 Please upload your CSV to begin.")
