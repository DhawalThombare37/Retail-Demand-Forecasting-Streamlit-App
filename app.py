import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

# --- Title
st.title("📈 Retail Demand Forecasting (XGB + Transformer Fusion)")

# --- Load trained model + scaler
@st.cache_resource
def load_model():
    model = joblib.load("xgb_transformer_final.pkl")
    try:
        scaler = joblib.load("scaler.pkl")
    except:
        scaler = None
    return model, scaler

xgb_model, scaler = load_model()

# --- Data upload
uploaded_file = st.file_uploader("📤 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Data Loaded — Shape:", df.shape)

    # === Ensure consistency ===
    df.columns = df.columns.str.strip()
    df.rename(columns={'Transformer_Pred': 'transformer_predictions_scaled'}, inplace=True)

    # --- Feature engineering (align with training)
    def create_features(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
        return df

    df = create_features(df)

    # --- One-hot encode categorical columns
    cat_cols = ['Discount', 'Holiday/Promotion']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # --- Check for missing training features and align
    required_features = [
        'Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Competitor Pricing',
        'year', 'month', 'day', 'dayofweek', 'weekofyear',
        'Inventory Level_lag_7', 'Units Sold_lag_7', 'Units Ordered_lag_7', 'Demand Forecast_lag_7', 'Price_lag_7',
        'Inventory Level_rolling_mean_7', 'Inventory Level_rolling_std_7',
        'Units Sold_rolling_mean_7', 'Units Sold_rolling_std_7',
        'Units Ordered_rolling_mean_7', 'Units Ordered_rolling_std_7',
        'Demand Forecast_rolling_mean_7', 'Demand Forecast_rolling_std_7',
        'Price_rolling_mean_7', 'Price_rolling_std_7',
        'Discount_0', 'Discount_5', 'Discount_10', 'Discount_15', 'Discount_20',
        'Holiday/Promotion_0', 'Holiday/Promotion_1',
        'transformer_predictions_scaled'
    ]

    for col in required_features:
        if col not in df.columns:
            df[col] = 0  # fill missing columns (if any)

    df = df[required_features]

    # --- Scale if applicable
    if scaler is not None:
        X_scaled = scaler.transform(df)
        X_scaled = pd.DataFrame(X_scaled, columns=required_features)
    else:
        X_scaled = df

    # --- Predict
    try:
        preds = xgb_model.predict(X_scaled)
        df["Predicted_Demand"] = preds

        if "Demand Forecast" in df.columns:
            mape = mean_absolute_percentage_error(df["Demand Forecast"], preds) * 100
            st.success(f"✅ Prediction Completed — MAPE: {mape:.2f}%")
        else:
            st.info("Predictions completed. No true labels found for MAPE.")

        # --- Display result
        st.dataframe(df[["Predicted_Demand"]].head(10))

        # --- Plot
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(df["Predicted_Demand"].values, label="Predicted", color="orange")
        if "Demand Forecast" in df.columns:
            ax.plot(df["Demand Forecast"].values, label="Actual", color="green")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("⬆️ Please upload your training/test CSV to start prediction.")
