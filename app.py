import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --------------------------
# TransformerPredictor Class
# --------------------------
class TransformerPredictor:
    def __init__(self, model, scaler, training_columns, sequence_length):
        self.model = model
        self.scaler = scaler
        self.training_columns = training_columns
        self.sequence_length = sequence_length

    def preprocess(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)

        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

        lag_period = 7
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)
            df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(window=rolling_window).mean()
            df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(window=rolling_window).std()

        df = df.fillna(0)

        features_to_use = [c for c in df.columns if c not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        df_processed = pd.get_dummies(df[features_to_use], columns=['Discount', 'Holiday/Promotion'])
        for col in self.training_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.training_columns]

        X_scaled = self.scaler.transform(df_processed)

        X_sequences = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_sequences.append(X_scaled[i:(i + self.sequence_length)])

        if not X_sequences:
            return np.array([]), df.iloc[self.sequence_length - 1:]
        return np.array(X_sequences), df.iloc[self.sequence_length - 1:].reset_index(drop=True)

    def predict(self, df):
        X_seq, original_df = self.preprocess(df)
        if X_seq.size == 0:
            return np.array([]), original_df
        preds = self.model.predict(X_seq)
        return preds.flatten(), original_df

# --------------------------
# Load the saved model
# --------------------------
@st.cache_resource
def load_model():
    with open("xgb_transformer_final.pkl", "rb") as f:
        model = pickle.load(f)
    return model

xgb_transformer_final = load_model()

# --------------------------
# Streamlit App UI
# --------------------------
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("🧠 Retail Demand Forecasting Dashboard")
st.write("Predict future demand using the trained Transformer + XGBoost model.")

# Sidebar for file upload
st.sidebar.header("📥 Input Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV for prediction", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset Preview")
    st.dataframe(df_input.head())

    st.info("Running preprocessing and prediction...")
    preds, df_processed = xgb_transformer_final.predict(df_input)

    if preds.size > 0:
        df_processed["Predicted_Demand"] = preds
        st.success("✅ Predictions generated successfully!")

        st.write("### 📊 Predictions (Top 20 Rows)")
        st.dataframe(df_processed[["Date", "Predicted_Demand"]].head(20))

        # Plot predictions
        st.write("### 📈 Predicted Demand Over Time")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_processed["Date"], df_processed["Predicted_Demand"], label="Predicted Demand", color="tab:blue")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Demand")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Download CSV
        csv = df_processed.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
    else:
        st.warning("⚠️ Not enough data to create valid sequences. Please upload a larger file.")
else:
    st.info("Upload a CSV file with the same structure as training data to generate predictions.")
