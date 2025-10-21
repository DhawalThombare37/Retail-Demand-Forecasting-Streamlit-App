import streamlit as st
import pandas as pd
import numpy as np
import joblib, tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

# ---------------------------------------------
# 🌈 PAGE CONFIGURATION + STYLING
# ---------------------------------------------
st.set_page_config(page_title="Retail Demand Forecast Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
    font-family: 'Poppins', sans-serif;
}
.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(31,38,135,0.37);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 20px;
    margin-bottom: 25px;
    color: white;
}
.metric-card {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.metric-card h2 {
    color: cyan;
    text-shadow: 0 0 15px cyan;
}
.plot-glow {
    text-align: center;
    font-size: 26px;
    font-weight: 600;
    color: cyan;
    text-shadow: 0px 0px 10px cyan;
    margin-bottom: 10px;
}
.download-button button {
    background-color: #00ffff !important;
    color: black !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 🧠 MODEL LOADING
# ---------------------------------------------
@st.cache_resource
def load_models():
    transformer = tf.keras.models.load_model("transformer_model.keras")
    xgb = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_cols = joblib.load("training_columns.pkl")
    xgb_cols = joblib.load("xgb_columns.pkl")
    seq_len = joblib.load("sequence_length.pkl")
    return transformer, xgb, scaler, training_cols, xgb_cols, seq_len

st.title("🛒 Retail Demand Forecasting Dashboard")
st.markdown("**Transformer + XGBoost Ensemble | Expected MAPE ≈ 2.9 %**")

try:
    transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length = load_models()
except Exception as e:
    st.error("❌ Model files missing or corrupted.")
    st.stop()

# ---------------------------------------------
# 🔧 PREDICTOR CLASS (same logic from your code)
# ---------------------------------------------
class Predictor:
    def __init__(self, transformer, xgb, scaler, train_cols, xgb_cols, seq_len):
        self.transformer = transformer
        self.xgb = xgb
        self.scaler = scaler
        self.training_columns = train_cols
        self.xgb_columns = xgb_cols
        self.sequence_length = seq_len

    def preprocess(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Feature engineering (same as your code)
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

        lag_period, roll_window = 7, 7
        for col in ['Inventory Level','Units Sold','Units Ordered','Demand Forecast','Price']:
            df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID','Product ID'])[col].shift(lag_period)
            df[f'{col}_roll_mean_{roll_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(roll_window).mean().reset_index(drop=True)
        df.fillna(0, inplace=True)

        features = [c for c in df.columns if c not in ['Date','Demand Forecast','Store ID','Product ID','Category','Region','Weather Condition','Seasonality']]
        X = df[features]
        y = df['Demand Forecast']
        X = pd.get_dummies(X, columns=['Discount','Holiday/Promotion'])
        return X, y, df

    def create_sequences(self, X, y):
        Xs, ys = [], []
        for i in range(len(X)-self.sequence_length):
            Xs.append(X[i:i+self.sequence_length])
            ys.append(y[i+self.sequence_length])
        return np.array(Xs), np.array(ys)

    def predict(self, df_input):
        X, y, df_prep = self.preprocess(df_input)
        for c in self.training_columns:
            if c not in X.columns: X[c] = 0
        X = X[self.training_columns]
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y.values)
        trans_preds = self.transformer.predict(X_seq, verbose=0)

        X_aligned = X.iloc[self.sequence_length:].copy()
        y_aligned = y.values[self.sequence_length:]
        df_aligned = df_prep.iloc[self.sequence_length:].copy()
        X_aligned['transformer_predictions_scaled'] = trans_preds.flatten()
        for c in self.xgb_columns:
            if c not in X_aligned.columns: X_aligned[c] = 0
        X_aligned = X_aligned[self.xgb_columns]

        final_preds = self.xgb.predict(X_aligned)
        df_results = df_aligned.copy()
        df_results['Predicted_Demand'] = final_preds
        y_safe = np.where(y_aligned==0,1e-8,y_aligned)
        mape = mean_absolute_percentage_error(y_safe, final_preds)*100
        return df_results, mape

# ---------------------------------------------
# 📂 FILE UPLOAD + PREDICTION
# ---------------------------------------------
uploaded = st.file_uploader("📁 Upload Retail Data (CSV)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)

    with st.spinner("🧮 Predicting…"):
        results, mape = predictor.predict(df)

    # ---------------------------------------------
    # 📊 METRIC CARDS
    # ---------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f"<div class='metric-card'><h2>{mape:.2f}%</h2><p>MAPE</p></div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='metric-card'><h2>{df['Store ID'].nunique()}</h2><p>Stores</p></div>", unsafe_allow_html=True)
    with col3: st.markdown(f"<div class='metric-card'><h2>{df['Product ID'].nunique()}</h2><p>Products</p></div>", unsafe_allow_html=True)
    with col4: st.markdown(f"<div class='metric-card'><h2>{df['Date'].min().split()[0]} → {df['Date'].max().split()[0]}</h2><p>Date Range</p></div>", unsafe_allow_html=True)

    # ---------------------------------------------
    # 🟢 CHART 1: Actual vs Predicted Demand
    # ---------------------------------------------
    st.markdown("<div class='glass plot-glow'>Actual vs Predicted Demand</div>", unsafe_allow_html=True)
    agg = results.groupby('Date')[['Demand Forecast','Predicted_Demand']].sum().reset_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], name='Actual', mode='lines+markers', line=dict(color='cyan',width=3)))
    fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Predicted_Demand'], name='Predicted', mode='lines+markers', line=dict(color='magenta',width=3,dash='dash')))
    fig1.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

    # ---------------------------------------------
    # 🟣 CHART 2: Product-wise MAPE
    # ---------------------------------------------
    st.markdown("<div class='glass plot-glow'>Product-wise MAPE</div>", unsafe_allow_html=True)
    results['APE'] = np.abs((results['Demand Forecast']-results['Predicted_Demand'])/results['Demand Forecast'])*100
    prod_mape = results.groupby('Product ID')['APE'].mean().reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=prod_mape['Product ID'], y=prod_mape['APE'], mode='lines+markers',
                              line=dict(color='lime',width=3), name='MAPE'))
    fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------
    # 💾 DOWNLOAD BUTTON
    # ---------------------------------------------
    st.markdown("<div class='glass plot-glow'>Download Predictions</div>", unsafe_allow_html=True)
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predicted CSV", csv, "predictions.csv", mime="text/csv", key="download-csv")
else:
    st.info("👆 Upload your CSV to generate predictions")
