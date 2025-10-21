import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="🧠 Cyber Demand Forecasting", layout="wide")

# Inject custom CSS for dark neon-glass theme
st.markdown("""
<style>
:root {
  --bg-dark: #0a0a0f;
  --card-bg: rgba(255,255,255,0.05);
  --neon-blue: #1E90FF;
  --neon-purple: #8E44AD;
  --neon-pink: #FF00FF;
  --muted: rgba(255,255,255,0.6);
  --text-color: #ffffff;
  --glass-border: rgba(255,255,255,0.12);
  --font-main: 'Sora', 'Inter', sans-serif;
}

/* main layout */
body, .stApp {
  background: radial-gradient(circle at 10% 20%, #0a0a0f, #090a10 70%);
  color: var(--text-color);
  font-family: var(--font-main);
}

/* neon pulse header */
.neon-title {
  text-align: center;
  font-size: 2.4rem;
  font-weight: 700;
  color: #00FFFF;
  text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 40px #0088FF;
  animation: pulse 3s ease-in-out infinite alternate;
  letter-spacing: 1px;
  margin-bottom: 10px;
}

@keyframes pulse {
  0% { text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 40px #0088FF; }
  100% { text-shadow: 0 0 25px #00FFFF, 0 0 50px #0099FF, 0 0 80px #00FFFF; }
}

/* metric cards */
.metric {
  padding: 18px;
  border-radius: 14px;
  transition: transform 0.25s cubic-bezier(.2,.9,.2,1), box-shadow 0.25s;
  transform: translateZ(0);
  border:1px solid rgba(255,255,255,0.04);
  background: linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
  box-shadow: 0 6px 18px rgba(2,6,23,0.65);
}
.metric:hover {
  transform: translateY(-6px) scale(1.02);
  box-shadow: 0 18px 40px rgba(2,6,23,0.8);
}
.metric .label { color: var(--muted); font-size:0.9rem; }
.metric .value { font-weight:700; font-size:1.5rem; margin-top:6px; color: white; }

/* uploader */
.uploader {
  border: 1px dashed rgba(255,255,255,0.08);
  border-radius:12px;
  padding:18px;
  text-align:center;
  color:var(--muted);
  transition: background 0.25s, transform 0.2s;
}
.uploader:hover {
  background: linear-gradient(135deg, rgba(30,144,255,0.08), rgba(142,68,173,0.08));
  transform: translateY(-4px);
}
.uploader strong { color: white; font-weight:700; }

/* badges */
.badge {
  display:inline-block;
  padding:8px 12px;
  border-radius:999px;
  background: linear-gradient(90deg, rgba(30,144,255,0.14), rgba(142,68,173,0.14));
  border: 1px solid rgba(255,255,255,0.08);
  color:white;
  font-weight:600;
  font-size:0.95rem;
  box-shadow: 0 6px 16px rgba(12, 22, 45, 0.5);
  backdrop-filter: blur(6px);
}

/* glow background */
.glow {
  position: absolute;
  pointer-events: none;
  filter: blur(80px);
  opacity: 0.6;
}

.stDataFrame table {
  border-radius: 10px !important;
  overflow: hidden;
  color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# APP HEADER
# ------------------------------------------------------------
st.markdown('<div class="neon-title">⚡ Retail Demand Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="badge">Transformer + XGBoost Ensemble | Expected MAPE: ~3%</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# BACKEND LOADING (UNCHANGED)
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    import os
    st.info("🔍 Checking for model files...")
    required_files = {
        'transformer_model.keras': 'Transformer Model',
        'xgb_model.pkl': 'XGBoost Model',
        'scaler.pkl': 'Scaler',
        'training_columns.pkl': 'Training Columns',
        'xgb_columns.pkl': 'XGBoost Columns',
        'sequence_length.pkl': 'Sequence Length'
    }
    missing_files = []
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)
    if missing_files:
        st.error(f"❌ Missing files: {missing_files}")
        return None, None, None, None, None, None
    try:
        st.info("📦 Loading models...")
        transformer = tf.keras.models.load_model("transformer_model.keras")
        xgb = joblib.load("xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        training_cols = joblib.load("training_columns.pkl")
        xgb_cols = joblib.load("xgb_columns.pkl")
        seq_len = joblib.load("sequence_length.pkl")
        st.success("✅ All models loaded!")
        return transformer, xgb, scaler, training_cols, xgb_cols, seq_len
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length = load_models()
if transformer_model is None:
    st.stop()

# ------------------------------------------------------------
# PREDICTOR CLASS (UNCHANGED)
# ------------------------------------------------------------
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
        df = df.sort_values(by='Date').reset_index(drop=True)
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
        lag_period = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
            df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
        df = df.fillna(0)
        features = [col for col in df.columns if col not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        X = df[features]
        y = df['Demand Forecast']
        X = pd.get_dummies(X, columns=['Discount', 'Holiday/Promotion'])
        return X, y, df

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def predict(self, df_input):
        test_date = pd.to_datetime(df_input['Date']).max() - pd.DateOffset(months=3)
        X, y, df_orig = self.preprocess(df_input)
        df_orig['Date'] = pd.to_datetime(df_orig['Date'])
        test_mask = df_orig['Date'] > test_date
        X = X[test_mask].reset_index(drop=True)
        y = y[test_mask].reset_index(drop=True)
        df_orig = df_orig[test_mask].reset_index(drop=True)
        for col in self.training_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.training_columns]
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y.values)
        if len(X_seq) == 0:
            st.error(f"Need at least {self.sequence_length + 1} rows")
            return None, None
        trans_preds = self.transformer.predict(X_seq, verbose=0)
        X_aligned = X.iloc[self.sequence_length:].copy()
        y_aligned = y.values[self.sequence_length:].copy()
        df_aligned = df_orig.iloc[self.sequence_length:].copy()
        X_aligned['transformer_predictions_scaled'] = trans_preds.flatten()
        for col in self.xgb_columns:
            if col not in X_aligned.columns:
                X_aligned[col] = 0
        X_aligned = X_aligned[self.xgb_columns]
        final_preds = self.xgb.predict(X_aligned)
        df_results = df_aligned.reset_index(drop=True).copy()
        df_results['Predicted_Demand'] = final_preds
        epsilon = 1e-8
        y_safe = y_aligned.copy()
        y_safe[y_safe == 0] = epsilon
        mape = mean_absolute_percentage_error(y_safe, final_preds) * 100
        return df_results, mape

# ------------------------------------------------------------
# UPLOAD + UI DISPLAY
# ------------------------------------------------------------
st.markdown("### 📁 Upload CSV File", unsafe_allow_html=True)
uploaded = st.file_uploader("Upload retail_store_inventory.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.markdown('<div class="uploader"><strong>File Uploaded Successfully!</strong></div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Rows", f"{len(df):,}")
    with col2: st.metric("Stores", df['Store ID'].nunique())
    with col3: st.metric("Products", df['Product ID'].nunique())

    predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)
    with st.spinner("⚡ Running predictions..."):
        results, mape = predictor.predict(df)

    if results is not None:
        st.markdown("---")
        st.markdown("### 🎯 Results Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            emoji = "🎉" if mape <= 5 else "✅" if mape <= 10 else "⚠️"
            st.metric("MAPE", f"{mape:.2f}%", f"{emoji}")
        with col2:
            st.metric("Predictions", f"{len(results):,}")
        with col3:
            st.metric("Accuracy", f"{max(0,100-mape):.1f}%")

        st.markdown("### 📈 Actual vs Predicted Demand")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results['Date'], y=results['Demand Forecast'],
                                 mode='lines', name='Actual Demand',
                                 line=dict(color='#00FFFF', width=3)))
        fig.add_trace(go.Scatter(x=results['Date'], y=results['Predicted_Demand'],
                                 mode='lines', name='Predicted Demand',
                                 line=dict(color='#FF00FF', width=2, dash='dot')))
        fig.update_layout(template='plotly_dark',
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)',
                          legend=dict(font=dict(color='white')),
                          font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

        display = results[['Date','Store ID','Product ID','Demand Forecast','Predicted_Demand']].copy()
        display['Error_%'] = (abs(display['Demand Forecast']-display['Predicted_Demand'])/(display['Demand Forecast']+1e-8)*100).round(2)
        st.dataframe(display.head(50), use_container_width=True)

        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", use_container_width=True)
else:
    st.info("👆 Upload CSV to start predictions")
