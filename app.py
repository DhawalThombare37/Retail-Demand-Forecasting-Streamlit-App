# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Retail Demand Forecasting — Glassmorphic", layout="wide", page_icon="🛒")

# -----------------------
# Custom CSS (Glassmorphism + 3D feel)
# -----------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Sora:wght@600;700&display=swap" rel="stylesheet">
<style>
:root{
    --glass-bg: rgba(255,255,255,0.08);
    --glass-border: rgba(255,255,255,0.12);
    --accent-1: linear-gradient(135deg, rgba(30,144,255,0.95), rgba(142,68,173,0.9));
    --accent-2: linear-gradient(135deg, rgba(44, 230, 183, 0.95), rgba(108, 96, 255, 0.9));
    --muted: rgba(255,255,255,0.6);
    --glass-blur: 12px;
}
/* page background with extended strokes */
.stApp {
    background: radial-gradient(1200px 500px at 15% 15%, rgba(142,68,173,0.12), transparent 10%),
                radial-gradient(1100px 400px at 85% 85%, rgba(30,144,255,0.12), transparent 7%),
                radial-gradient(900px 300px at 50% 50%, rgba(108,96,255,0.08), transparent 5%),
                linear-gradient(180deg, #0f1226 0%, #071028 100%);
    color: #e9eef8;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
}

/* uploader button text fix */
div.stFileUploader > label > div {
    color: black !important;
    font-weight: 600;
}

/* header */
header .decoration, header .css-18e3th9 { background: transparent; }
.app-header { display:flex; align-items:center; gap:14px; }
.logo-plate{
    width:56px; height:56px; border-radius:12px;
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 6px 18px rgba(12,19,40,0.6), 0 1px 0 rgba(255,255,255,0.02) inset;
    display:flex; align-items:center; justify-content:center;
    transform: perspective(800px) rotateX(6deg); transition: transform 0.25s ease;
}
.logo-plate:hover { transform: perspective(800px) rotateX(0deg) translateY(-6px); }
.app-title { font-family: 'Sora', sans-serif; font-size: 1.45rem; font-weight:700; letter-spacing: -0.5px; color: white; }
.app-sub { color: var(--muted); font-size: 0.92rem; margin-top: -4px; }

/* glass card */
.glass { background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02)); border-radius: 16px; border: 1px solid var(--glass-border); backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur)); box-shadow: 0 8px 30px rgba(2,6,23,0.6); }
.glass-strong { background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.035)); border-radius: 18px; border: 1px solid rgba(255,255,255,0.12); backdrop-filter: blur(calc(var(--glass-blur) + 6px)); padding:18px; }

/* metric card */
.metric { padding:18px; border-radius:14px; transition: transform 0.25s cubic-bezier(.2,.9,.2,1), box-shadow 0.25s; transform: translateZ(0); border:1px solid rgba(255,255,255,0.04); background: linear-gradient(135deg, rgba(255,255,255,0.022), rgba(255,255,255,0.01)); box-shadow: 0 6px 18px rgba(2,6,23,0.45); }
.metric:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 18px 40px rgba(2,6,23,0.6); }
.metric .label { color: rgba(255,255,255,0.75); font-size:0.9rem; }
.metric .value { font-weight:700; font-size:1.5rem; margin-top:6px; color: white; }

/* uploader */
.uploader { border: 1px dashed rgba(255,255,255,0.06); border-radius:12px; padding:18px; text-align:center; color:var(--muted); transition: background 0.25s, transform 0.2s; }
.uploader:hover{ background: linear-gradient(135deg, rgba(30,144,255,0.02), rgba(142,68,173,0.02)); transform: translateY(-4px); }
.uploader strong { color: white; font-weight:700; }

/* small helper */
.muted { color:var(--muted); font-size:0.9rem; }

/* tiny animated shine for badges */
.badge { display:inline-block; padding:8px 12px; border-radius:999px; background: linear-gradient(90deg, rgba(30,144,255,0.14), rgba(142,68,173,0.14)); border: 1px solid rgba(255,255,255,0.04); color:white; font-weight:600; font-size:0.95rem; box-shadow: 0 6px 16px rgba(12, 22, 45, 0.5); backdrop-filter: blur(6px); }

/* tiny tooltip helper */
[title] { cursor: help; }

/* small table style fix */
.stDataFrame table { border-radius: 10px !important; overflow: hidden; }

/* responsive tweaks */
@media (max-width: 900px) { .app-title { font-size:1.15rem; } }

/* subtle floating background glows */
.glow { position: absolute; pointer-events: none; filter: blur(80px); opacity: 0.7; }

/* glass plots glow */
.plot-glow { padding: 12px; border-radius: 14px; margin-bottom: 16px; box-shadow: 0 6px 20px rgba(30,144,255,0.35), 0 1px 0 rgba(255,255,255,0.03) inset; background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Floating glows (extended)
# -----------------------
st.markdown("""
<div style="position: absolute; right: -200px; top: 30px;">
    <div class="glow" style="width:600px;height:300px;background: radial-gradient(circle at 20% 30%, rgba(30,144,255,0.26), transparent 20%), radial-gradient(circle at 80% 70%, rgba(142,68,173,0.18), transparent 20%); border-radius: 50%;"></div>
</div>
<div style="position: absolute; left: -150px; bottom: 50px;">
    <div class="glow" style="width:500px;height:200px;background: radial-gradient(circle at 50% 50%, rgba(44,230,183,0.18), transparent 25%), radial-gradient(circle at 20% 70%, rgba(108,96,255,0.14), transparent 15%); border-radius: 50%;"></div>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Header area
# -----------------------
header_col1, header_col2 = st.columns([0.12, 0.88])
with header_col1:
    st.markdown("""
    <div class="logo-plate glass">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="1.5" y="1.5" width="21" height="21" rx="5" fill="url(#g)"/>
          <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#1E90FF"/><stop offset="1" stop-color="#8E44AD"/></linearGradient></defs>
        </svg>
    </div>
    """, unsafe_allow_html=True)
with header_col2:
    st.markdown('<div class="app-header"><div><div class="app-title">Retail Demand Forecasting</div><div class="app-sub">Transformer + XGBoost ensemble • Glassmorphic dashboard</div></div></div>', unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------
# Model loader
# -----------------------
@st.cache_resource
def load_models():
    import os
    transformer = tf.keras.models.load_model("transformer_model.keras")
    xgb = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_cols = joblib.load("training_columns.pkl")
    xgb_cols = joblib.load("xgb_columns.pkl")
    seq_len = joblib.load("sequence_length.pkl")
    return transformer, xgb, scaler, training_cols, xgb_cols, seq_len

try:
    transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length = load_models()
    models_loaded = True
except Exception as e:
    st.error("❌ Model files missing or failed to load.")
    st.exception(e)
    models_loaded = False

# -----------------------
# Predictor class (unchanged logic)
# -----------------------
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
            if col in df.columns:
                df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)
            else:
                df[f'{col}_lag_{lag_period}'] = 0
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            if col in df.columns:
                df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
            else:
                df[f'{col}_rolling_mean_{rolling_window}'] = 0
                df[f'{col}_rolling_std_{rolling_window}'] = 0
        df = df.fillna(0)
        features = [c for c in df.columns if c not in ['Date','Demand Forecast','Store ID','Product ID','Category','Region','Weather Condition','Seasonality']]
        X = df[features].copy()
        y = df['Demand Forecast'].copy()
        for col in ['Discount','Holiday/Promotion']:
            if col in X.columns:
                X = pd.get_dummies(X, columns=[col])
        return X, y, df

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X)-self.sequence_length):
            X_seq.append(X[i:(i+self.sequence_length)])
            y_seq.append(y[i+self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def predict(self, df_input):
        X, y, df_orig = self.preprocess(df_input)
        df_orig['Date'] = pd.to_datetime(df_orig['Date'])
        test_date = df_orig['Date'].max() - pd.DateOffset(months=3)
        mask = df_orig['Date'] > test_date
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        df_orig = df_orig[mask].reset_index(drop=True)
        for col in self.training_columns:
            if col not in X.columns: X[col] = 0
        X = X[self.training_columns]
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y.values)
        if len(X_seq)==0: raise ValueError(f"Need at least {self.sequence_length+1} rows in test period")
        trans_preds = self.transformer.predict(X_seq, verbose=0)
        X_aligned = X.iloc[self.sequence_length:].copy().reset_index(drop=True)
        y_aligned = y.values[self.sequence_length:].copy()
        df_aligned = df_orig.iloc[self.sequence_length:].copy().reset_index(drop=True)
        X_aligned['transformer_predictions_scaled'] = trans_preds.flatten()
        for col in self.xgb_columns:
            if col not in X_aligned.columns: X_aligned[col]=0
        X_aligned = X_aligned[self.xgb_columns]
        final_preds = self.xgb.predict(X_aligned)
        df_results = df_aligned.copy()
        df_results['Predicted_Demand'] = final_preds
        y_safe = y_aligned.copy()
        y_safe[y_safe==0]=1e-8
        mape = mean_absolute_percentage_error(y_safe, final_preds)*100
        return df_results.reset_index(drop=True), mape

# -----------------------
# Uploader
# -----------------------
st.markdown("<div class='glass-strong'><h3>Upload your Retail CSV</h3></div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag & drop CSV or click to browse", type=["csv"], help="CSV must include Date, Store ID, Product ID, Demand Forecast, etc.")

if uploaded_file and models_loaded:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)
    with st.spinner("🌊 Running preprocessing and generating predictions..."):
        results, mape = predictor.predict(df)

    # metrics cards
    total_rows = len(results)
    total_stores = results['Store ID'].nunique()
    total_products = results['Product ID'].nunique()
    accuracy = max(0, 100 - mape)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric'><div class='label'>Total Rows</div><div class='value'>{total_rows:,}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric'><div class='label'>Stores</div><div class='value'>{total_stores}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric'><div class='label'>Products</div><div class='value'>{total_products}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric'><div class='label'>MAPE</div><div class='value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Actual vs Predicted chart
    fig1 = go.Figure()
    agg = results.groupby('Date')[['Demand Forecast','Predicted_Demand']].sum().reset_index()
    fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], mode='lines+markers',
                              name='Actual', line=dict(color='cyan', width=3), marker=dict(size=6)))
    fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Predicted_Demand'], mode='lines+markers',
                              name='Predicted', line=dict(color='magenta', width=3, dash='dash'), marker=dict(size=6)))
    fig1.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use
