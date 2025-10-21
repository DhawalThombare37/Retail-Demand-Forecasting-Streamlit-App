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
# Custom CSS (glassmorphic, vibrant cards, background glows)
# -----------------------
st.markdown("""
<style>
:root{
    --glass-blur: 12px;
    --muted: rgba(255,255,255,0.6);
}
.stApp {
    background: radial-gradient(1000px 400px at 10% 10%, rgba(142,68,173,0.12), transparent 8%),
                radial-gradient(900px 300px at 95% 90%, rgba(30,144,255,0.10), transparent 5%),
                linear-gradient(180deg, #0f1226 0%, #071028 100%);
    color: #e9eef8;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
}
.glass-strong {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.035));
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(var(--glass-blur));
    padding:18px;
}
.metric {
    padding:18px; border-radius:14px; transition: transform 0.25s; 
    border:1px solid rgba(255,255,255,0.04);
    background: linear-gradient(135deg, rgba(255,255,255,0.022), rgba(255,255,255,0.01));
    color:white; box-shadow: 0 6px 18px rgba(2,6,23,0.45);
}
.metric:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 18px 40px rgba(2,6,23,0.6);}
.metric .label { color: rgba(255,255,255,0.75); font-size:0.9rem; }
.metric .value { font-weight:700; font-size:1.5rem; margin-top:6px; }
div.stFileUploader > label > div { color: black !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("<h2 style='color:white;'>Retail Demand Forecasting Dashboard</h2>", unsafe_allow_html=True)

# -----------------------
# Model loader
# -----------------------
@st.cache_resource
def load_models():
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
except:
    st.error("❌ Model files missing or failed to load.")
    models_loaded = False

# -----------------------
# Predictor class
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
        features = [c for c in df.columns if c not in ['Date','Demand Forecast','Store ID','Product ID']]
        X = df[features].fillna(0)
        y = df['Demand Forecast'].copy()
        return X, y, df

    def predict(self, df_input):
        X, y, df_orig = self.preprocess(df_input)
        df_orig['Date'] = pd.to_datetime(df_orig['Date'])
        for col in self.training_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.training_columns]
        X_scaled = self.scaler.transform(X)
        # Transformer predictions
        trans_preds = self.transformer.predict(X_scaled.reshape((X_scaled.shape[0],1,X_scaled.shape[1])), verbose=0)
        # XGB predictions
        for col in self.xgb_columns:
            if col not in X.columns:
                X[col]=0
        X_xgb = X[self.xgb_columns]
        final_preds = self.xgb.predict(X_xgb)
        df_results = df_orig.copy()
        df_results['Predicted_Demand'] = final_preds
        y_safe = y.copy()
        y_safe[y_safe==0] = 1e-8
        mape = mean_absolute_percentage_error(y_safe, final_preds)*100
        return df_results.reset_index(drop=True), mape

# -----------------------
# File uploader
# -----------------------
st.markdown("<div class='glass-strong'><h3>Upload your Retail CSV</h3></div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag & drop CSV or click to browse", type=["csv"])

if uploaded_file and models_loaded:
    df = pd.read_csv(uploaded_file)
    predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)
    with st.spinner("Running predictions..."):
        results, mape = predictor.predict(df)

    # -----------------------
    # Metrics cards
    # -----------------------
    total_rows = len(results)
    total_stores = results['Store ID'].nunique()
    total_products = results['Product ID'].nunique()
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric'><div class='label'>Total Rows</div><div class='value'>{total_rows}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric'><div class='label'>Stores</div><div class='value'>{total_stores}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric'><div class='label'>Products</div><div class='value'>{total_products}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric'><div class='label'>MAPE</div><div class='value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -----------------------
    # Plot 1: Actual vs Predicted Demand
    # -----------------------
    agg = results.groupby('Date')[['Demand Forecast','Predicted_Demand']].sum().reset_index()
    with st.container():
        st.markdown("<div class='glass-strong' style='padding:16px'><h4>Actual vs Predicted Demand</h4></div>", unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], mode='lines+markers',
                                  name='Actual', line=dict(color='cyan', width=3)))
        fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Predicted_Demand'], mode='lines+markers',
                                  name='Predicted', line=dict(color='magenta', width=3, dash='dash')))
        fig1.update_layout(template='plotly_dark',
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(t=30,b=10,l=10,r=10))
        st.plotly_chart(fig1, use_container_width=True)

    # -----------------------
    # Plot 2: Product-wise Average Prediction Error
    # -----------------------
    results['abs_error'] = abs(results['Demand Forecast'] - results['Predicted_Demand'])
    product_error = results.groupby('Product ID')['abs_error'].mean().reset_index()
    with st.container():
        st.markdown("<div class='glass-strong' style='padding:16px'><h4>Product-wise Average Prediction Error</h4></div>", unsafe_allow_html=True)
        fig2 = px.line(product_error, x='Product ID', y='abs_error', markers=True, template='plotly_dark')
        fig2.update_traces(line=dict(color='orange', width=3))
        fig2.update_layout(margin=dict(t=30,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)
