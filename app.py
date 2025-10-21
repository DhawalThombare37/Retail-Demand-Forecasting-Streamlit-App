# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Retail Demand Forecasting — Glassmorphic", layout="wide", page_icon="🛒")

# -----------------------
# Custom CSS (Glassmorphism + Animated Background)
# -----------------------
st.markdown("""
<style>
:root{
    --glass-bg: rgba(255,255,255,0.08);
    --glass-border: rgba(255,255,255,0.12);
    --accent-1: linear-gradient(135deg, rgba(30,144,255,0.95), rgba(142,68,173,0.9));
    --accent-2: linear-gradient(135deg, rgba(44, 230, 183, 0.95), rgba(108, 96, 255, 0.9));
    --muted: rgba(255,255,255,0.75);
    --glass-blur: 14px;
}

.stApp {
    background: radial-gradient(1200px 500px at 15% 15%, rgba(30,144,255,0.14), transparent 15%),
                radial-gradient(1000px 400px at 80% 80%, rgba(142,68,173,0.12), transparent 15%),
                radial-gradient(600px 600px at 40% 70%, rgba(44,230,183,0.08), transparent 20%),
                linear-gradient(180deg, #0f1226 0%, #071028 100%);
    color: #e9eef8;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
}

.glass {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
    border-radius: 18px;
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    box-shadow: 0 10px 30px rgba(2,6,23,0.6);
    padding:18px;
    margin-bottom:18px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 48px rgba(2,6,23,0.75);
}

.metric {
    padding:20px;
    border-radius:16px;
    background: linear-gradient(135deg, rgba(30,144,255,0.12), rgba(142,68,173,0.1));
    border:1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 26px rgba(12,22,45,0.7);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 18px 44px rgba(30,144,255,0.6);
}
.metric .label { color: var(--muted); font-size:0.95rem; font-weight:600; }
.metric .value { font-weight:700; font-size:1.6rem; margin-top:6px; color:#ffffff; text-shadow:0 1px 6px rgba(0,0,0,0.3); }

h2, h3 { color:#ffffff; text-shadow:0 1px 6px rgba(0,0,0,0.4); }

.uploader {
    border: 1px dashed rgba(255,255,255,0.08);
    border-radius:14px;
    padding:22px;
    text-align:center;
    color:var(--muted);
    transition: background 0.25s, transform 0.2s;
}
.uploader:hover{
    background: linear-gradient(135deg, rgba(30,144,255,0.03), rgba(142,68,173,0.03));
    transform: translateY(-4px);
}

.plot-glow {
    position: relative;
}
.plot-glow::before {
    content:'';
    position: absolute;
    top:-60px; left:-60px;
    width:300px; height:300px;
    border-radius:50%;
    background: radial-gradient(circle, rgba(30,144,255,0.18), transparent 60%);
    filter: blur(80px);
    animation: float 12s ease-in-out infinite alternate;
    z-index:-1;
}
.plot-glow::after {
    content:'';
    position: absolute;
    bottom:-60px; right:-60px;
    width:300px; height:300px;
    border-radius:50%;
    background: radial-gradient(circle, rgba(142,68,173,0.15), transparent 60%);
    filter: blur(60px);
    animation: float2 15s ease-in-out infinite alternate;
    z-index:-1;
}
@keyframes float {
    0% {transform: translateY(0px) translateX(0px);}
    50% {transform: translateY(40px) translateX(20px);}
    100% {transform: translateY(0px) translateX(0px);}
}
@keyframes float2 {
    0% {transform: translateY(0px) translateX(0px);}
    50% {transform: translateY(-30px) translateX(-20px);}
    100% {transform: translateY(0px) translateX(0px);}
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("<h2>🛒 Retail Demand Forecasting</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:rgba(255,255,255,0.75)'>Upload CSV containing: Date, Store ID, Product ID, Demand Forecast (Actual), Predicted_Demand (optional)</p>", unsafe_allow_html=True)

# -----------------------
# File uploader
# -----------------------
uploaded_file = st.file_uploader("Upload your CSV here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Ensure Predicted_Demand column exists
    if 'Predicted_Demand' not in df.columns:
        st.warning("⚠️ 'Predicted_Demand' column not found. Only actual demand will be shown.")
        df['Predicted_Demand'] = np.nan

    # -----------------------
    # Metrics cards
    # -----------------------
    total_rows = df.shape[0]
    total_stores = df['Store ID'].nunique() if 'Store ID' in df.columns else 0
    total_products = df['Product ID'].nunique() if 'Product ID' in df.columns else 0
    mape = np.mean(np.abs((df['Demand Forecast'] - df['Predicted_Demand'])/(df['Demand Forecast']+1e-8)))*100 if df['Predicted_Demand'].notna().any() else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric'><div class='label'>Total Rows</div><div class='value'>{total_rows:,}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric'><div class='label'>Stores</div><div class='value'>{total_stores}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric'><div class='label'>Products</div><div class='value'>{total_products}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric'><div class='label'>MAPE</div><div class='value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # -----------------------
    # Visualization 1: Actual vs Predicted
    # -----------------------
    st.markdown("<div class='glass plot-glow'><h3>Actual vs Predicted Demand</h3></div>", unsafe_allow_html=True)
    fig1 = go.Figure()

    if df['Predicted_Demand'].notna().any():
        agg = df.groupby('Date')[['Demand Forecast', 'Predicted_Demand']].sum().reset_index()
        fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], mode='lines+markers',
                                  name='Actual', line=dict(color='cyan', width=3), marker=dict(size=6)))
        fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Predicted_Demand'], mode='lines+markers',
                                  name='Predicted', line=dict(color='magenta', width=3, dash='dash'), marker=dict(size=6)))
    else:
        agg = df.groupby('Date')[['Demand Forecast']].sum().reset_index()
        fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], mode='lines+markers',
                                  name='Actual Demand', line=dict(color='cyan', width=3), marker=dict(size=6)))
        st.warning("⚠️ 'Predicted_Demand' column missing — only Actual Demand is displayed.")

    fig1.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       legend=dict(bgcolor='rgba(255,255,255,0.03)'))
    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------
    # Visualization 2: Product-wise Error (only if predictions exist)
    # -----------------------
    if df['Predicted_Demand'].notna().any():
        st.markdown("<div class='glass plot-glow'><h3>Product-wise Average Prediction Error</h3></div>", unsafe_allow_html=True)
        df['Abs_Error'] = abs(df['Demand Forecast'] - df['Predicted_Demand'])
        prod_err = df.groupby('Product ID', dropna=True)['Abs_Error'].mean().reset_index().sort_values('Abs_Error', ascending=False)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=prod_err['Product ID'], y=prod_err['Abs_Error'], mode='lines+markers',
                                  line=dict(color='orange', width=3), marker=dict(size=6)))
        fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ℹ️ Prediction Error chart unavailable — no 'Predicted_Demand' column found.")
