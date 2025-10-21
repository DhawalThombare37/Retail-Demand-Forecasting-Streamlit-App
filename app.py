# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide", page_icon="🛒")

# -----------------------
# Custom CSS (Glassmorphism + Animated Background)
# -----------------------
st.markdown("""
<style>
:root{
    --glass-bg: rgba(255,255,255,0.08);
    --glass-border: rgba(255,255,255,0.12);
    --accent-1: linear-gradient(135deg, rgba(30,144,255,0.95), rgba(142,68,173,0.9));
    --accent-2: linear-gradient(135deg, rgba(44,230,183,0.95), rgba(108,96,255,0.9));
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

.plot-glow::before, .plot-glow::after {
    content:'';
    position:absolute;
    border-radius:50%;
    filter: blur(70px);
    z-index:-1;
}
.plot-glow::before {
    top:-60px; left:-60px; width:300px; height:300px;
    background: radial-gradient(circle, rgba(30,144,255,0.18), transparent 60%);
    animation: float 12s ease-in-out infinite alternate;
}
.plot-glow::after {
    bottom:-60px; right:-60px; width:300px; height:300px;
    background: radial-gradient(circle, rgba(142,68,173,0.15), transparent 60%);
    animation: float2 15s ease-in-out infinite alternate;
}
@keyframes float { 0%{transform:translate(0,0);} 50%{transform:translate(30px,30px);} 100%{transform:translate(0,0);} }
@keyframes float2 { 0%{transform:translate(0,0);} 50%{transform:translate(-30px,-30px);} 100%{transform:translate(0,0);} }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("<h2>🛒 Retail Demand Forecasting Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:rgba(255,255,255,0.75)'>Upload your dataset to view Actual vs Predicted Demand and Product-wise Error Analysis.</p>", unsafe_allow_html=True)

# -----------------------
# File Upload
# -----------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # --- Safe column handling (from 2nd code logic)
    if 'Demand Forecast' not in df.columns:
        st.error("❌ 'Demand Forecast' column missing — cannot continue.")
        st.stop()

    if 'Predicted_Demand' not in df.columns:
        df['Predicted_Demand'] = np.nan
        st.warning("⚠️ 'Predicted_Demand' column missing — only actual demand will be shown.")

    # --- Metrics
    total_rows = len(df)
    total_stores = df['Store ID'].nunique() if 'Store ID' in df.columns else 0
    total_products = df['Product ID'].nunique() if 'Product ID' in df.columns else 0
    if df['Predicted_Demand'].notna().any():
        mape = np.mean(np.abs((df['Demand Forecast'] - df['Predicted_Demand']) / (df['Demand Forecast'] + 1e-8))) * 100
    else:
        mape = 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric'><div class='label'>Total Rows</div><div class='value'>{total_rows:,}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><div class='label'>Stores</div><div class='value'>{total_stores}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><div class='label'>Products</div><div class='value'>{total_products}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric'><div class='label'>MAPE</div><div class='value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # --- Plot 1: Actual vs Predicted Demand
    st.markdown("<div class='glass plot-glow'><h3>📈 Actual vs Predicted Demand Over Time</h3></div>", unsafe_allow_html=True)
    fig1 = go.Figure()

    if df['Predicted_Demand'].notna().any():
        agg = df.groupby('Date')[['Demand Forecast', 'Predicted_Demand']].sum().reset_index()
        fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], mode='lines+markers',
                                  name='Actual Demand', line=dict(color='cyan', width=3)))
        fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Predicted_Demand'], mode='lines+markers',
                                  name='Predicted Demand', line=dict(color='magenta', width=3, dash='dash')))
    else:
        agg = df.groupby('Date')[['Demand Forecast']].sum().reset_index()
        fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], mode='lines+markers',
                                  name='Actual Demand', line=dict(color='cyan', width=3)))

    fig1.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       legend=dict(bgcolor='rgba(255,255,255,0.03)', font=dict(color='white')),
                       xaxis_title="Date", yaxis_title="Demand")
    st.plotly_chart(fig1, use_container_width=True)

    # --- Plot 2: Product-wise Average Error
    if df['Predicted_Demand'].notna().any():
        st.markdown("<div class='glass plot-glow'><h3>📊 Product-wise Average Prediction Error</h3></div>", unsafe_allow_html=True)
        df['Abs_Error'] = abs(df['Demand Forecast'] - df['Predicted_Demand'])
        prod_err = df.groupby('Product ID', dropna=True)['Abs_Error'].mean().reset_index().sort_values('Abs_Error', ascending=False)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=prod_err['Product ID'].astype(str), y=prod_err['Abs_Error'],
                              marker=dict(color='rgba(255,105,180,0.6)')))
        fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           xaxis_title="Product ID", yaxis_title="Average Error")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ℹ️ 'Predicted_Demand' column missing — skipping Product-wise error plot.")
