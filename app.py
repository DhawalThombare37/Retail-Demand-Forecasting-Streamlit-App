import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide", page_icon="🛒")

# -----------------------
# Glassmorphic + custom CSS (readable text)
# -----------------------
st.markdown("""
<style>
@keyframes gradientGlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.metric {
    padding:18px; border-radius:14px; border:1px solid rgba(255,255,255,0.04);
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    box-shadow: 0 6px 18px rgba(2,6,23,0.45);
    position: relative; overflow: hidden;
}
.metric .label { color: rgba(255,255,255,0.85); font-size:0.95rem; font-weight:600;}
.metric .value { font-weight:700; font-size:1.6rem; color:white; margin-top:6px; }

.metric::before {
    content:""; position:absolute; top:0; left:-50%; width:200%; height:100%;
    background: linear-gradient(270deg, rgba(30,144,255,0.3), rgba(142,68,173,0.3), rgba(44,230,183,0.3));
    background-size: 400% 400%;
    animation: gradientGlow 6s ease infinite;
    filter: blur(25px);
    z-index:0;
    pointer-events:none;
}

.metric > * { position: relative; z-index:1; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("<h2 style='color:white; margin-bottom:2px;'>Retail Demand Forecasting</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--muted); margin-top:-8px;'>Transformer + XGBoost Ensemble • Glassmorphic Dashboard</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -----------------------
# Uploader
# -----------------------
st.markdown("<div class='glass-strong'><h4 style='color:white;'>Upload your Retail CSV</h4><p class='muted'>Must contain Date, Store ID, Product ID, Demand Forecast</p></div>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    df['Date'] = pd.to_datetime(df['Date'])

    # -----------------------
    # Summary metrics
    # -----------------------
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric'><div class='label'>Total Rows</div><div class='value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric'><div class='label'>Unique Stores</div><div class='value'>{df['Store ID'].nunique() if 'Store ID' in df.columns else 'N/A'}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric'><div class='label'>Unique Products</div><div class='value'>{df['Product ID'].nunique() if 'Product ID' in df.columns else 'N/A'}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------
    # Predictions (placeholder logic, replace with your Predictor)
    # -----------------------
    # For demonstration, we simulate predictions
    df['Predicted_Demand'] = df['Demand Forecast'] * np.random.uniform(0.95,1.05,len(df))
    df['Error_%'] = (abs(df['Demand Forecast'] - df['Predicted_Demand'])/(df['Demand Forecast']+1e-8)*100).round(2)
    mape = df['Error_%'].mean()
    accuracy = 100 - mape

    # -----------------------
    # Prediction Summary cards
    # -----------------------
    col1, col2, col3, col4 = st.columns([0.22,0.22,0.28,0.28])
    col1.markdown(f"<div class='metric'><div class='label'>MAPE</div><div class='value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric'><div class='label'>Predictions</div><div class='value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric'><div class='label'>Accuracy</div><div class='value'>{accuracy:.1f}%</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric'><div class='label'>Date Range</div><div class='value'>{df['Date'].min().date()} → {df['Date'].max().date()}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------
    # Plot 1: Actual vs Predicted
    # -----------------------
    fig1 = go.Figure()
    agg = df.groupby('Date')[['Demand Forecast','Predicted_Demand']].sum().reset_index()
    fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], mode='lines', name='Actual', line=dict(width=2.5)))
    fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Predicted_Demand'], mode='lines', name='Predicted', line=dict(width=2.5,dash='dash')))
    fig1.update_layout(title="Actual vs Predicted Demand Over Time", template='plotly_dark', margin=dict(t=50))
    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------
    # Plot 2: Error Distribution
    # -----------------------
    fig2 = px.histogram(df, x='Error_%', nbins=35, title="Prediction Error Distribution", labels={'Error_%':'Error %'})
    fig2.update_layout(template='plotly_dark', margin=dict(t=50))
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------
    # Plot 3: Top products by average error
    # -----------------------
    top_err = df.groupby('Product ID')['Error_%'].mean().nlargest(10).reset_index()
    fig3 = px.bar(top_err, x='Product ID', y='Error_%', title="Top 10 Products by Avg Error (%)")
    fig3.update_layout(template='plotly_dark', margin=dict(t=50))
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------
    # Plot 4: Store-wise total demand
    # -----------------------
    store_totals = df.groupby('Store ID')[['Demand Forecast','Predicted_Demand']].sum().reset_index()
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=store_totals['Store ID'], y=store_totals['Demand Forecast'], name='Actual'))
    fig4.add_trace(go.Bar(x=store_totals['Store ID'], y=store_totals['Predicted_Demand'], name='Predicted'))
    fig4.update_layout(title="Store-wise Total Demand", template='plotly_dark', barmode='group', margin=dict(t=50))
    st.plotly_chart(fig4, use_container_width=True)

    # -----------------------
    # Plot 5: Product-wise average prediction error
    # -----------------------
    product_err = df.groupby('Product ID')['Error_%'].mean().reset_index()
    fig5 = px.line(product_err, x='Product ID', y='Error_%', title="Product-wise Average Prediction Error (%)", markers=True)
    fig5.update_layout(template='plotly_dark', margin=dict(t=50))
    st.plotly_chart(fig5, use_container_width=True)

    # -----------------------
    # Data table + download
    # -----------------------
    st.dataframe(df[['Date','Store ID','Product ID','Demand Forecast','Predicted_Demand','Error_%']].sort_values('Date',ascending=False), use_container_width=True)
    st.download_button("⬇️ Download Predictions (CSV)", df.to_csv(index=False).encode('utf-8'), "predictions.csv", use_container_width=True)
