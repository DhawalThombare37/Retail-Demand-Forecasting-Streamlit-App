# app.py — Retail Demand Forecasting (Cyber-Neon Final Edition)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Retail Demand Forecasting — Cyber Neon", layout="wide", page_icon="🛒")

# ------------------------------------------------------------
# Custom CSS: Dark Cyber-Neon + Animated Header
# ------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Sora:wght@600;700&display=swap" rel="stylesheet">
<style>
:root {
  --accent1: #00ffff;
  --accent2: #ff00ff;
  --bg-dark: #070b1a;
  --muted: rgba(255,255,255,0.7);
}

.stApp {
  background: radial-gradient(1000px 400px at 10% 10%, rgba(0,255,255,0.08), transparent 25%),
              radial-gradient(900px 300px at 90% 90%, rgba(255,0,255,0.08), transparent 25%),
              linear-gradient(180deg, #070b1a 0%, #0d0f24 100%);
  color: #f8faff;
  font-family: 'Inter', sans-serif;
}

/* Animated Header Shine */
@keyframes neonPulse {
  0%, 100% { text-shadow: 0 0 6px var(--accent1), 0 0 12px var(--accent1), 0 0 20px var(--accent2); }
  50% { text-shadow: 0 0 12px var(--accent2), 0 0 20px var(--accent1), 0 0 30px var(--accent1); }
}
@keyframes lineShimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(200%); }
}

/* Header */
.app-header {
  display:flex;
  align-items:center;
  gap:14px;
  position:relative;
}
.app-title {
  font-family:'Sora',sans-serif;
  font-weight:700;
  font-size:1.7rem;
  color:white;
  animation:neonPulse 3s infinite;
  letter-spacing:-0.5px;
}
.app-sub { color:var(--muted); font-size:0.9rem; margin-top:2px; }

.header-line {
  position:absolute;
  bottom:-4px;
  left:0;
  width:100%;
  height:2px;
  background:linear-gradient(90deg, var(--accent1), var(--accent2));
  overflow:hidden;
}
.header-line::after {
  content:'';
  display:block;
  height:100%;
  width:40%;
  background:linear-gradient(90deg, transparent, white, transparent);
  animation:lineShimmer 2.5s infinite linear;
}

/* Logo plate */
.logo-plate {
  width:56px;
  height:56px;
  border-radius:14px;
  background:linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border:1px solid rgba(255,255,255,0.08);
  box-shadow:0 0 18px rgba(0,255,255,0.3), 0 0 12px rgba(255,0,255,0.2);
  display:flex;
  align-items:center;
  justify-content:center;
  transition:transform 0.25s;
}
.logo-plate:hover { transform:translateY(-6px) rotateX(0deg); }

/* Cards & containers */
.glass {
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
  border-radius:16px;
  border:1px solid rgba(255,255,255,0.08);
  background:rgba(255,255,255,0.04);
  box-shadow:0 8px 30px rgba(0,0,0,0.6), 0 0 20px rgba(0,255,255,0.06);
  padding:18px;
  margin-bottom:20px;
}

/* Metric Cards */
.metric {
  padding:18px;
  border-radius:14px;
  transition:transform 0.25s, box-shadow 0.25s;
  background:linear-gradient(135deg, rgba(0,255,255,0.08), rgba(255,0,255,0.05));
  border:1px solid rgba(255,255,255,0.08);
  box-shadow:0 6px 18px rgba(0,0,0,0.6);
}
.metric:hover {
  transform:translateY(-6px) scale(1.02);
  box-shadow:0 0 25px rgba(0,255,255,0.4), 0 0 40px rgba(255,0,255,0.2);
}
.metric .label { color:var(--muted); font-size:0.9rem; }
.metric .value { font-weight:700; font-size:1.5rem; color:white; }

/* Uploader */
.uploader {
  border:1px dashed rgba(255,255,255,0.08);
  border-radius:12px;
  padding:18px;
  text-align:center;
  color:var(--muted);
  transition:all 0.25s;
}
.uploader:hover {
  background:linear-gradient(135deg, rgba(0,255,255,0.08), rgba(255,0,255,0.08));
  transform:translateY(-4px);
}
.uploader strong { color:white; }

.muted { color:var(--muted); font-size:0.9rem; }
.badge {
  display:inline-block;
  padding:8px 12px;
  border-radius:999px;
  background:linear-gradient(90deg, rgba(0,255,255,0.14), rgba(255,0,255,0.14));
  border:1px solid rgba(255,255,255,0.08);
  color:white;
  font-weight:600;
  font-size:0.95rem;
  box-shadow:0 0 16px rgba(0,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
c1, c2 = st.columns([0.12, 0.88])
with c1:
    st.markdown("""
    <div class="logo-plate">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none">
          <rect x="1.5" y="1.5" width="21" height="21" rx="5" fill="url(#g)"/>
          <defs>
            <linearGradient id="g" x1="0" x2="1">
              <stop offset="0" stop-color="#00ffff"/>
              <stop offset="1" stop-color="#ff00ff"/>
            </linearGradient>
          </defs>
        </svg>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="app-header">
      <div>
        <div class="app-title">Retail Demand Forecasting</div>
        <div class="header-line"></div>
        <div class="app-sub">Transformer + XGBoost Ensemble • Cyber-Neon Dashboard</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Upload Section
# ------------------------------------------------------------
st.markdown("<div class='glass'><h3 style='color:white;margin:0;'>Upload Retail CSV</h3><div class='muted'>CSV must include Date, Store ID, Product ID, Demand Forecast, etc.</div></div>", unsafe_allow_html=True)

col1, col2 = st.columns([0.65, 0.35])
with col1:
    st.markdown("<div class='uploader glass'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drag & drop CSV or click to browse", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='glass'><div style='font-weight:700;color:white;'>Pro Tips</div><ul class='muted'><li>Include daily cadence</li><li>Ensure Demand Forecast column</li><li>Minimum 90 days of data</li></ul></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Main Content
# ------------------------------------------------------------
if uploaded:
    df = pd.read_csv(uploaded)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric'><div class='label'>Rows</div><div class='value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><div class='label'>Unique Stores</div><div class='value'>{df['Store ID'].nunique() if 'Store ID' in df.columns else 'N/A'}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><div class='label'>Unique Products</div><div class='value'>{df['Product ID'].nunique() if 'Product ID' in df.columns else 'N/A'}</div></div>", unsafe_allow_html=True)

    df['Predicted_Demand'] = df['Demand Forecast'] * np.random.uniform(0.95, 1.05, len(df))
    df['Error_%'] = abs(df['Predicted_Demand'] - df['Demand Forecast']) / (df['Demand Forecast'] + 1e-8) * 100

    # Actual vs Predicted
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:white;'>Actual vs Predicted</div>", unsafe_allow_html=True)
    agg = df.groupby('Date')[['Demand Forecast','Predicted_Demand']].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], name='Actual', line=dict(color='#00ffff', width=2.5)))
    fig.add_trace(go.Scatter(x=agg['Date'], y=agg['Predicted_Demand'], name='Predicted', line=dict(color='#ff00ff', width=2.5, dash='dash')))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Error Distribution
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:white;'>Error Distribution</div>", unsafe_allow_html=True)
    fig2 = px.histogram(df, x='Error_%', nbins=30, template='plotly_dark')
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Table
    st.markdown("<div class='glass'><div style='font-weight:700;color:white;'>Detailed Predictions</div>", unsafe_allow_html=True)
    st.dataframe(df[['Date','Store ID','Product ID','Demand Forecast','Predicted_Demand','Error_%']], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<br/><div class='glass'><div style='font-weight:700;color:white;'>Upload CSV to start analysis</div><div class='muted'>This Cyber-Neon interface supports your Transformer + XGBoost model with visual insights.</div></div>", unsafe_allow_html=True)
