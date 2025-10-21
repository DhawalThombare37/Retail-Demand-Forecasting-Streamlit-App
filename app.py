import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------
# 🌈 PAGE CONFIGURATION
# ---------------------------------------------
st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")

# ---------------------------------------------
# 💎 STYLING - GLASSMORPHIC UI
# ---------------------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .glass {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
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
# 📤 FILE UPLOAD
# ---------------------------------------------
st.title("🔮 Demand Forecast Dashboard")
uploaded_file = st.file_uploader("📂 Upload your Forecast CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    if 'Predicted_Demand' not in df.columns:
        st.error("❌ 'Predicted_Demand' column missing in your file. Please include it.")
    else:
        # -------------------------------------------------
        # 🧮 BACKEND LOGIC (from 2nd code)
        # -------------------------------------------------
        df['APE'] = np.abs((df['Demand Forecast'] - df['Predicted_Demand']) / df['Demand Forecast']) * 100
        overall_mape = np.mean(df['APE'])

        # Grouping for charts
        agg = df.groupby('Date')[['Demand Forecast', 'Predicted_Demand']].sum().reset_index()
        product_mape = df.groupby('Product')['APE'].mean().reset_index()

        # -------------------------------------------------
        # 📊 METRIC CARDS SECTION
        # -------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='metric-card'><h2>{:.2f}%</h2><p>Overall MAPE</p></div>".format(overall_mape), unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'><h2>{}</h2><p>Total Products</p></div>".format(df['Product'].nunique()), unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'><h2>{}</h2><p>Total Records</p></div>".format(len(df)), unsafe_allow_html=True)
        with col4:
            st.markdown("<div class='metric-card'><h2>{}</h2><p>Date Range</p></div>".format(f"{df['Date'].min().date()} → {df['Date'].max().date()}"), unsafe_allow_html=True)

        # -------------------------------------------------
        # 🟩 Visualization 1: Actual vs Predicted Demand
        # -------------------------------------------------
        st.markdown("<div class='glass plot-glow'>Actual vs Predicted Demand</div>", unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=agg['Date'], y=agg['Demand Forecast'],
            mode='lines+markers', name='Actual Demand',
            line=dict(color='cyan', width=3), marker=dict(size=7)
        ))
        fig1.add_trace(go.Scatter(
            x=agg['Date'], y=agg['Predicted_Demand'],
            mode='lines+markers', name='Predicted Demand',
            line=dict(color='magenta', width=3, dash='dash'), marker=dict(size=7)
        ))
        fig1.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig1, use_container_width=True)

        # -------------------------------------------------
        # 🟦 Visualization 2: Product-wise MAPE (Line)
        # -------------------------------------------------
        st.markdown("<div class='glass plot-glow'>Product-wise MAPE (Error Trend)</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=product_mape['Product'], y=product_mape['APE'],
            mode='lines+markers', name='MAPE by Product',
            line=dict(color='lime', width=3), marker=dict(size=8)
        ))
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig2, use_container_width=True)

        # -------------------------------------------------
        # 💾 DOWNLOAD PROCESSED DATA
        # -------------------------------------------------
        st.markdown("<div class='glass plot-glow'>Download Processed Forecast Data</div>", unsafe_allow_html=True)
        csv_output = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Processed CSV",
            data=csv_output,
            file_name='processed_forecast.csv',
            mime='text/csv',
            key='download-csv',
        )

else:
    st.info("📁 Please upload your CSV file to begin.")
