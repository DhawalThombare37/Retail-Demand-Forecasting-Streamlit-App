# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide", page_icon="🛒")

# -----------------------
# CSS: Glassmorphism + styling
# -----------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(1000px 400px at 10% 10%, rgba(142,68,173,0.12), transparent 8%),
                radial-gradient(900px 300px at 95% 90%, rgba(30,144,255,0.10), transparent 5%),
                linear-gradient(180deg, #0f1226 0%, #071028 100%);
    color: #e9eef8;
    font-family: 'Inter', sans-serif;
    min-height:100vh;
}
.glass {
    background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02));
    border-radius:16px;
    border:1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    padding:16px;
    margin-bottom:16px;
}
.stDataFrame table { border-radius:10px !important; overflow:hidden; }
.metric {
    padding:18px; border-radius:14px; border:1px solid rgba(255,255,255,0.04);
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    box-shadow: 0 6px 18px rgba(2,6,23,0.45); margin-bottom:12px; position: relative;
}
.metric .label { color: rgba(255,255,255,0.85); font-size:0.95rem; font-weight:600; }
.metric .value { font-weight:700; font-size:1.6rem; color:white; margin-top:6px; }
h2, h3 { color:white; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("<h2>🛒 Retail Demand Forecasting</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:rgba(255,255,255,0.6)'>Upload CSV containing columns: Date, Product_ID, Actual, Predicted</p>", unsafe_allow_html=True)

# -----------------------
# File uploader
# -----------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ['Date','Product_ID','Actual','Predicted']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {', '.join(required_cols)}")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        
        # -----------------------
        # Show uploaded data info
        # -----------------------
        st.markdown("<div class='glass'><strong>Uploaded Data Preview:</strong></div>", unsafe_allow_html=True)
        st.dataframe(df.head())

        st.markdown("<div class='glass'><strong>Data Info:</strong></div>", unsafe_allow_html=True)
        st.markdown(f"""
        - **Total rows uploaded:** {df.shape[0]}  
        - **Total unique products:** {df['Product_ID'].nunique()}  
        - **Columns in dataset:** {', '.join(df.columns)}  
        - **Target being predicted:** 'Predicted' column  
        """, unsafe_allow_html=True)

        # -----------------------
        # Metrics
        # -----------------------
        total_rows = df.shape[0]
        total_products = df['Product_ID'].nunique()
        overall_mape = np.mean(np.abs((df['Actual'] - df['Predicted'])/df['Actual'])) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric'><div class='label'>Total Rows</div><div class='value'>{total_rows}</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric'><div class='label'>Total Products</div><div class='value'>{total_products}</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric'><div class='label'>Overall MAPE (%)</div><div class='value'>{overall_mape:.2f}</div></div>", unsafe_allow_html=True)

        # -----------------------
        # 1. Actual vs Predicted (Line Chart)
        # -----------------------
        df_time = df.groupby('Date').agg({'Actual':'sum','Predicted':'sum'}).reset_index()
        fig_actual_pred = go.Figure()
        fig_actual_pred.add_trace(go.Scatter(
            x=df_time['Date'], y=df_time['Actual'], mode='lines+markers',
            name='Actual', line=dict(color='cyan', width=3), marker=dict(size=6)
        ))
        fig_actual_pred.add_trace(go.Scatter(
            x=df_time['Date'], y=df_time['Predicted'], mode='lines+markers',
            name='Predicted', line=dict(color='magenta', width=3), marker=dict(size=6)
        ))
        fig_actual_pred.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date",
            yaxis_title="Total Demand",
            legend=dict(font=dict(color='white'))
        )
        st.markdown("<div class='glass'><strong>Actual vs Predicted Demand Over Time</strong></div>", unsafe_allow_html=True)
        st.plotly_chart(fig_actual_pred, use_container_width=True)

        # -----------------------
        # 2. Product-wise Average Prediction Error (Line Chart)
        # -----------------------
        df['Abs_Error'] = np.abs(df['Actual'] - df['Predicted'])
        df_prod_error = df.groupby('Product_ID')['Abs_Error'].mean().reset_index().sort_values('Abs_Error', ascending=False)
        fig_prod_error = go.Figure()
        fig_prod_error.add_trace(go.Scatter(
            x=df_prod_error['Product_ID'], y=df_prod_error['Abs_Error'], mode='lines+markers',
            line=dict(color='orange', width=3), marker=dict(size=6)
        ))
        fig_prod_error.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Product ID",
            yaxis_title="Avg Prediction Error",
            xaxis_tickangle=-45
        )
        st.markdown("<div class='glass'><strong>Product-wise Average Prediction Error</strong></div>", unsafe_allow_html=True)
        st.plotly_chart(fig_prod_error, use_container_width=True)
