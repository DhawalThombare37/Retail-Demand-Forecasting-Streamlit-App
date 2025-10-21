# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

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

/* Page background with animated glow circles */
.stApp {
    background: radial-gradient(1200px 500px at 15% 15%, rgba(30,144,255,0.14), transparent 15%),
                radial-gradient(1000px 400px at 80% 80%, rgba(142,68,173,0.12), transparent 15%),
                radial-gradient(600px 600px at 40% 70%, rgba(44,230,183,0.08), transparent 20%),
                linear-gradient(180deg, #0f1226 0%, #071028 100%);
    color: #e9eef8;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

/* Glass cards */
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

/* Metrics cards vibrant + glowing */
.metric {
    padding:20px;
    border-radius:16px;
    background: linear-gradient(135deg, rgba(30,144,255,0.12), rgba(142,68,173,0.1));
    border:1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 26px rgba(12,22,45,0.7);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    text-align:center;

    /* NEW */
    margin: 8px;           /* space between cards */
    flex: 1; 
}
.metric:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 18px 44px rgba(30,144,255,0.6);
}
.metric .label { color: var(--muted); font-size:0.95rem; font-weight:600; }
.metric .value { font-weight:700; font-size:1.6rem; margin-top:6px; color:#ffffff; text-shadow:0 1px 6px rgba(0,0,0,0.3); }

/* Headings */
h2, h3 { color:#ffffff; text-shadow:0 1px 6px rgba(0,0,0,0.4); }

/* File uploader */
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

/* Table */
.stDataFrame table {
    border-radius:12px !important;
    overflow:hidden;
}

/* Animated floating plot glow wrapper */
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

/* Download button styling */
.stDownloadButton>button {
    background: linear-gradient(90deg, rgba(30,144,255,0.95), rgba(142,68,173,0.9)) !important;
    color: #000 !important;
    font-weight: 700;
    border-radius: 10px;
    padding: 10px 16px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("<h2>🛒 Retail Demand Forecasting</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:rgba(255,255,255,0.75)'>Upload CSV containing: Date, Store ID, Product ID, Demand Forecast (Actual)</p>", unsafe_allow_html=True)

# -----------------------
# Model loading
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

transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length = load_models()

# -----------------------
# Predictor (exact 2.9% logic)
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
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            if col in df.columns:
                df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window, min_periods=1).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID','Product ID'])[col].rolling(window=rolling_window, min_periods=1).std().reset_index(drop=True).fillna(0)
        df = df.fillna(0)
        features = [col for col in df.columns if col not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        X = df[features]
        y = df['Demand Forecast']
        one_hot_cols = [c for c in ['Discount','Holiday/Promotion'] if c in X.columns]
        if one_hot_cols:
            X = pd.get_dummies(X, columns=one_hot_cols)
        return X, y, df
    
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X)-self.sequence_length):
            X_seq.append(X[i:(i+self.sequence_length)])
            y_seq.append(y[i+self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def predict(self, df_input):
        test_date = pd.to_datetime(df_input['Date']).max() - pd.DateOffset(months=3)
        X, y, df_orig = self.preprocess(df_input)
        df_orig['Date'] = pd.to_datetime(df_orig['Date'])
        test_mask = df_orig['Date'] > test_date
        X = X[test_mask].reset_index(drop=True)
        y = y[test_mask].reset_index(drop=True)
        df_orig = df_orig[test_mask].reset_index(drop=True)
        # Align columns
        for col in self.training_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.training_columns]
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y.values)
        if len(X_seq) == 0:
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
        y_safe[y_safe==0] = epsilon
        mape = mean_absolute_percentage_error(y_safe, final_preds)*100
        return df_results, mape

# -----------------------
# Upload / Run
# -----------------------
st.markdown("### 📁 Upload Data")
uploaded = st.file_uploader("retail_store_inventory.csv", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Metrics cards
        total_rows = len(df)
        total_stores = df['Store ID'].nunique() if 'Store ID' in df.columns else 0
        total_products = df['Product ID'].nunique() if 'Product ID' in df.columns else 0

        predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)
        with st.spinner("Predicting..."):
            results, mape = predictor.predict(df)
        
        if results is not None:
            # Metrics display
            st.markdown("<h3>🎯 Results</h3>", unsafe_allow_html=True)
           col1, col2, col3 = st.columns([1,1,1], gap="medium")
            with col1:
                st.markdown(f"""
                <div class="metric">
                    <div class="label">MAPE</div>
                    <div class="value">{mape:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric">
                    <div class="label">Predictions</div>
                    <div class="value">{len(results):,}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric">
                    <div class="label">Accuracy</div>
                    <div class="value">{100-mape:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            col4, col5, col6 = st.columns([1,1,1], gap="medium")
            with col4:
                st.markdown(f"""
                <div class="metric">
                    <div class="label">Total Rows</div>
                    <div class="value">{total_rows:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with col5:
                st.markdown(f"""
                <div class="metric">
                    <div class="label">Stores</div>
                    <div class="value">{total_stores}</div>
                </div>
                """, unsafe_allow_html=True)
            with col6:
                st.markdown(f"""
                <div class="metric">
                    <div class="label">Products</div>
                    <div class="value">{total_products}</div>
                </div>
                """, unsafe_allow_html=True)

            # Plot 1: Actual vs Predicted
            st.markdown("<div class='glass plot-glow'><h3>Actual vs Predicted Demand</h3></div>", unsafe_allow_html=True)
            agg = results.groupby('Date')[['Demand Forecast','Predicted_Demand']].sum().reset_index()
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], mode='lines+markers', name='Actual', line=dict(color='cyan', width=3), marker=dict(size=6)))
            fig1.add_trace(go.Scatter(x=agg['Date'], y=agg['Predicted_Demand'], mode='lines+markers', name='Predicted', line=dict(color='magenta', width=3, dash='dash'), marker=dict(size=6)))
            fig1.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               legend=dict(bgcolor='rgba(255,255,255,0.03)'))
            st.plotly_chart(fig1, use_container_width=True)

            # Plot 2: Product-wise MAPE (Top 20)
            st.markdown("<div class='glass plot-glow'><h3>Product-wise MAPE (Top 20)</h3></div>", unsafe_allow_html=True)
            results['Error_%'] = (abs(results['Demand Forecast']-results['Predicted_Demand'])/(results['Demand Forecast']+1e-8))*100
            prod_err = results.groupby('Product ID')['Error_%'].mean().reset_index().sort_values('Error_%', ascending=False).head(20)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=prod_err['Product ID'].astype(str), y=prod_err['Error_%'], mode='lines+markers', line=dict(color='orange', width=3), marker=dict(size=6)))
            fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

            # Display table
            display = results[['Date','Store ID','Product ID','Demand Forecast','Predicted_Demand']].copy()
            display['Error_%'] = (abs(display['Demand Forecast']-display['Predicted_Demand'])/(display['Demand Forecast']+1e-8)*100).round(2)
            st.dataframe(display.head(50), use_container_width=True)

            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv, "predictions.csv", use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👆 Upload CSV to see predictions")
