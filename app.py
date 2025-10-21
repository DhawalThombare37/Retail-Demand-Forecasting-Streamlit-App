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
# Glassmorphic + minimal CSS
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
/* page background */
.stApp {
    background: radial-gradient(1000px 400px at 10% 10%, rgba(142,68,173,0.12), transparent 8%),
                radial-gradient(900px 300px at 95% 90%, rgba(30,144,255,0.10), transparent 5%),
                linear-gradient(180deg, #0f1226 0%, #071028 100%);
    color: #e9eef8;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
}
/* metric card */
.metric { padding:18px; border-radius:14px; border:1px solid rgba(255,255,255,0.04);
         background: linear-gradient(135deg, rgba(255,255,255,0.022), rgba(255,255,255,0.01));
         box-shadow: 0 6px 18px rgba(2,6,23,0.45);}
.metric .label { color: rgba(255,255,255,0.75); font-size:0.9rem; }
.metric .value { font-weight:700; font-size:1.6rem; color: white; margin-top:6px; }
.glass-strong { background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.035));
                border-radius: 18px; border: 1px solid rgba(255,255,255,0.12);
                backdrop-filter: blur(calc(var(--glass-blur) + 6px)); padding:18px; margin-bottom:14px; }
.uploader { border: 1px dashed rgba(255,255,255,0.06); border-radius:12px;
           padding:18px; text-align:center; color:var(--muted);}
.uploader:hover { background: linear-gradient(135deg, rgba(30,144,255,0.02), rgba(142,68,173,0.02)); }
.muted { color:var(--muted); font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("<h2 style='color:white; margin-bottom:2px;'>Retail Demand Forecasting</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--muted); margin-top:-8px;'>Transformer + XGBoost Ensemble | Glassmorphic Dashboard</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -----------------------
# Model loader (unchanged)
# -----------------------
@st.cache_resource
def load_models():
    import os
    required_files = ['transformer_model.keras','xgb_model.pkl','scaler.pkl','training_columns.pkl','xgb_columns.pkl','sequence_length.pkl']
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing: raise FileNotFoundError(f"Missing files: {missing}")
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
    st.error("❌ Models not loaded.")
    st.exception(e)
    models_loaded = False

# -----------------------
# Predictor class (unchanged)
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
        df = df.sort_values('Date').reset_index(drop=True)
        df['year'] = df['Date'].dt.year; df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day; df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
        for col in ['Inventory Level','Units Sold','Units Ordered','Demand Forecast','Price']:
            if col in df.columns:
                df[f'{col}_lag_7'] = df.groupby(['Store ID','Product ID'])[col].shift(7)
                df[f'{col}_rolling_mean_7'] = df.groupby(['Store ID','Product ID'])[col].rolling(7).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_7'] = df.groupby(['Store ID','Product ID'])[col].rolling(7).std().reset_index(drop=True)
            else:
                df[f'{col}_lag_7'] = df[f'{col}_rolling_mean_7'] = df[f'{col}_rolling_std_7'] = 0
        df = df.fillna(0)
        features = [c for c in df.columns if c not in ['Date','Demand Forecast','Store ID','Product ID','Category','Region','Weather Condition','Seasonality']]
        X = df[features]
        y = df['Demand Forecast']
        for col in ['Discount','Holiday/Promotion']:
            if col in X.columns: X = pd.get_dummies(X, columns=[col])
        return X, y, df

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
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
        for col in self.training_columns:
            if col not in X.columns: X[col]=0
        X = X[self.training_columns]
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y.values)
        if len(X_seq)==0: raise ValueError(f"Need ≥{self.sequence_length+1} rows for sequences.")
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
st.markdown("<div class='glass-strong'><h4 style='color:white;'>Upload your Retail CSV</h4><p class='muted'>Must contain Date, Store ID, Product ID, Demand Forecast</p></div>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=['csv'])

if uploaded and models_loaded:
    df = pd.read_csv(uploaded)
    df['Date'] = pd.to_datetime(df['Date'])
    predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)
    results, mape = predictor.predict(df)

    # Summary cards
    col1, col2, col3, col4 = st.columns([0.22,0.22,0.28,0.28])
    col1.markdown(f"<div class='metric'><div class='label'>MAPE</div><div class='value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric'><div class='label'>Predictions</div><div class='value'>{len(results):,}</div></div>", unsafe_allow_html=True)
    accuracy = max(0,100-mape)
    col3.markdown(f"<div class='metric'><div class='label'>Accuracy</div><div class='value'>{accuracy:.1f}%</div></div>", unsafe_allow_html=True)
    first_date = results['Date'].min().date()
    last_date = results['Date'].max().date()
    col4.markdown(f"<div class='metric'><div class='label'>Prediction Range</div><div class='value'>{first_date} → {last_date}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------
    # Line chart: Actual vs Predicted
    # -----------------------
    fig = go.Figure()
    agg = results.groupby('Date')[['Demand Forecast','Predicted_Demand']].sum().reset_index()
    fig.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], mode='lines', name='Actual', line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=agg['Date'], y=agg['Predicted_Demand'], mode='lines', name='Predicted', line=dict(width=2.5,dash='dash')))
    fig.update_layout(template='plotly_dark', margin=dict(t=20,b=20,l=20,r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # Additional plots: Error histogram + top products
    # -----------------------
    results['Error_%'] = (abs(results['Demand Forecast']-results['Predicted_Demand'])/(results['Demand Forecast']+1e-8)*100).round(2)
    col1, col2 = st.columns(2)
    with col1:
        fig_err = px.histogram(results, x='Error_%', nbins=35, labels={'Error_%':'Error %'})
        fig_err.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_err, use_container_width=True)
    with col2:
        top_err = results.groupby('Product ID')['Error_%'].mean().nlargest(8).reset_index()
        fig_top = px.bar(top_err, x='Product ID', y='Error_%', labels={'Error_%':'Avg Error %'})
        fig_top.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_top, use_container_width=True)

    # -----------------------
    # Detailed results
    # -----------------------
    st.dataframe(results[['Date','Store ID','Product ID','Demand Forecast','Predicted_Demand','Error_%']].sort_values('Date',ascending=False), use_container_width=True)
    st.download_button("⬇️ Download Predictions (CSV)", results.to_csv(index=False).encode('utf-8'), "predictions.csv", use_container_width=True)

else:
    st.info("Upload CSV to visualize predictions")
