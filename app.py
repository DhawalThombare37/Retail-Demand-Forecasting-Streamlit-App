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
st.markdown(
    """
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
    /* header */
    header .decoration, header .css-18e3th9 {
        background: transparent;
    }
    .app-header {
        display:flex;
        align-items:center;
        gap:14px;
    }
    .logo-plate{
        width:56px;
        height:56px;
        border-radius:12px;
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 6px 18px rgba(12,19,40,0.6), 0 1px 0 rgba(255,255,255,0.02) inset;
        display:flex;
        align-items:center;
        justify-content:center;
        transform: perspective(800px) rotateX(6deg);
        transition: transform 0.25s ease;
    }
    .logo-plate:hover { transform: perspective(800px) rotateX(0deg) translateY(-6px); }
    .app-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.45rem;
        font-weight:700;
        letter-spacing: -0.5px;
        color: white;
    }
    .app-sub {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: -4px;
    }

    /* glass card */
    .glass {
        background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02));
        border-radius: 16px;
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(var(--glass-blur));
        -webkit-backdrop-filter: blur(var(--glass-blur));
        box-shadow: 0 8px 30px rgba(2,6,23,0.6);
    }
    .glass-strong {
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.035));
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(calc(var(--glass-blur) + 6px));
        padding:18px;
    }

    /* metric card */
    .metric {
        padding:18px;
        border-radius:14px;
        transition: transform 0.25s cubic-bezier(.2,.9,.2,1), box-shadow 0.25s;
        transform: translateZ(0);
        border:1px solid rgba(255,255,255,0.04);
        background: linear-gradient(135deg, rgba(255,255,255,0.022), rgba(255,255,255,0.01));
        box-shadow: 0 6px 18px rgba(2,6,23,0.45);
    }
    .metric:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 18px 40px rgba(2,6,23,0.6);
    }
    .metric .label { color: rgba(255,255,255,0.75); font-size:0.9rem; }
    .metric .value { font-weight:700; font-size:1.5rem; margin-top:6px; color: white; }

    /* uploader */
    .uploader {
        border: 1px dashed rgba(255,255,255,0.06);
        border-radius:12px;
        padding:18px;
        text-align:center;
        color:var(--muted);
        transition: background 0.25s, transform 0.2s;
    }
    .uploader:hover{
        background: linear-gradient(135deg, rgba(30,144,255,0.02), rgba(142,68,173,0.02));
        transform: translateY(-4px);
    }
    .uploader strong { color: white; font-weight:700; }

    /* small helper */
    .muted { color:var(--muted); font-size:0.9rem; }

    /* tiny animated shine for badges */
    .badge {
        display:inline-block;
        padding:8px 12px;
        border-radius:999px;
        background: linear-gradient(90deg, rgba(30,144,255,0.14), rgba(142,68,173,0.14));
        border: 1px solid rgba(255,255,255,0.04);
        color:white;
        font-weight:600;
        font-size:0.95rem;
        box-shadow: 0 6px 16px rgba(12, 22, 45, 0.5);
        backdrop-filter: blur(6px);
    }

    /* tiny tooltip helper (via title attributes) */
    [title] { cursor: help; }

    /* small table style fix */
    .stDataFrame table {
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* responsive tweaks */
    @media (max-width: 900px) {
        .app-title { font-size:1.15rem; }
    }

    /* subtle floating background glows */
    .glow {
        position: absolute;
        pointer-events: none;
        filter: blur(80px);
        opacity: 0.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Floating glows (pure decorative)
st.markdown(
    """
    <div style="position: absolute; right: -120px; top: 40px;">
        <div class="glow" style="width:420px;height:180px;background: radial-gradient(circle at 20% 30%, rgba(30,144,255,0.26), transparent 20%), radial-gradient(circle at 80% 70%, rgba(142,68,173,0.18), transparent 20%); border-radius: 50%;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Header area (logo + title + subtitle)
# -----------------------
header_col1, header_col2 = st.columns([0.12, 0.88])
with header_col1:
    st.markdown(
        """
        <div class="logo-plate glass">
            <!-- 3D-ish logo placeholder -->
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="1.5" y="1.5" width="21" height="21" rx="5" fill="url(#g)"/>
              <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#1E90FF"/><stop offset="1" stop-color="#8E44AD"/></linearGradient></defs>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    st.markdown('<div class="app-header"><div><div class="app-title">Retail Demand Forecasting</div><div class="app-sub">Transformer + XGBoost ensemble • Glassmorphic dashboard</div></div></div>', unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------
# Model loader (unchanged logic)
# -----------------------
@st.cache_resource
def load_models():
    import os
    required_files = {
        'transformer_model.keras': 'Transformer Model',
        'xgb_model.pkl': 'XGBoost Model',
        'scaler.pkl': 'Scaler',
        'training_columns.pkl': 'Training Columns',
        'xgb_columns.pkl': 'XGBoost Columns',
        'sequence_length.pkl': 'Sequence Length'
    }

    # quick check
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        # don't stop silently — show list for developer
        raise FileNotFoundError(f"Missing model files: {missing}")

    transformer = tf.keras.models.load_model("transformer_model.keras")
    xgb = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_cols = joblib.load("training_columns.pkl")
    xgb_cols = joblib.load("xgb_columns.pkl")
    seq_len = joblib.load("sequence_length.pkl")
    return transformer, xgb, scaler, training_cols, xgb_cols, seq_len

# Try loading and show a graceful message on failure
try:
    transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length = load_models()
    models_loaded = True
except Exception as e:
    st.error("❌ Model files not found or failed to load. Please ensure all files are present in the working directory.")
    st.exception(e)
    models_loaded = False

# -----------------------
# Predictor (keeps all logic intact)
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

        # time features
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
                df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
            else:
                df[f'{col}_rolling_mean_{rolling_window}'] = 0
                df[f'{col}_rolling_std_{rolling_window}'] = 0

        df = df.fillna(0)

        features = [c for c in df.columns if c not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        X = df[features].copy()
        y = df['Demand Forecast'].copy()

        # one-hot
        for col in ['Discount', 'Holiday/Promotion']:
            if col in X.columns:
                X = pd.get_dummies(X, columns=[col])
        return X, y, df

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def predict(self, df_input):
        test_date = pd.to_datetime(df_input['Date']).max() - pd.DateOffset(months=3)
        X, y, df_orig = self.preprocess(df_input)
        df_orig['Date'] = pd.to_datetime(df_orig['Date'])
        test_mask = df_orig['Date'] > test_date

        X = X[test_mask].reset_index(drop=True)
        y = y[test_mask].reset_index(drop=True)
        df_orig = df_orig[test_mask].reset_index(drop=True)

        # Align training columns
        for col in self.training_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.training_columns]

        # Scale
        X_scaled = self.scaler.transform(X)

        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y.values)
        if len(X_seq) == 0:
            raise ValueError(f"Need at least {self.sequence_length + 1} rows in the test period to create sequences.")

        # Transformer preds
        trans_preds = self.transformer.predict(X_seq, verbose=0)

        # XGBoost alignment
        X_aligned = X.iloc[self.sequence_length:].copy().reset_index(drop=True)
        y_aligned = y.values[self.sequence_length:].copy()
        df_aligned = df_orig.iloc[self.sequence_length:].copy().reset_index(drop=True)

        X_aligned['transformer_predictions_scaled'] = trans_preds.flatten()

        for col in self.xgb_columns:
            if col not in X_aligned.columns:
                X_aligned[col] = 0
        X_aligned = X_aligned[self.xgb_columns]

        final_preds = self.xgb.predict(X_aligned)

        df_results = df_aligned.copy()
        df_results['Predicted_Demand'] = final_preds

        epsilon = 1e-8
        y_safe = y_aligned.copy()
        y_safe[y_safe == 0] = epsilon
        mape = mean_absolute_percentage_error(y_safe, final_preds) * 100

        return df_results.reset_index(drop=True), mape

# -----------------------
# Uploader and UI flow
# -----------------------
st.markdown("<div class='glass-strong'><div style='display:flex;justify-content:space-between;align-items:center'>"
            "<div><h3 style='margin:0;color:white;'>Upload your Retail CSV</h3><div class='muted'>CSV should contain Date, Store ID, Product ID, Demand Forecast, etc.</div></div>"
            "<div><span class='badge' title='Expected MAPE target'>Target: ~3% MAPE</span></div>"
            "</div></div>",
            unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns([0.65, 0.35])

with upload_col1:
    st.markdown("<div class='uploader glass'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drag & drop CSV or click to browse", type=["csv"], help="Upload the retail_store_inventory.csv used for predictions.")
    st.markdown("</div>", unsafe_allow_html=True)

with upload_col2:
    # Quick example / tips card
    st.markdown(
        """
        <div class="glass" style="padding:16px;">
            <div style="font-weight:700;color:white;">Pro Tips</div>
            <ul style="color:var(--muted);margin-top:8px;">
                <li>Include daily cadence for best results</li>
                <li>Keep 'Demand Forecast' as your true target column</li>
                <li>Ensure at least 90 days + 1 sequence for predictions</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# If no models loaded, provide a stop message but keep UI graceful
if not models_loaded:
    st.stop()

# If a file was uploaded, process it
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Failed to read CSV. Please ensure it is a valid UTF-8 CSV and contains header row.")
        st.exception(e)
        st.stop()

    # basic datetime handling
    if 'Date' not in df.columns:
        st.error("CSV must contain a 'Date' column.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    # header summary row
    summary_col1, summary_col2, summary_col3 = st.columns([0.33,0.33,0.34])
    summary_col1.markdown(f"<div class='metric'><div class='label'>Total Rows</div><div class='value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    summary_col2.markdown(f"<div class='metric'><div class='label'>Unique Stores</div><div class='value'>{df['Store ID'].nunique() if 'Store ID' in df.columns else 'N/A'}</div></div>", unsafe_allow_html=True)
    summary_col3.markdown(f"<div class='metric'><div class='label'>Unique Products</div><div class='value'>{df['Product ID'].nunique() if 'Product ID' in df.columns else 'N/A'}</div></div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Predict
    predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)

    with st.spinner("🌊 Running preprocessing and generating predictions..."):
        try:
            results, mape = predictor.predict(df)
        except Exception as e:
            st.error("Prediction failed — likely insufficient test-period rows or mismatched columns.")
            st.exception(e)
            st.stop()

    # Top summary cards with glass + 3D style
    cards_col1, cards_col2, cards_col3, cards_col4 = st.columns([0.22,0.22,0.28,0.28])
    cards_col1.markdown(f"<div class='metric'><div class='label'>MAPE</div><div class='value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
    cards_col2.markdown(f"<div class='metric'><div class='label'>Predictions</div><div class='value'>{len(results):,}</div></div>", unsafe_allow_html=True)
    # accuracy (derived)
    accuracy = max(0.0, 100.0 - mape)
    cards_col3.markdown(f"<div class='metric'><div class='label'>Accuracy</div><div class='value'>{accuracy:.2f}%</div></div>", unsafe_allow_html=True)
    # date range card
    first_date = results['Date'].min().date() if 'Date' in results.columns else 'N/A'
    last_date = results['Date'].max().date() if 'Date' in results.columns else 'N/A'
    cards_col4.markdown(f"<div class='metric'><div class='label'>Prediction Range</div><div class='value'>{first_date} → {last_date}</div></div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------
    # Visualizations: multi-graph area
    # -------------------------
    viz1, viz2 = st.columns([0.6, 0.4])

    # prepare results for charts
    df_viz = results.copy()
    df_viz['Date'] = pd.to_datetime(df_viz['Date'])
    # error column
    df_viz['Error'] = (df_viz['Predicted_Demand'] - df_viz['Demand Forecast']).abs()
    df_viz['Error_%'] = (df_viz['Error'] / (df_viz['Demand Forecast'] + 1e-8)) * 100

    # Smoothed actual vs predicted line with area + glow
    with viz1:
        st.markdown("<div class='glass' style='padding:18px'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><div style='font-weight:700;color:white;'>Actual vs Predicted</div><div class='muted'>Interactive • Hover to inspect</div></div>", unsafe_allow_html=True)
        fig = go.Figure()
        # aggregated by date
        agg = df_viz.groupby('Date')[['Demand Forecast', 'Predicted_Demand']].sum().reset_index()
        fig.add_trace(go.Scatter(x=agg['Date'], y=agg['Demand Forecast'], name='Actual', mode='lines', line=dict(width=2.5), hovertemplate='%{x|%Y-%m-%d}<br>Actual: %{y:.2f}'))
        fig.add_trace(go.Scatter(x=agg['Date'], y=agg['Predicted_Demand'], name='Predicted', mode='lines', line=dict(width=2.5, dash='dash'), hovertemplate='%{x|%Y-%m-%d}<br>Predicted: %{y:.2f}'))
        # area fill between
        fig.add_trace(go.Scatter(x=agg['Date'].tolist() + agg['Date'].tolist()[::-1],
                                 y=(agg['Predicted_Demand'] + agg['Demand Forecast']).tolist() + (agg['Predicted_Demand'] - agg['Demand Forecast']).tolist()[::-1],
                                 fill='toself', fillcolor='rgba(30,144,255,0.06)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(bgcolor='rgba(255,255,255,0.03)')
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Right column: error histogram + top products
    with viz2:
        st.markdown("<div class='glass' style='padding:18px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;color:white;'>Error Distribution</div>", unsafe_allow_html=True)
        # Histogram of Error %
        fig_h = px.histogram(df_viz, x='Error_%', nbins=35, title="", labels={'Error_%':'Error %'})
        fig_h.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.04); margin: 10px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;color:white;margin-bottom:8px'>Top products by avg error</div>", unsafe_allow_html=True)
        top_err = df_viz.groupby('Product ID')['Error_%'].mean().nlargest(8).reset_index()
        if top_err.empty:
            st.info("No product-level data available.")
        else:
            fig_p = px.bar(top_err, x='Product ID', y='Error_%', labels={'Error_%':'Avg Error %'}, title="")
            fig_p.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=5,b=5,l=5,r=5), showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------
    # Heatmap: store x product error (sample top N to keep it readable)
    # -------------------------
    st.markdown("<div class='glass' style='padding:16px;margin-bottom:18px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:white;margin-bottom:8px'>Product x Store Error Heatmap (sample)</div>", unsafe_allow_html=True)
    df_heat = df_viz.copy()
    # aggregate
    heat_agg = df_heat.groupby(['Store ID', 'Product ID'])['Error_%'].mean().reset_index()
    # sample top stores & products
    top_stores = heat_agg.groupby('Store ID')['Error_%'].mean().nlargest(6).index.tolist()
    top_products = heat_agg.groupby('Product ID')['Error_%'].mean().nlargest(12).index.tolist()
    heat_small = heat_agg[heat_agg['Store ID'].isin(top_stores) & heat_agg['Product ID'].isin(top_products)]
    if not heat_small.empty:
        heat_pivot = heat_small.pivot_table(index='Store ID', columns='Product ID', values='Error_%', fill_value=0)
        fig_heat = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=[str(x) for x in heat_pivot.columns],
            y=[str(y) for y in heat_pivot.index],
            colorscale='Viridis',
            colorbar=dict(title='Avg Error %')
        ))
        fig_heat.update_layout(template='plotly_dark', margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough data for heatmap sample (requires store & product columns).")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # Detailed results table with subtle 3D card
    # -------------------------
    st.markdown("<div class='glass' style='padding:16px;margin-bottom:18px;'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><div style='font-weight:700;color:white;'>Detailed Predictions</div><div class='muted'>Click header to sort • Use download below</div></div>", unsafe_allow_html=True)
    display = results[['Date','Store ID','Product ID','Demand Forecast','Predicted_Demand']].copy()
    display['Error_%'] = (abs(display['Demand Forecast'] - display['Predicted_Demand']) / (display['Demand Forecast'] + 1e-8) * 100).round(2)
    st.dataframe(display.sort_values('Date', ascending=False).reset_index(drop=True), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Download button (preserve full CSV)
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download full predictions (CSV)", csv, "predictions.csv", use_container_width=True)

    # small footer note
    st.markdown("<div class='muted' style='margin-top:8px;'>Models used: Transformer (sequence input) + XGBoost ensemble. All backend prediction logic unchanged.</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='glass' style='padding:26px;margin-top:18px;'><div style='font-weight:700;color:white;font-size:1.05rem;'>Upload CSV to visualize predictions</div><div class='muted' style='margin-top:8px;'>This demo preserves your transformer + XGBoost pipeline unchanged and displays rich glassmorphic visuals.</div></div>", unsafe_allow_html=True)

# End of file
