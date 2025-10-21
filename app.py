import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error

# ---------------------------- APP CONFIG ----------------------------
st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="expanded"
)

# ---------------------------- CUSTOM CSS ----------------------------
st.markdown("""
<style>
/* Global */
body {
    font-family: 'Poppins', sans-serif;
}

/* Header */
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #2E86DE;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.sub-title {
    color: #7f8c8d;
    font-size: 1.1rem;
    margin-bottom: 25px;
}

/* Card Styling */
.metric-card {
    background: linear-gradient(145deg, #f2f4f7, #ffffff);
    border-radius: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    padding: 20px;
    transition: all 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 18px rgba(0,0,0,0.12);
}

/* Upload box */
.stFileUploader {
    border: 2px dashed #90CAF9 !important;
    border-radius: 15px !important;
    background: rgba(144,202,249,0.1) !important;
    padding: 2rem !important;
}
.stFileUploader:hover {
    background: rgba(144,202,249,0.15) !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #74b9ff, #0984e3);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-size: 1rem;
    font-weight: 600;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    transition: 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #0984e3, #74b9ff);
    transform: translateY(-2px);
}

/* Tables */
.dataframe {
    border-radius: 10px !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.05);
}

/* Chart Area */
.chart-container {
    background: linear-gradient(145deg, #fafbfc, #ffffff);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.07);
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------- HEADER ----------------------------
st.markdown("<div class='main-title'>🛒 Retail Demand Forecasting</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Transformer + XGBoost Ensemble | Expected MAPE: ~3%</div>", unsafe_allow_html=True)

# ---------------------------- MODEL LOADING ----------------------------
@st.cache_resource
def load_models():
    import os
    st.info("🔍 Checking for model files...")

    required_files = {
        'transformer_model.keras': 'Transformer Model',
        'xgb_model.pkl': 'XGBoost Model',
        'scaler.pkl': 'Scaler',
        'training_columns.pkl': 'Training Columns',
        'xgb_columns.pkl': 'XGBoost Columns',
        'sequence_length.pkl': 'Sequence Length'
    }

    missing_files = []
    for file, name in required_files.items():
        if not os.path.exists(file):
            st.error(f"❌ {name}: {file} not found")
            missing_files.append(file)
    if missing_files:
        st.stop()

    transformer = tf.keras.models.load_model("transformer_model.keras")
    xgb = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_cols = joblib.load("training_columns.pkl")
    xgb_cols = joblib.load("xgb_columns.pkl")
    seq_len = joblib.load("sequence_length.pkl")
    return transformer, xgb, scaler, training_cols, xgb_cols, seq_len

transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length = load_models()

# ---------------------------- PREDICTOR CLASS ----------------------------
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
            df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)

        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
            df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)

        df = df.fillna(0)
        features = [col for col in df.columns if col not in ['Date','Demand Forecast','Store ID','Product ID','Category','Region','Weather Condition','Seasonality']]
        X = df[features]
        y = df['Demand Forecast']
        X = pd.get_dummies(X, columns=['Discount','Holiday/Promotion'])
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
        for col in self.training_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.training_columns]
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y.values)
        if len(X_seq) == 0:
            st.error("Not enough rows for sequence creation")
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
        y_safe[y_safe == 0] = epsilon
        mape = mean_absolute_percentage_error(y_safe, final_preds) * 100
        return df_results, mape

# ---------------------------- FILE UPLOAD ----------------------------
st.markdown("### 📂 Upload Retail Data")
uploaded = st.file_uploader("Upload a CSV file (retail_store_inventory.csv)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df['Date'] = pd.to_datetime(df['Date'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Rows", f"{len(df):,}")
    with col2:
        st.metric("🏬 Stores", df['Store ID'].nunique())
    with col3:
        st.metric("📦 Products", df['Product ID'].nunique())

    test_date = df['Date'].max() - pd.DateOffset(months=3)
    st.info(f"Predicting for last 3 months: from {test_date.date()} onwards")

    predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)
    with st.spinner("⚙️ Generating predictions..."):
        results, mape = predictor.predict(df)

    if results is not None:
        st.markdown("---")
        st.subheader("🎯 Prediction Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>MAPE</h3><h2 style='color:#0984e3;'>{mape:.2f}%</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>Predictions</h3><h2>{len(results):,}</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3>Accuracy</h3><h2 style='color:#00b894;'>{max(0,100-mape):.2f}%</h2></div>", unsafe_allow_html=True)

        # ---------------------------- VISUALIZATIONS ----------------------------
        st.markdown("### 📈 Visualization Dashboard")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            fig1 = px.line(results, x='Date', y=['Demand Forecast','Predicted_Demand'],
                           labels={'value':'Demand','variable':'Legend'},
                           title="📊 Actual vs Predicted Demand Over Time")
            fig1.update_layout(hovermode='x unified', template='plotly_white', title_font=dict(size=18))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_col2:
            results['Error_%'] = abs(results['Demand Forecast'] - results['Predicted_Demand']) / (results['Demand Forecast'] + 1e-8) * 100
            top_errors = results.groupby('Product ID')['Error_%'].mean().nlargest(10).reset_index()
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            fig2 = px.bar(top_errors, x='Product ID', y='Error_%', title="🔥 Top 10 Products with Highest Prediction Error",
                          color='Error_%', color_continuous_scale='Reds')
            fig2.update_layout(template='plotly_white', title_font=dict(size=18))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------- DATA TABLE ----------------------------
        st.markdown("### 📋 Detailed Results")
        display = results[['Date','Store ID','Product ID','Demand Forecast','Predicted_Demand']].copy()
        display['Error_%'] = (abs(display['Demand Forecast'] - display['Predicted_Demand']) / (display['Demand Forecast'] + 1e-8) * 100).round(2)
        st.dataframe(display.head(100), use_container_width=True)

        # ---------------------------- DOWNLOAD ----------------------------
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Full Predictions", csv, "predictions.csv", use_container_width=True)
else:
    st.info("👆 Upload a CSV to start predictions")
