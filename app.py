import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# =================== PAGE CONFIG ===================
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

# =================== CYBER-NEON STYLE ===================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background: radial-gradient(circle at 30% 10%, #050512 0%, #000010 100%);
    color: #E5E9FF;
    overflow-x: hidden;
}

/* --- Glowing header pulse --- */
.header-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #00E6FF;
    text-align: center;
    text-shadow: 0 0 12px rgba(0,230,255,0.8), 0 0 24px rgba(255,0,255,0.4);
    animation: pulse 3s ease-in-out infinite;
    margin-top: 20px;
}
@keyframes pulse {
    0% { text-shadow: 0 0 10px rgba(0,230,255,0.8), 0 0 30px rgba(255,0,255,0.4); }
    50% { text-shadow: 0 0 25px rgba(0,230,255,1), 0 0 55px rgba(255,0,255,0.6); }
    100% { text-shadow: 0 0 10px rgba(0,230,255,0.8), 0 0 30px rgba(255,0,255,0.4); }
}

/* --- Neon underline shimmer --- */
.header-underline {
    width: 220px;
    height: 3px;
    margin: 0 auto 25px auto;
    background: linear-gradient(90deg, #00E6FF, #FF00FF, #00E6FF);
    background-size: 300% 100%;
    border-radius: 3px;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}

/* --- Metric cards --- */
.metric {
    padding: 22px;
    border-radius: 14px;
    transition: all 0.25s ease;
    border: 1px solid rgba(0,255,255,0.1);
    background: linear-gradient(135deg, rgba(0,255,255,0.08), rgba(255,0,255,0.08));
    box-shadow: 0 0 22px rgba(0,255,255,0.15);
    text-align: center;
    margin-bottom: 25px;
}
.metric:hover {
    transform: translateY(-6px) scale(1.03);
    box-shadow: 0 0 35px rgba(0,255,255,0.35), 0 0 20px rgba(255,0,255,0.25);
}
.metric .label { color: rgba(255,255,255,0.75); font-size: 0.9rem; }
.metric .value { font-weight: 700; font-size: 1.6rem; margin-top: 6px; color: #00E6FF; }

/* --- File uploader --- */
.uploader {
    border: 1px dashed rgba(0,255,255,0.2);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    color: #E5E9FF;
    transition: all 0.25s;
    background: linear-gradient(145deg, rgba(0,255,255,0.06), rgba(255,0,255,0.05));
    margin-bottom: 35px;
}
.uploader:hover {
    background: linear-gradient(145deg, rgba(0,255,255,0.12), rgba(255,0,255,0.1));
    transform: translateY(-3px);
    box-shadow: 0 0 18px rgba(0,255,255,0.25);
}
.uploader strong { color: #00E6FF; font-weight: 700; }

/* --- Badge --- */
.badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(0,255,255,0.25), rgba(255,0,255,0.25));
    border: 1px solid rgba(0,255,255,0.1);
    color: white;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 0 18px rgba(0,255,255,0.25);
    backdrop-filter: blur(6px);
    margin-bottom: 15px;
}

/* --- Chart layout --- */
.chart-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 50px;
    margin-top: 50px;
}

/* --- Background neon glows --- */
.glow {
    position: fixed;
    width: 500px;
    height: 500px;
    filter: blur(100px);
    z-index: -1;
    opacity: 0.35;
}
.glow.cyan { background: radial-gradient(circle, rgba(0,255,255,0.6), transparent 70%); top: -150px; left: -150px; }
.glow.pink { background: radial-gradient(circle, rgba(255,0,255,0.5), transparent 70%); bottom: -150px; right: -150px; }

@media (max-width: 900px) {
    .metric .value { font-size: 1.2rem; }
}
</style>
""", unsafe_allow_html=True)

# Background glows
st.markdown("<div class='glow cyan'></div><div class='glow pink'></div>", unsafe_allow_html=True)

# =================== HEADER ===================
st.markdown("<div class='badge'>🛒 Transformer + XGBoost Ensemble</div>", unsafe_allow_html=True)
st.markdown("<div class='header-title'>Retail Demand Forecasting</div>", unsafe_allow_html=True)
st.markdown("<div class='header-underline'></div>", unsafe_allow_html=True)
st.caption("Modern cyber-neon interface for intelligent retail forecasting (Expected MAPE ≈ 3 %)")

# =================== FILE UPLOAD ===================
st.markdown("<div class='uploader'><strong>Upload your retail_store_inventory.csv</strong><br><span style='opacity:0.8;'>The model will predict demand for the last 3 months of data</span></div>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["csv"])

# =================== LOAD MODELS ===================
@st.cache_resource
def load_models():
    transformer = tf.keras.models.load_model("transformer_model.keras")
    xgb = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    training_cols = joblib.load("training_columns.pkl")
    xgb_cols = joblib.load("xgb_columns.pkl")
    seq_len = joblib.load("sequence_length.pkl")
    return transformer, xgb, scaler, training_cols, xgb_cols, seq_len

try:
    transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length = load_models()
except Exception as e:
    st.error("⚠️ Model files missing or corrupted. Please ensure all model assets are uploaded correctly.")
    st.stop()

# =================== PREDICTOR CLASS ===================
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
        df = df.fillna(0)
        features = [col for col in df.columns if col not in ['Date','Demand Forecast']]
        X = df[features]
        y = df['Demand Forecast']
        return X, y, df

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def predict(self, df_input):
        X, y, df_orig = self.preprocess(df_input)
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y.values)
        trans_preds = self.transformer.predict(X_seq, verbose=0)
        X_aligned = X.iloc[self.sequence_length:].copy()
        y_aligned = y.values[self.sequence_length:].copy()
        X_aligned['transformer_predictions_scaled'] = trans_preds.flatten()
        X_aligned = X_aligned.reindex(columns=self.xgb_columns, fill_value=0)
        final_preds = self.xgb.predict(X_aligned)
        df_results = df_orig.iloc[self.sequence_length:].copy()
        df_results['Predicted_Demand'] = final_preds
        mape = mean_absolute_percentage_error(y_aligned + 1e-8, final_preds) * 100
        return df_results, mape

# =================== APP LOGIC ===================
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(10), use_container_width=True)
    predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)
    with st.spinner("⚙️ Running Forecast..."):
        results, mape = predictor.predict(df)

    if results is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric'><div class='label'>MAPE</div><div class='value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric'><div class='label'>Predictions</div><div class='value'>{len(results):,}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric'><div class='label'>Accuracy</div><div class='value'>{100 - mape:.2f}%</div></div>", unsafe_allow_html=True)

        # === Charts Section ===
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(results['Date'], results['Demand Forecast'], label='Actual', color='#00E6FF', linewidth=2.2)
        ax1.plot(results['Date'], results['Predicted_Demand'], label='Predicted', color='#FF00FF', linewidth=2.2, linestyle='--')
        ax1.legend(facecolor='none', edgecolor='white', labelcolor='white')
        ax1.set_title("Actual vs Predicted Demand", color='white', fontsize=13, weight='bold')
        ax1.set_facecolor('none')
        for spine in ax1.spines.values(): spine.set_visible(False)
        fig1.patch.set_alpha(0)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        errors = results['Demand Forecast'] - results['Predicted_Demand']
        ax2.hist(errors, bins=20, color='#00E6FF', alpha=0.8)
        ax2.set_title("Error Distribution", color='white', fontsize=13, weight='bold')
        ax2.set_facecolor('none')
        fig2.patch.set_alpha(0)
        for spine in ax2.spines.values(): spine.set_visible(False)
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", use_container_width=True)

else:
    st.info("👆 Upload CSV to start predictions")
