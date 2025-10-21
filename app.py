import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ultra-modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background - Dark theme with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Animated background particles effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(88, 166, 255, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(255, 119, 190, 0.3) 0%, transparent 50%);
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes float {
        0%, 100% { opacity: 0.5; transform: translateY(0px); }
        50% { opacity: 0.8; transform: translateY(-20px); }
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        position: relative;
        z-index: 1;
    }
    
    /* Header with 3D effect */
    .main-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        padding: 2.5rem;
        border-radius: 25px;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 
            0 8px 32px 0 rgba(31, 38, 135, 0.37),
            inset 0 1px 0 0 rgba(255,255,255,0.1);
        transform: perspective(1000px) rotateX(2deg);
        transition: all 0.3s ease;
    }
    
    .main-header:hover {
        transform: perspective(1000px) rotateX(0deg) translateY(-5px);
        box-shadow: 
            0 15px 45px 0 rgba(31, 38, 135, 0.5),
            inset 0 1px 0 0 rgba(255,255,255,0.1);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        text-align: center;
        animation: gradient 3s ease infinite;
        letter-spacing: -1px;
    }
    
    @keyframes gradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .subtitle {
        color: rgba(255,255,255,0.8);
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.8rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Glassmorphic cards with 3D depth */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.06) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 
            0 8px 32px 0 rgba(31, 38, 135, 0.37),
            inset 0 1px 0 0 rgba(255,255,255,0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 
            0 20px 60px 0 rgba(31, 38, 135, 0.6),
            inset 0 1px 0 0 rgba(255,255,255,0.2);
        border-color: rgba(255,255,255,0.3);
    }
    
    /* Upload section with pulse animation */
    .upload-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(20px);
        padding: 3rem;
        border-radius: 25px;
        border: 2px dashed rgba(255,255,255,0.3);
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.1);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .upload-section:hover::before {
        width: 500px;
        height: 500px;
    }
    
    .upload-section:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
        transform: scale(1.02);
    }
    
    /* Metric styling with glow */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255,255,255,0.5);
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetricDelta"] {
        color: rgba(255,255,255,0.7) !important;
    }
    
    /* Button with 3D effect */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 3rem !important;
        border-radius: 15px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stDownloadButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stDownloadButton button:hover::before {
        left: 100%;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.18);
        overflow: hidden;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Section headers */
    h3 {
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        margin-bottom: 1.5rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: linear-gradient(135deg, #ffffff 0%, #a0aec0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Plotly charts container */
    .js-plotly-plot {
        border-radius: 20px;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load models silently
@st.cache_resource
def load_models():
    try:
        transformer = tf.keras.models.load_model("transformer_model.keras", compile=False)
        xgb = joblib.load("xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        training_cols = joblib.load("training_columns.pkl")
        xgb_cols = joblib.load("xgb_columns.pkl")
        seq_len = joblib.load("sequence_length.pkl")
        return transformer, xgb, scaler, training_cols, xgb_cols, seq_len
    except:
        return None, None, None, None, None, None

transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length = load_models()

# Predictor class
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
        
        features = [col for col in df.columns if col not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        X = df[features]
        y = df['Demand Forecast']
        
        X = pd.get_dummies(X, columns=['Discount', 'Holiday/Promotion'])
        
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
            return None, None, None
        
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
        
        return df_results, mape, test_date

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🛒 AI Demand Forecasting</h1>
    <p class="subtitle">Next-Generation Predictions • Transformer × XGBoost Architecture</p>
</div>
""", unsafe_allow_html=True)

if transformer_model is None:
    st.error("⚠️ System Error: Models not loaded")
    st.stop()

# File upload
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded = st.file_uploader("📂 Drop your CSV here or click to browse", type=["csv"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded:
    with st.spinner("🔮 AI is analyzing your data..."):
        df = pd.read_csv(uploaded)
        predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)
        results, mape, test_date = predictor.predict(df)
    
    if results is not None:
        # Metrics
        st.markdown("### 📊 Performance Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🎯 MAPE",
                value=f"{mape:.2f}%",
                delta="Excellent" if mape <= 5 else "Good" if mape <= 10 else "Review",
                delta_color="normal" if mape <= 5 else "off"
            )
        
        with col2:
            st.metric(
                label="📈 Predictions",
                value=f"{len(results):,}"
            )
        
        with col3:
            accuracy = max(0, 100 - mape)
            st.metric(
                label="✓ Accuracy",
                value=f"{accuracy:.1f}%"
            )
        
        with col4:
            avg_demand = results['Demand Forecast'].mean()
            st.metric(
                label="📦 Avg Demand",
                value=f"{avg_demand:.0f}"
            )
        
        st.markdown("---")
        
        # Main visualization - Time series with connected lines
        st.markdown("### 📈 Demand Forecast Timeline")
        
        # Use sample for better visibility
        sample_size = min(200, len(results))
        chart_data = results.head(sample_size).copy()
        chart_data['Date'] = pd.to_datetime(chart_data['Date'])
        
        fig = go.Figure()
        
        # Actual demand with markers
        fig.add_trace(go.Scatter(
            x=chart_data['Date'],
            y=chart_data['Demand Forecast'],
            name='Actual Demand',
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, symbol='circle', line=dict(width=2, color='white')),
            hovertemplate='<b>Actual</b><br>Date: %{x}<br>Demand: %{y:.2f}<extra></extra>'
        ))
        
        # Predicted demand with markers
        fig.add_trace(go.Scatter(
            x=chart_data['Date'],
            y=chart_data['Predicted_Demand'],
            name='Predicted Demand',
            mode='lines+markers',
            line=dict(color='#f093fb', width=3, dash='dot'),
            marker=dict(size=8, symbol='diamond', line=dict(width=2, color='white')),
            hovertemplate='<b>Predicted</b><br>Date: %{x}<br>Demand: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                title='Date',
                title_font=dict(size=14, color='rgba(255,255,255,0.8)')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                title='Demand',
                title_font=dict(size=14, color='rgba(255,255,255,0.8)')
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0.3)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Multi-plot section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Prediction Accuracy Distribution")
            
            # Error distribution histogram
            errors = results['Demand Forecast'] - results['Predicted_Demand']
            
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=errors,
                nbinsx=50,
                marker=dict(
                    color=errors,
                    colorscale='RdYlGn_r',
                    line=dict(width=1, color='white')
                ),
                hovertemplate='Error: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ))
            
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title='Prediction Error'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title='Frequency'
                ),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.markdown("### 🔝 Top 10 Products by Error")
            
            error_data = results.copy()
            error_data['Absolute_Error'] = abs(error_data['Demand Forecast'] - error_data['Predicted_Demand'])
            top_errors = error_data.nlargest(10, 'Absolute_Error')[['Product ID', 'Absolute_Error']]
            
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=top_errors['Absolute_Error'],
                y=top_errors['Product ID'].astype(str),
                orientation='h',
                marker=dict(
                    color=top_errors['Absolute_Error'],
                    colorscale='Plasma',
                    line=dict(width=1, color='white')
                ),
                hovertemplate='Product: %{y}<br>Error: %{x:.2f}<extra></extra>'
            ))
            
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title='Absolute Error'
                ),
                yaxis=dict(
                    showgrid=False,
                    title='Product ID'
                ),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # 3D Scatter Plot
        st.markdown("### 🎨 3D Prediction Analysis")
        
        sample_3d = results.sample(min(500, len(results))).copy()
        sample_3d['Error'] = abs(sample_3d['Demand Forecast'] - sample_3d['Predicted_Demand'])
        
        fig4 = go.Figure(data=[go.Scatter3d(
            x=sample_3d['Demand Forecast'],
            y=sample_3d['Predicted_Demand'],
            z=sample_3d['Error'],
            mode='markers',
            marker=dict(
                size=5,
                color=sample_3d['Error'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Error", titleside="right", tickmode="linear"),
                line=dict(width=0.5, color='white')
            ),
            hovertemplate='<b>Actual:</b> %{x:.2f}<br><b>Predicted:</b> %{y:.2f}<br><b>Error:</b> %{z:.2f}<extra></extra>'
        )])
        
        fig4.update_layout(
            scene=dict(
                xaxis=dict(
                    title='Actual Demand',
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(255,255,255,0.1)',
                    showbackground=True
                ),
                yaxis=dict(
                    title='Predicted Demand',
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(255,255,255,0.1)',
                    showbackground=True
                ),
                zaxis=dict(
                    title='Error',
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(255,255,255,0.1)',
                    showbackground=True
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("---")
        
        # Correlation heatmap
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 📊 Prediction vs Actual Scatter")
            
            fig5 = go.Figure()
            
            # Perfect prediction line
            max_val = max(results['Demand Forecast'].max(), results['Predicted_Demand'].max())
            min_val = min(results['Demand Forecast'].min(), results['Predicted_Demand'].min())
            
            fig5.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'),
                hoverinfo='skip'
            ))
            
            # Actual scatter
            fig5.add_trace(go.Scatter(
                x=results['Demand Forecast'],
                y=results['Predicted_Demand'],
                mode='markers',
                name='Predictions',
                marker=dict(
                    size=6,
                    color=results['Predicted_Demand'],
                    colorscale='Turbo',
                    showscale=True,
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
            ))
            
            fig5.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title='Actual Demand'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title='Predicted Demand'
                ),
                height=500,
                legend=dict(bgcolor='rgba(0,0,0,0.3)')
            )
            
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Error Statistics")
            
            # Calculate statistics
            errors = results['Demand Forecast'] - results['Predicted_
