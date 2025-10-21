import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-align: center;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        border: 2px dashed rgba(255,255,255,0.3);
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: rgba(255,255,255,0.6);
        background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1));
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Metric value styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
    }
    
    /* Button styling */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
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
    <h1 class="main-title">🛒 Retail Demand Forecasting</h1>
    <p class="subtitle">AI-Powered Predictions using Transformer + XGBoost Ensemble</p>
</div>
""", unsafe_allow_html=True)

if transformer_model is None:
    st.error("⚠️ Models not loaded. Please check configuration.")
    st.stop()

# File upload
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded = st.file_uploader("📂 Upload your retail inventory CSV", type=["csv"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded:
    with st.spinner("🔮 Processing your data..."):
        df = pd.read_csv(uploaded)
        predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)
        results, mape, test_date = predictor.predict(df)
    
    if results is not None:
        # Metrics row
        st.markdown("### 📊 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🎯 MAPE Score",
                value=f"{mape:.2f}%",
                delta="Excellent" if mape <= 5 else "Good" if mape <= 10 else "Review"
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
            st.metric(
                label="📅 Test Period",
                value=f"{(pd.to_datetime(results['Date']).max() - pd.to_datetime(results['Date']).min()).days} days"
            )
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 Actual vs Predicted Demand")
            
            # Prepare data for chart
            chart_data = results.head(100).copy()
            chart_data['Date'] = pd.to_datetime(chart_data['Date'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=chart_data['Date'],
                y=chart_data['Demand Forecast'],
                name='Actual Demand',
                line=dict(color='#667eea', width=3),
                mode='lines+markers'
            ))
            
            fig.add_trace(go.Scatter(
                x=chart_data['Date'],
                y=chart_data['Predicted_Demand'],
                name='Predicted Demand',
                line=dict(color='#f093fb', width=3, dash='dash'),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🔝 Top Products by Error")
            
            error_data = results.copy()
            error_data['Error'] = abs(error_data['Demand Forecast'] - error_data['Predicted_Demand'])
            top_errors = error_data.nlargest(10, 'Error')[['Product ID', 'Error']]
            
            fig2 = go.Figure(go.Bar(
                x=top_errors['Error'],
                y=top_errors['Product ID'],
                orientation='h',
                marker=dict(
                    color=top_errors['Error'],
                    colorscale='Reds',
                    showscale=False
                )
            ))
            
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=False),
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # Data table
        st.markdown("### 📋 Detailed Predictions")
        
        display = results[['Date', 'Store ID', 'Product ID', 'Demand Forecast', 'Predicted_Demand']].copy()
        display['Error'] = abs(display['Demand Forecast'] - display['Predicted_Demand'])
        display['Error_%'] = (display['Error'] / (display['Demand Forecast'] + 1e-8) * 100).round(2)
        display = display.round(2)
        
        st.dataframe(display, use_container_width=True, height=400)
        
        # Download button
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Full Predictions",
            data=csv,
            file_name="demand_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # Empty state with instructions
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; color: white;'>
        <h2 style='color: rgba(255,255,255,0.9); margin-bottom: 1rem;'>👋 Welcome to Retail Demand Forecasting</h2>
        <p style='font-size: 1.1rem; color: rgba(255,255,255,0.7); margin-bottom: 2rem;'>
            Upload your retail inventory CSV to get AI-powered demand predictions
        </p>
        <div style='background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 2rem; max-width: 600px; margin: 0 auto; border: 1px solid rgba(255,255,255,0.2);'>
            <h3 style='color: white; margin-bottom: 1rem;'>📋 Required Columns:</h3>
            <p style='color: rgba(255,255,255,0.8); line-height: 1.8;'>
                Date • Store ID • Product ID • Category • Region<br>
                Inventory Level • Units Sold • Units Ordered<br>
                Demand Forecast • Price • Discount<br>
                Weather Condition • Holiday/Promotion<br>
                Competitor Pricing • Seasonality
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 2rem; color: rgba(255,255,255,0.5); margin-top: 3rem;'>
    <p>Powered by Transformer + XGBoost Ensemble • Expected MAPE: 3-5%</p>
</div>
""", unsafe_allow_html=True)
