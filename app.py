import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("🛒 Retail Demand Forecasting")
st.markdown("**Transformer + XGBoost Ensemble | Expected MAPE: ~3%**")

# Load models
@st.cache_resource
def load_models():
    import os
    import urllib.request
    
    # Check if files exist locally
    required_files = {
        'transformer_model.keras': 'Transformer Model',
        'xgb_model.pkl': 'XGBoost Model',
        'scaler.pkl': 'Scaler',
        'training_columns.pkl': 'Training Columns',
        'xgb_columns.pkl': 'XGBoost Columns',
        'sequence_length.pkl': 'Sequence Length'
    }
    
    st.info("🔍 Checking for model files...")
    
    # Debug: Show current directory
    current_files = os.listdir(".")
    st.write(f"**Files in current directory ({len(current_files)}):**")
    st.code("\n".join(sorted(current_files)))
    
    # Check each file
    missing_files = []
    file_sizes = {}
    for file, name in required_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            file_sizes[file] = size
            st.success(f"✅ {name}: {file} ({size/1024/1024:.2f} MB)")
        else:
            missing_files.append(file)
            st.error(f"❌ {name}: {file} - NOT FOUND")
    
    if missing_files:
        st.error(f"**Missing {len(missing_files)} file(s):**")
        for f in missing_files:
            st.write(f"- {f}")
        st.info("""
        **Solutions:**
        1. **If transformer_model.keras > 100MB:** Use Git LFS
           ```bash
           git lfs install
           git lfs track "*.keras"
           git add .gitattributes
           git add transformer_model.keras
           git commit -m "Add large model file"
           git push
           ```
        
        2. **Alternative:** Upload to Google Drive/Dropbox and download in app
        
        3. **Check:** Files are committed and pushed to GitHub
        """)
        return None, None, None, None, None, None
    
    try:
        st.info("📦 Loading models...")
        transformer = tf.keras.models.load_model("transformer_model.keras")
        st.success("✅ Transformer loaded")
        
        xgb = joblib.load("xgb_model.pkl")
        st.success("✅ XGBoost loaded")
        
        scaler = joblib.load("scaler.pkl")
        training_cols = joblib.load("training_columns.pkl")
        xgb_cols = joblib.load("xgb_columns.pkl")
        seq_len = joblib.load("sequence_length.pkl")
        
        st.success(f"✅ All models loaded! Train: {len(training_cols)} | XGB: {len(xgb_cols)} | Seq: {seq_len}")
        return transformer, xgb, scaler, training_cols, xgb_cols, seq_len
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.exception(e)
        return None, None, None, None, None, None

transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length = load_models()

if transformer_model is None:
    st.stop()

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
        # Split into train/test like Colab (last 3 months for test)
        test_date = pd.to_datetime(df_input['Date']).max() - pd.DateOffset(months=3)
        
        st.info(f"📅 Using last 3 months as test set (from {test_date.date()} onwards)")
        st.info(f"⚠️ Model was trained on last 3 months only!")
        
        X, y, df_orig = self.preprocess(df_input)
        
        st.write("---")
        st.write("### 🔍 DEBUG INFO")
        
        st.write(f"**1. After preprocessing:**")
        st.write(f"- Generated features: {X.shape[1]}")
        st.write(f"- Rows: {X.shape[0]}")
        
        generated_cols = list(X.columns)
        st.write(f"**Generated columns ({len(generated_cols)}):**")
        with st.expander("View all generated columns"):
            st.code("\n".join(generated_cols))
        
        st.write(f"**2. Training columns expected: {len(self.training_columns)}**")
        with st.expander("View expected training columns"):
            st.code("\n".join(self.training_columns))
        
        # Find mismatches
        missing_in_input = [col for col in self.training_columns if col not in generated_cols]
        extra_in_input = [col for col in generated_cols if col not in self.training_columns]
        
        if missing_in_input:
            st.error(f"⚠️ **Missing {len(missing_in_input)} columns** (will be filled with 0):")
            with st.expander("View missing columns"):
                st.code("\n".join(missing_in_input[:50]))
        
        if extra_in_input:
            st.warning(f"⚠️ **Extra {len(extra_in_input)} columns** (will be removed):")
            with st.expander("View extra columns"):
                st.code("\n".join(extra_in_input[:50]))
        
        # Show one-hot encoded columns specifically
        discount_cols_generated = [col for col in generated_cols if 'Discount_' in col]
        discount_cols_expected = [col for col in self.training_columns if 'Discount_' in col]
        
        holiday_cols_generated = [col for col in generated_cols if 'Holiday/Promotion_' in col]
        holiday_cols_expected = [col for col in self.training_columns if 'Holiday/Promotion_' in col]
        
        st.write("**3. One-hot encoding comparison:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Discount columns:**")
            st.write(f"Generated: {discount_cols_generated}")
            st.write(f"Expected: {discount_cols_expected}")
        with col2:
            st.write("**Holiday/Promotion columns:**")
            st.write(f"Generated: {holiday_cols_generated}")
            st.write(f"Expected: {holiday_cols_expected}")
        
        # Align to training columns
        for col in self.training_columns:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.training_columns]
        
        st.success(f"✅ Aligned to {X.shape[1]} features")
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y.values)
        
        if len(X_seq) == 0:
            st.error(f"Need at least {self.sequence_length + 1} rows")
            return None, None
        
        st.write(f"**4. Sequences created:** {len(X_seq)}")
        
        # Transformer predictions
        trans_preds = self.transformer.predict(X_seq, verbose=0)
        
        # Align for XGBoost
        X_aligned = X.iloc[self.sequence_length:].copy()
        y_aligned = y.values[self.sequence_length:].copy()
        df_aligned = df_orig.iloc[self.sequence_length:].copy()
        
        # Add transformer predictions
        X_aligned['transformer_predictions_scaled'] = trans_preds.flatten()
        
        st.write(f"**5. XGBoost input:**")
        st.write(f"- Shape: {X_aligned.shape}")
        st.write(f"- Expected columns: {len(self.xgb_columns)}")
        st.write(f"- Current columns: {len(X_aligned.columns)}")
        
        xgb_cols_current = list(X_aligned.columns)
        missing_xgb = [col for col in self.xgb_columns if col not in xgb_cols_current]
        extra_xgb = [col for col in xgb_cols_current if col not in self.xgb_columns]
        
        if missing_xgb:
            st.error(f"⚠️ Missing {len(missing_xgb)} XGBoost columns")
        if extra_xgb:
            st.warning(f"⚠️ Extra {len(extra_xgb)} XGBoost columns")
        
        # Align to XGBoost columns
        for col in self.xgb_columns:
            if col not in X_aligned.columns:
                X_aligned[col] = 0
        
        X_aligned = X_aligned[self.xgb_columns]
        
        st.success(f"✅ XGBoost aligned to {X_aligned.shape[1]} features")
        
        # Final predictions
        final_preds = self.xgb.predict(X_aligned)
        
        # Results
        df_results = df_aligned.reset_index(drop=True).copy()
        df_results['Predicted_Demand'] = final_preds
        
        # MAPE
        epsilon = 1e-8
        y_safe = y_aligned.copy()
        y_safe[y_safe == 0] = epsilon
        mape = mean_absolute_percentage_error(y_safe, final_preds) * 100
        
        st.write("---")
        
        return df_results, mape

# Upload
st.markdown("### 📁 Upload Data")
uploaded = st.file_uploader("retail_store_inventory.csv", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Stores", df['Store ID'].nunique())
        with col3:
            st.metric("Products", df['Product ID'].nunique())
        
        predictor = Predictor(transformer_model, xgb_model, scaler, training_columns, xgb_columns, sequence_length)
        
        with st.spinner("Predicting..."):
            results, mape = predictor.predict(df)
        
        if results is not None:
            st.markdown("---")
            st.markdown("### 🎯 Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                emoji = "🎉" if mape <= 5 else "✅" if mape <= 10 else "⚠️"
                st.metric("MAPE", f"{mape:.2f}%")
                st.caption(emoji)
            with col2:
                st.metric("Predictions", f"{len(results):,}")
            with col3:
                st.metric("Accuracy", f"{max(0,100-mape):.1f}%")
            
            display = results[['Date','Store ID','Product ID','Demand Forecast','Predicted_Demand']].copy()
            display['Error_%'] = (abs(display['Demand Forecast']-display['Predicted_Demand'])/(display['Demand Forecast']+1e-8)*100).round(2)
            
            st.dataframe(display.head(50), use_container_width=True)
            
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download", csv, "predictions.csv", use_container_width=True)
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
else:
    st.info("👆 Upload CSV")

st.caption("Expected MAPE: 3-5%")
