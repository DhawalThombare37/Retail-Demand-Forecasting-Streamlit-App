import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("🛒 Retail Demand Forecasting Dashboard")
st.markdown("**Transformer + XGBoost Model | Expected MAPE: ~3%**")

# --- Load models and ALL preprocessors ---
@st.cache_resource
def load_models():
    try:
        transformer_model = tf.keras.models.load_model("transformer_model.keras")
        xgb_model = joblib.load("xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        training_columns = joblib.load("training_columns.pkl")
        
        # Load XGBoost training columns (includes transformer_predictions_scaled)
        try:
            xgb_training_columns = joblib.load("xgb_training_columns.pkl")
        except:
            # Fallback: assume it's training_columns + transformer feature
            xgb_training_columns = training_columns + ['transformer_predictions_scaled']
            st.warning("Using fallback XGBoost columns. Please save xgb_training_columns.pkl from Colab.")
        
        sequence_length = joblib.load("sequence_length.pkl")
        
        st.success(f"✅ Models loaded! Transformer features: {len(training_columns)}, XGBoost features: {len(xgb_training_columns)}")
        return transformer_model, xgb_model, scaler, training_columns, xgb_training_columns, sequence_length
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

transformer_model, xgb_model, scaler, training_columns, xgb_training_columns, sequence_length = load_models()

if transformer_model is None:
    st.stop()

# --- EXACT replication of Colab code ---
class TransformerXGBPredictor:
    def __init__(self, transformer_model, xgb_model, scaler, training_columns, xgb_training_columns, sequence_length):
        self.transformer_model = transformer_model
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.training_columns = training_columns
        self.xgb_training_columns = xgb_training_columns
        self.sequence_length = sequence_length

    def preprocess_data(self, df):
        """EXACT copy of Colab preprocessing"""
        df = df.copy()
        
        # Convert 'Date' column to datetime objects
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort the DataFrame by 'Date' (DO THIS FIRST!)
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        # Extract time-based features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
        
        # Create lag features
        lag_period = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)
        
        # Create rolling window features (mean and std)
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
            df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
        
        # Handle potential missing values (fill with 0)
        df = df.fillna(0)
        
        # Define features (X) and target variable (y)
        features = [col for col in df.columns if col not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        X = df[features]
        y = df['Demand Forecast']
        
        # Convert categorical columns to numerical using one-hot encoding
        X = pd.get_dummies(X, columns=['Discount', 'Holiday/Promotion'])
        
        return X, y, df

    def create_sequences(self, X, y, sequence_length):
        """EXACT copy from Colab"""
        X_sequences, y_sequences = [], []
        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:(i + sequence_length)])
            y_sequences.append(y[i + sequence_length])
        return np.array(X_sequences), np.array(y_sequences)

    def predict(self, df_input):
        """EXACT replication of Colab prediction pipeline"""
        
        # Step 1: Preprocess
        X, y, df_original = self.preprocess_data(df_input)
        
        st.info(f"After preprocessing: {X.shape[1]} features generated")
        
        # Step 2: Align columns with training
        missing_cols = []
        for col in self.training_columns:
            if col not in X.columns:
                X[col] = 0
                missing_cols.append(col)
        
        if missing_cols:
            st.warning(f"Added {len(missing_cols)} missing columns with zeros")
        
        # Ensure exact column order
        X = X[self.training_columns]
        
        st.info(f"After alignment: {X.shape[1]} features (expected: {len(self.training_columns)})")
        
        # Step 3: Scale
        X_scaled = self.scaler.transform(X)
        
        # Step 4: Create sequences
        X_sequences, y_sequences = self.create_sequences(X_scaled, y.values, self.sequence_length)
        
        if len(X_sequences) == 0:
            st.error(f"Not enough data. Need at least {self.sequence_length + 1} rows.")
            return None, None
        
        st.info(f"Created {len(X_sequences)} sequences")
        
        # Step 5: Transformer predictions
        transformer_predictions_scaled = self.transformer_model.predict(X_sequences, verbose=0)
        
        # Step 6: Align data for XGBoost (CRITICAL!)
        X_test_aligned = X.iloc[self.sequence_length:].copy()
        y_test_aligned = y.values[self.sequence_length:].copy()
        df_original_aligned = df_original.iloc[self.sequence_length:].copy()
        
        # Add transformer predictions
        X_test_aligned['transformer_predictions_scaled'] = transformer_predictions_scaled.flatten()
        
        st.info(f"XGBoost input shape: {X_test_aligned.shape}")
        st.info(f"XGBoost columns: {list(X_test_aligned.columns[:5])}... + {list(X_test_aligned.columns[-3:])}")
        
        # Step 7: Ensure XGBoost gets exact training columns
        for col in self.xgb_training_columns:
            if col not in X_test_aligned.columns:
                st.error(f"Missing XGBoost column: {col}")
                X_test_aligned[col] = 0
        
        # Reorder to match XGBoost training
        X_test_aligned = X_test_aligned[self.xgb_training_columns]
        
        # Step 8: XGBoost predictions
        final_predictions = self.xgb_model.predict(X_test_aligned)
        
        # Step 9: Results
        df_results = df_original_aligned.reset_index(drop=True).copy()
        df_results['Predicted_Demand'] = final_predictions
        
        # Step 10: Calculate MAPE (EXACT as Colab)
        epsilon = 1e-8
        y_test_safe = y_test_aligned.copy()
        y_test_safe[y_test_safe == 0] = epsilon
        mape = mean_absolute_percentage_error(y_test_safe, final_predictions) * 100
        
        return df_results, mape

# --- File uploader ---
st.markdown("### 📁 Upload Your Data")

uploaded_file = st.file_uploader(
    "Choose retail_store_inventory.csv", 
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        
        st.markdown("### 📊 Input Data")
        st.dataframe(df_input.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df_input):,}")
        with col2:
            st.metric("Stores", df_input['Store ID'].nunique())
        with col3:
            st.metric("Products", df_input['Product ID'].nunique())
        
        # Validate required columns
        required_cols = ['Date', 'Store ID', 'Product ID', 'Inventory Level', 'Units Sold', 
                        'Units Ordered', 'Demand Forecast', 'Price', 'Discount', 'Holiday/Promotion']
        missing = [col for col in required_cols if col not in df_input.columns]
        
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()
        
        # Initialize predictor
        predictor = TransformerXGBPredictor(
            transformer_model, xgb_model, scaler, training_columns, xgb_training_columns, sequence_length
        )
        
        # Generate predictions
        with st.spinner("🔮 Running pipeline..."):
            df_results, mape = predictor.predict(df_input)
        
        if df_results is not None:
            st.markdown("---")
            st.markdown("### 🎯 Results")
            
            # MAPE Display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                color = "🎉" if mape <= 5 else "⚠️" if mape <= 10 else "❌"
                st.metric("MAPE", f"{mape:.2f}%", help="Target: ~3%")
                st.caption(f"{color} {'Excellent!' if mape <= 5 else 'Needs review'}")
            
            with col2:
                st.metric("Predictions", f"{len(df_results):,}")
            
            with col3:
                st.metric("Accuracy", f"{max(0, 100-mape):.1f}%")
            
            # Comparison
            st.markdown("#### 📈 Sample Predictions")
            display = df_results[['Date', 'Store ID', 'Product ID', 'Demand Forecast', 'Predicted_Demand']].head(50)
            display['Error'] = abs(display['Demand Forecast'] - display['Predicted_Demand'])
            display['Error_%'] = (display['Error'] / (display['Demand Forecast'] + 1e-8) * 100).round(2)
            st.dataframe(display, use_container_width=True)
            
            # Download
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Predictions",
                csv,
                "predictions.csv",
                "text/csv",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload CSV to start")

st.markdown("---")
st.caption("Expected MAPE: 3-5%")
