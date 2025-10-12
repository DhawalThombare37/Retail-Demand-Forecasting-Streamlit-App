import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("🛒 Retail Demand Forecasting Dashboard")
st.markdown("**Transformer + XGBoost Model**")

# --- Load models and scaler ---
@st.cache_resource
def load_models():
    try:
        transformer_model = tf.keras.models.load_model("transformer_model.keras")
        xgb_model = joblib.load("xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        training_columns = joblib.load("training_columns.pkl")
        sequence_length = joblib.load("sequence_length.pkl")
        st.success("✅ Models loaded successfully!")
        return transformer_model, xgb_model, scaler, training_columns, sequence_length
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

transformer_model, xgb_model, scaler, training_columns, sequence_length = load_models()

if transformer_model is None:
    st.stop()

# --- Wrapper class for preprocessing + prediction ---
class TransformerXGBPredictor:
    def __init__(self, transformer_model, xgb_model, scaler, training_columns, sequence_length):
        self.transformer_model = transformer_model
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.training_columns = training_columns
        self.sequence_length = sequence_length

    def preprocess(self, df):
        """Preprocess data exactly as in training"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Time-based features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

        # Lag & rolling features (MUST match training exactly)
        lag_period = 7
        rolling_window = 7
        
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            if 'Store ID' in df.columns and 'Product ID' in df.columns:
                # Lag features
                df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)
                # Rolling mean - match training column name exactly
                df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(rolling_window).mean().reset_index(drop=True)
                # Rolling std - match training column name exactly
                df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(rolling_window).std().reset_index(drop=True)
            else:
                df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)
                df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(rolling_window).mean()
                df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(rolling_window).std()

        # Fill NaN values
        df = df.fillna(0)

        # Select features (exclude non-feature columns)
        features_to_keep = [col for col in df.columns if col not in 
                           ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 
                            'Category', 'Region', 'Weather Condition', 'Seasonality']]
        
        df_features = df[features_to_keep].copy()
        
        # One-hot encoding (MUST match training)
        df_processed = pd.get_dummies(df_features, columns=['Discount', 'Holiday/Promotion'])
        
        # Ensure all training columns exist
        for col in self.training_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Select only training columns in exact order
        df_processed = df_processed[self.training_columns]
        
        return df_processed, df

    def predict(self, df_input):
        """Generate predictions using Transformer + XGBoost pipeline"""
        df_processed, df_original = self.preprocess(df_input)
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed)
        
        # Create sequences for Transformer
        X_sequences = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_sequences.append(X_scaled[i:i + self.sequence_length])
        
        if len(X_sequences) == 0:
            st.error("Not enough data to create sequences. Need at least 7 consecutive rows per Store-Product.")
            return None, None
        
        X_sequences = np.array(X_sequences)
        
        # Get Transformer predictions
        transformer_preds = self.transformer_model.predict(X_sequences, verbose=0).flatten()
        
        # Align data (skip first sequence_length-1 rows)
        df_aligned = df_processed.iloc[self.sequence_length - 1:].reset_index(drop=True)
        df_original_aligned = df_original.iloc[self.sequence_length - 1:].reset_index(drop=True)
        
        # Add transformer predictions as feature for XGBoost
        df_aligned['transformer_predictions_scaled'] = transformer_preds
        
        # Ensure columns match XGBoost training
        xgb_features = df_aligned.columns.tolist()
        
        # Final predictions with XGBoost
        final_preds = self.xgb_model.predict(df_aligned)
        
        # Add predictions to results
        df_results = df_original_aligned.copy()
        df_results['Predicted_Demand'] = final_preds
        
        # Calculate MAPE if ground truth exists
        if 'Demand Forecast' in df_results.columns:
            y_true = df_results['Demand Forecast'].values
            epsilon = 1e-8
            y_true_safe = np.where(y_true == 0, epsilon, y_true)
            mape = mean_absolute_percentage_error(y_true_safe, final_preds) * 100
        else:
            mape = None
        
        return df_results, mape

# --- File uploader ---
st.markdown("### 📁 Upload Your Data")
uploaded_file = st.file_uploader(
    "Upload retail_store_inventory.csv", 
    type=["csv"],
    help="CSV must contain: Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Demand Forecast, Price, Discount, Weather Condition, Holiday/Promotion, Competitor Pricing, Seasonality"
)

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        
        st.markdown("### 📊 Input Data Preview")
        st.dataframe(df_input.head(10), use_container_width=True)
        st.caption(f"Total rows: {len(df_input)}")
        
        # Initialize predictor
        predictor = TransformerXGBPredictor(
            transformer_model, xgb_model, scaler, training_columns, sequence_length
        )
        
        with st.spinner("🔮 Generating predictions..."):
            df_results, mape = predictor.predict(df_input)
        
        if df_results is not None:
            st.markdown("### 🎯 Prediction Results")
            
            # Display MAPE
            if mape is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAPE Score", f"{mape:.2f}%", 
                             help="Mean Absolute Percentage Error - lower is better")
                with col2:
                    st.metric("Predictions Generated", len(df_results))
                with col3:
                    accuracy = max(0, 100 - mape)
                    st.metric("Accuracy", f"{accuracy:.2f}%")
            
            # Show results
            st.markdown("#### Detailed Predictions")
            display_cols = ['Date', 'Store ID', 'Product ID', 'Demand Forecast', 'Predicted_Demand']
            display_cols = [col for col in display_cols if col in df_results.columns]
            st.dataframe(df_results[display_cols], use_container_width=True)
            
            # Download button
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name="demand_predictions.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please upload a CSV file to get started")
    
    # Show example format
    with st.expander("📋 View Required CSV Format"):
        example_data = {
            'Date': ['2024-01-01', '2024-01-02'],
            'Store ID': [1, 1],
            'Product ID': [101, 101],
            'Category': ['Electronics', 'Electronics'],
            'Region': ['North', 'North'],
            'Inventory Level': [100, 95],
            'Units Sold': [10, 12],
            'Units Ordered': [50, 50],
            'Demand Forecast': [15, 18],
            'Price': [299.99, 299.99],
            'Discount': ['No', 'No'],
            'Weather Condition': ['Clear', 'Clear'],
            'Holiday/Promotion': ['No', 'No'],
            'Competitor Pricing': [289.99, 289.99],
            'Seasonality': ['Low', 'Low']
        }
        st.dataframe(pd.DataFrame(example_data))

st.markdown("---")
st.caption("Built with Streamlit | Transformer + XGBoost Model")
