import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("🛒 Retail Demand Forecasting Dashboard")
st.markdown("**Transformer + XGBoost Model | Target MAPE: ~3%**")

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

# --- Exact replication of your Colab preprocessing ---
class TransformerXGBPredictor:
    def __init__(self, transformer_model, xgb_model, scaler, training_columns, sequence_length):
        self.transformer_model = transformer_model
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.training_columns = training_columns
        self.sequence_length = sequence_length

    def preprocess_data(self, df):
        """EXACT preprocessing from Colab - Line by line match"""
        df = df.copy()
        
        # Convert 'Date' column to datetime objects
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract time-based features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
        
        # Sort the DataFrame by 'Date'
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        # Create lag features
        lag_period = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)
        
        # Create rolling window features (mean and std)
        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
            df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
            df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
        
        # Handle potential missing values created by lag and rolling window features (fill with 0)
        df = df.fillna(0)
        
        # Define features (X) and target variable (y)
        features = [col for col in df.columns if col not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        X = df[features]
        y = df['Demand Forecast']
        
        # Convert categorical columns to numerical using one-hot encoding
        X = pd.get_dummies(X, columns=['Discount', 'Holiday/Promotion'])
        
        return X, y, df

    def create_sequences(self, X, y, sequence_length):
        """EXACT sequence creation from Colab"""
        X_sequences, y_sequences = [], []
        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:(i + sequence_length)])
            y_sequences.append(y[i + sequence_length])
        return np.array(X_sequences), np.array(y_sequences)

    def predict(self, df_input):
        """EXACT prediction pipeline from Colab"""
        
        # Step 1: Preprocess data
        X, y, df_original = self.preprocess_data(df_input)
        
        # Step 2: Ensure all training columns are present and in correct order
        for col in self.training_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.training_columns]
        
        # Step 3: Scale the data (EXACT as Colab)
        X_scaled = self.scaler.transform(X)
        
        # Step 4: Create sequences for Transformer
        X_sequences, y_sequences = self.create_sequences(X_scaled, y.values, self.sequence_length)
        
        if len(X_sequences) == 0:
            st.error(f"Not enough data to create sequences. Need at least {self.sequence_length + 1} rows per Store-Product.")
            return None, None
        
        # Step 5: Generate Transformer predictions
        transformer_predictions_scaled = self.transformer_model.predict(X_sequences, verbose=0)
        
        # Step 6: Prepare data for XGBoost (EXACT as Colab)
        # Align original test features
        X_aligned = X.iloc[self.sequence_length:].copy()
        y_aligned = y.values[self.sequence_length:].copy()
        df_original_aligned = df_original.iloc[self.sequence_length:].copy()
        
        # Add Transformer predictions as a feature
        X_aligned['transformer_predictions_scaled'] = transformer_predictions_scaled.flatten()
        
        # Step 7: Generate final predictions with XGBoost
        final_predictions = self.xgb_model.predict(X_aligned)
        
        # Step 8: Prepare results
        df_results = df_original_aligned.reset_index(drop=True).copy()
        df_results['Predicted_Demand'] = final_predictions
        
        # Step 9: Calculate MAPE (EXACT as Colab)
        epsilon = 1e-8
        y_aligned_safe = y_aligned.copy()
        y_aligned_safe[y_aligned_safe == 0] = epsilon
        mape = mean_absolute_percentage_error(y_aligned_safe, final_predictions) * 100
        
        return df_results, mape

# --- File uploader ---
st.markdown("### 📁 Upload Your Data")
st.markdown("Upload the same CSV format used in training")

uploaded_file = st.file_uploader(
    "Choose CSV file", 
    type=["csv"],
    help="Must include all original columns: Date, Store ID, Product ID, etc."
)

if uploaded_file is not None:
    try:
        # Load data
        df_input = pd.read_csv(uploaded_file)
        
        st.markdown("### 📊 Input Data Preview")
        st.dataframe(df_input.head(20), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df_input):,}")
        with col2:
            unique_stores = df_input['Store ID'].nunique()
            st.metric("Unique Stores", unique_stores)
        with col3:
            unique_products = df_input['Product ID'].nunique()
            st.metric("Unique Products", unique_products)
        
        # Initialize predictor
        predictor = TransformerXGBPredictor(
            transformer_model, xgb_model, scaler, training_columns, sequence_length
        )
        
        # Generate predictions
        with st.spinner("🔮 Running Transformer + XGBoost pipeline..."):
            df_results, mape = predictor.predict(df_input)
        
        if df_results is not None:
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            # Display MAPE prominently
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if mape <= 5:
                    status = "🎉 Excellent!"
                elif mape <= 10:
                    status = "✅ Good"
                elif mape <= 20:
                    status = "⚠️ Fair"
                else:
                    status = "❌ Poor"
                
                st.metric("MAPE Score", f"{mape:.2f}%", 
                         help="Mean Absolute Percentage Error")
                st.caption(status)
            
            with col2:
                st.metric("Predictions", f"{len(df_results):,}")
            
            with col3:
                accuracy = max(0, 100 - mape)
                st.metric("Accuracy Est.", f"{accuracy:.1f}%")
            
            with col4:
                avg_actual = df_results['Demand Forecast'].mean()
                avg_pred = df_results['Predicted_Demand'].mean()
                diff_pct = ((avg_pred - avg_actual) / avg_actual * 100) if avg_actual != 0 else 0
                st.metric("Avg Demand Diff", f"{diff_pct:.1f}%")
            
            st.markdown("---")
            
            # Show comparison table
            st.markdown("#### 📈 Actual vs Predicted Comparison")
            
            display_df = df_results[['Date', 'Store ID', 'Product ID', 'Demand Forecast', 'Predicted_Demand']].copy()
            display_df['Absolute_Error'] = abs(display_df['Demand Forecast'] - display_df['Predicted_Demand'])
            display_df['Error_%'] = (display_df['Absolute_Error'] / (display_df['Demand Forecast'] + 1e-8) * 100).round(2)
            
            # Sort by error for review
            display_df_sorted = display_df.sort_values('Error_%', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Best Predictions (Lowest Error)**")
                st.dataframe(display_df.nsmallest(10, 'Error_%'), use_container_width=True)
            
            with col2:
                st.markdown("**Highest Errors (Need Review)**")
                st.dataframe(display_df_sorted.head(10), use_container_width=True)
            
            # Summary statistics
            st.markdown("#### 📊 Statistical Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Actual Demand", f"{df_results['Demand Forecast'].mean():.2f}")
                st.metric("Median Actual", f"{df_results['Demand Forecast'].median():.2f}")
            
            with col2:
                st.metric("Mean Predicted Demand", f"{df_results['Predicted_Demand'].mean():.2f}")
                st.metric("Median Predicted", f"{df_results['Predicted_Demand'].median():.2f}")
            
            with col3:
                st.metric("Mean Absolute Error", f"{display_df['Absolute_Error'].mean():.2f}")
                st.metric("Std Dev of Error %", f"{display_df['Error_%'].std():.2f}%")
            
            # Full data view
            st.markdown("#### 📋 Complete Results")
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Complete Predictions CSV",
                data=csv,
                file_name="demand_predictions_transformer_xgb.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        with st.expander("🔍 View Full Error Details"):
            st.exception(e)

else:
    st.info("👆 Please upload your CSV file to generate predictions")
    
    with st.expander("📋 Required CSV Format"):
        st.markdown("""
        **Your CSV must contain these exact columns:**
        
        | Column | Type | Example |
        |--------|------|---------|
        | Date | Date | 2024-01-01 |
        | Store ID | Text | S001 |
        | Product ID | Text | P0001 |
        | Category | Text | Groceries |
        | Region | Text | North |
        | Inventory Level | Numeric | 231 |
        | Units Sold | Numeric | 127 |
        | Units Ordered | Numeric | 55 |
        | Demand Forecast | Numeric | 135.47 |
        | Price | Numeric | 33.5 |
        | Discount | Numeric/Text | 20 or "20%" |
        | Weather Condition | Text | Rainy |
        | Holiday/Promotion | Numeric | 0 or 1 |
        | Competitor Pricing | Numeric | 29.69 |
        | Seasonality | Text | Autumn |
        
        **Important Notes:**
        - Data should be sorted by Date
        - Need at least 7+ consecutive rows per Store-Product combination
        - Missing values will be filled with 0
        """)

st.markdown("---")
st.caption("Transformer + XGBoost Ensemble | Expected MAPE: 3-5% on test data")
