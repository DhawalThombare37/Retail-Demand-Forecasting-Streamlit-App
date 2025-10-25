import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb

# ==========================================================
# 🧠 MODEL UTILITIES
# ==========================================================

def load_models(transformer_path: str, xgb_path: str):
    """
    Load both Transformer (Keras) and XGBoost models.
    """
    transformer_model = tf.keras.models.load_model(transformer_path)
    xgb_model = joblib.load(xgb_path)
    return transformer_model, xgb_model


def generate_transformer_predictions(model, X_seq):
    """
    Generate Transformer model predictions.
    """
    preds = model.predict(X_seq, verbose=0)
    return preds.flatten()


def generate_xgb_predictions(model, df_test):
    """
    Generate XGBoost predictions using preprocessed test DataFrame.
    """
    preds = model.predict(df_test)
    return preds


def save_model(model, path: str, framework: str = 'keras'):
    """
    Save either Keras or sklearn-compatible model.
    """
    if framework == 'keras':
        model.save(path)
    else:
        joblib.dump(model, path)
    print(f"✅ Model saved at: {path}")
