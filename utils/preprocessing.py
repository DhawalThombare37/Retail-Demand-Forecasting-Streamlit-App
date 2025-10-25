import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==========================================================
# 🧹 DATA PREPROCESSING & FEATURE ENGINEERING
# ==========================================================

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset and perform initial cleaning.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df.dropna(subset=['Date'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features derived from the 'Date' column.
    """
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, lag_period=7, rolling_window=7) -> pd.DataFrame:
    """
    Create lag and rolling mean/std features for selected numeric columns.
    """
    numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']

    for col in numeric_cols:
        if 'Store ID' in df.columns and 'Product ID' in df.columns:
            df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)
            df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
            df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
        else:
            df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)
            df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(window=rolling_window).mean()
            df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(window=rolling_window).std()

    df.fillna(0, inplace=True)
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare X and y datasets and store column info after one-hot encoding.
    """
    target_col = 'Demand Forecast'
    exclude_cols = ['Date', target_col, 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']

    X = df[[c for c in df.columns if c not in exclude_cols]]
    X = pd.get_dummies(X, columns=['Discount', 'Holiday/Promotion'], drop_first=False)
    y = df[target_col]

    training_columns = X.columns.tolist()
    return X, y, training_columns


def split_train_test(df: pd.DataFrame, months_test=3):
    """
    Split data into train and test sets based on the last N months.
    """
    test_date = df['Date'].max() - pd.DateOffset(months=months_test)
    train_mask = df['Date'] <= test_date

    return train_mask, ~train_mask


def create_sequences(X, y, sequence_length=7):
    """
    Create sequential samples for time-series models (e.g., Transformer).
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:(i + sequence_length)])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)


# ==========================================================
# ⚙️ SCALING & UTILITIES
# ==========================================================

def scale_data(X_train, X_test, scaler_path=None):
    """
    Scale the training and test features using MinMaxScaler.
    Optionally saves/loads the scaler from a file path.
    """
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_path:
        joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled, scaler


def compute_mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-8
    y_true = np.where(y_true == 0, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return round(mape, 2)
