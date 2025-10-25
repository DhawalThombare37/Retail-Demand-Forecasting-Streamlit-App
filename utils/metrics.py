import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================================
# 📊 METRICS UTILITIES
# ==========================================================

def mape(y_true, y_pred):
    epsilon = 1e-8
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
