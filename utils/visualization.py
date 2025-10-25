import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

# ==========================================================
# 🎨 VISUALIZATION UTILITIES
# ==========================================================

def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=range(len(y_true)), y=y_true, label='Actual', ax=ax)
    sns.lineplot(x=range(len(y_pred)), y=y_pred, label='Predicted', ax=ax)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)


def plot_feature_importance(booster, top_n=10):
    """
    Plot top N XGBoost feature importances.
    """
    importance = booster.get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Score'])
    importance_df = importance_df.sort_values(by='Score', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=importance_df, x='Score', y='Feature', ax=ax)
    ax.set_title(f'Top {top_n} Important Features')
    st.pyplot(fig)
