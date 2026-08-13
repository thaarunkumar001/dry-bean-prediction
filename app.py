from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Dry Bean Classification",
    layout="wide",
)

model_dir=Path("model")
target_column="Class"

model_filenames = {
    "Logistic Regression":
        "logistic_regression.joblib",
    "Decision Tree":
        "decision_tree.joblib",
    "KNN":
        "knn.joblib",
    "Gaussian Naive Bayes":
        "naive_bayes.joblib",
    "Random Forest":
        "random_forest.joblib"
}

st.markdown(
    """
    <style>
        .title {
            color: #000000;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #403C3C;
            font-size: 1.05rem;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #000000;
            font-size: 1.5rem;
            font-weight: 650;
            margin-bottom: 1.5rem;
            margin-bottom: 0.2rem;
        }
        div[data-testid="stMetric"] {
            background-color: #547861;
            border: 1px solid #437d2e;
            border-radius: 10px;
            padding: 12px;
        }
        div[data-testid="stMetricLabel"] {
            color: #d4d2d2 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title">Dry Bean Classification</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="subtitle">Implementing Multiple Classification Models</div>'
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def model_load(model_path):
    return joblib.load(model_path)

def validate_dataset(data):
    if target_column not in data.columns:
        st.error(
            f'The Uploaded file is invalid'
        )
        return False
    feature_columns=[
        column for column in data.columns if column !=target_column

    ]
    