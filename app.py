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
    if len(feature_columns)!=16:
        st.error(
            f'Expected 16 input feature, but the uploaded file'
            f'contains {len(feature_columns)}'
        )
        return False
    missing_values=data.isnull().sum().sum()

    if missing_values>0:
        st.error(
            f'The updated dataset contains {missing_values}'
        )
        return False
    return True

def model_evaluation(model,X,y):
    predictions=model.predict(X)
    probabilities=None
    if hasattr(model,"predict_proba"):
        probabilities=model.predict_proba(X)

    accuracy=accuracy_score(y,predictions)
    precision=precision_score(y,predictions,average="weighted",zero_division=0,)
    recall=recall_score(y,predictions,average="weighted",zero_division=0,)
    f1=f1_score(y,predictions,average="weighted",zero_division=0,)
    mcc = matthews_corrcoef(y, predictions)
    auc=None
    if probabilities is not None:
        try:
            auc=roc_auc_score(y,predictions,average="weighted",zero_division=0,)
        except ValueError:
            auc=None
    
    metrics={
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
    }
    return precision,metrics

def show_metrics(metrics):
    columns=st.columns(6)
    metric_names=[
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]

    for column,name in zip(columns,metric_names):
        value=metrics[name]
        if value is None:
            display_value="N/A"
        else:
            display_value=f"{value:.f}"

        column.metric(
            label=name,
            value=display_value,
        )

def model_comparison(data):
    X=data.drop(columns=[target_column])
    y=data[target_column]
    rows=[]
    for model_name,filename in model_filenames.items():
        path=model_dir/filename
        if not path.exists():
            continue
        model=model_load(path)
        _,metrics=model_evaluation(model,X,y)
        rows.append(
            {
                "Model": model_name,
                "Accracy": metrics["Accuracy"],
                "AUC": metrics["AUC"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "MCC":metrics["MCC"],
            }
        )
    return pd.DataFrame(rows)

with st.sidebar:
    st.header("Evaluation Controls")
    st.write(
        "Upload the dataset and select the classifier"
    )
    uploaded_file=st.file_update(
        "Uploade the test dataset",
        type=["csv"],
    )
    selected_model=st.selectbox(
        "Select the classifier model",
        list(model_filenames.keys()),
    )
    st.divider()
    st.caption(
        "Models are trained offline and loaded from "
        "Joblib files."
    )

if uploaded_file is None:

    st.info(
        "Upload the Dry Bean test CSV using the sidebar "
        "to start the evaluation."
    )

    st.markdown(
        """
        ### What this application does

        This application evaluates five pretrained classification
        models using the uploaded test dataset.

        **Classifiers**

        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbors
        - Gaussian Naive Bayes
        - Random Forest

        **Evaluation**

        - Accuracy
        - AUC
        - Precision
        - Recall
        - F1 Score
        - Matthews Correlation Coefficient

        The application also provides a model comparison table,
        confusion matrix, and classification report.
        """
    )

    st.stop()

    try:
        test_data=pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(
            f"Could not read the file. Error: {error}"
        )
        st.stop()

if not validate_dataset(test_data):
    st.stop()


st.markdown(
    '<div class="section-heading">Test Dataset Overview</div>',
    unsafe_allow_html=True,
)

col1,col2,col3=st.columns(3)

col1.metric(
    "Test Samples",
    len(test_data),
)
col2.metric(
    "Input Features",
    len(test_data.columns)-1,
)
col3.metric(
    "Classes",
    test_data[target_column].nunique(),
)
with st.expander("Uploaded Data Preview"):
    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )

X_test=test_data.drop(columns=[target_column])
y_test=test_data[target_column]

non_numeric=X_test.select_dtypes(
    exclude='number'
).columns.tolist()

if non_numeric:
    st.error(
        "The following features are non numeric:".join(non_numeric)
    )
    st.stop()

filename_selected=model_filenames[selected_model]
selected_path=model_dir/filename_selected

if not selected_path.exists():
    st.error(
        f"The selected model was not found:\n\n"
        f"`{selected_path}`\n\n"
        "Run train_models.py and retry..."
    )
    st.stop()

model=model_load(selected_path)
st.markdown(
    '<div class="section-heading">Selected Classifier</div>',
    unsafe_allow_html=True,
)

st.success(
    f"Evaluating **{selected_model}**"
)

try:
    predictions, metrics=model_evaluation(model,X_test,y_test)
except Exception as e:
    st.error(f'Prediction Failed due to Error: {error}')
    st.stop()

st.markdown(
    '<div class="section-heading">Evaluation Metrics</div>',
    unsafe_allow_html=True,
)

show_metrics(metrics)

st.markdown(
    '<div class="section-heading">Classifier Comparison</div>',
    unsafe_allow_html=True,
)

comparison=model_comparison(test_data)
if comparison.empty:
    st.warning("No filed found")
else:
    metric_columns=[
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]
    comparison[metric_columns]=comparison[metric_columns].round(4)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

st.markdown(
    '<div class="section-heading">🔲 Confusion Matrix</div>',
    unsafe_allow_html=True,
)

class_labels=sorted(y_test.unique())

c_matrix=confusion_matrix(
    y_test,
    predictions,
    labels=class_labels
    )
fig, ax=plt.subplots(
    figsize=(9,7)
)
sns.heatmap(
    c_matrix,
    annot=True,
    fmt="d",
    cmap="YlGn",
    linewidths=0.5,
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=ax,
)


ax.set_title(f"Confusion Matrix — {selected_model}")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
plt.tight_layout()
st.pyplot(
    fig,
    use_container_width=True,
)
plt.close(fig)

st.markdown(
    '<div class="section-heading">Classification Report</div>',
    unsafe_allow_html=True,
)

report = classification_report(
    y_test,
    predictions,
    zero_division=0,
)

st.code(
    report,
    language="text",
)

st.markdown(
    '<div class="section-heading">Prediction Summary</div>',
    unsafe_allow_html=True,
)
correct_predications=int((predictions==y_test).sum())
incorrect_predications=int((predictions!=y_test).sum())
summary1,summary2=st.columns(2)
summary1.metric("Correct Predictions",correct_predications)
summary2.metric("In Correct Predictions",incorrect_predications)

st.divider()
st.caption(
    "Dry Bean Classification Lab • "
    "Evaluation performed on uploaded test data"
)
