import os
from pathlib import Path

import joblib
import kagglehub
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import(
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

dataset_name="muratkokludataset/dry-bean-dataset"
target_column="class"
test_size=0.20
random_state=42
model_directory="model"
test_file_name="test_data.csv"
result_file="model_results.csv"

#create model directory
os.makedirs(model_directory,exist_ok=True)

print("Downloading the dataset...")

dataset_path=kagglehub.dataset_download(dataset_name)
print(f"\n Dataset downloaded to: {dataset_path}")

dataset_path=Path(dataset_path)
csv_files=list(dataset_path.rglob("*.csv"))
excel_files=list(dataset_path.rglob("*.xlsx"))
if csv_files:
    data_file=csv_files[0]

elif excel_files:
    data_file=excel_files[0]

else:
    raise FileNotFoundError(
        "Could not find a CSV or XLSX file in the downloaded "
        "Kaggle dataset."
    )

print(f"\nUsing dataset file:")
print(data_file)

#loading the dataset
if data_file.suffix.lower()==".csv":
    df=pd.read_csv(data_file)
elif data_file.suffix.lower()==".xlsx":
    df=pd.read_excel(data_file)
else:
    raise ValueError(
        f"Unsupported dataset format: {data_file.suffix}"
    )
print("\nDataset loaded successfully.")
print(f"Original shape: {df.shape}")

df.columns=df.columns.str.strip()
if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' was not found.\n"
    )


df = df.dropna(how="any")
df = df.drop_duplicates
print(f"Shape after cleaning: {df.shape}")

X=df.frop(columns=[target_column])
y=df[target_column]

print("\n"+"="*70)
print("DATASET INFORMATION")
print("="*70)

print(f"Number of instances : {X.shape[0]}")
print(f"Number of features  : {X.shape[1]}")
print(f"Number of classes   : {y.nunique()}")

print("\nClasses:")
print(y.unique())

print("\nClass distribution:")
print(y.value_counts())

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=test_size,random_state=42,stratify=y
)

print("\n"+"="*70)
print("TRAIN / TEST SPLIT")
print("="*70)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

models={
    "Logestic Regression":Pipeline([
        (
            "scalar",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        ),
    ]),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "KNN": Pipeline([
        (
            "scalar",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        ),
    ]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Random Forst": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )  
}

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

results=[]

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)

    accuracy=accuracy_score(y_test,y_pred)
    auc = roc_auc_score(y_test,y_prob,multi_class="ovr",average="weighted")
    precision = precision_score(y_test,y_pred,average="weighted",zero_division=0)
    recall = recall_score(y_test,y_pred,average="weighted",zero_division=0)
    f1 = f1_score(y_test,y_pred,average="weighted",zero_division=0)
    mcc=matthews_corrcoef(y_test,y_pred)

    results.append({
        "Model":model_name,
        "Accuracy":accuracy,
        "AUC":auc,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "MCC":mcc
    })

    filename=model_filenames[model_name]
    model_path=os.path.join(model_directory,filename)
    joblib.dump(model,model_path)

    print(f"Saved: {model_path}")

    print(
        f"Accuracy: {accuracy:.4f} | "
        f"AUC: {auc:.4f} | "
        f"F1: {f1:.4f} | "
        f"MCC: {mcc:.4f}"
    )

test_data=X_test.copy()
test_data[target_column]=y_test.values
test_data.to_csv(test_file_name, index=False)

print("\n"+"="*70)
print("TEST DATA CREATED")
print("="*70)

print(f"File: {TEST_DATA_FILE}")
print(f"Rows: {len(test_data)}")

results_df=pd.DataFrame(
    results
)
eval_columns=[
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]

results_df[eval_columns]=(
    results_df[eval_columns].round(4)
)
results_df.to_csv(result_file,index=False)

print("\n"+"="*70)
print("MODEL COMPARISON")
print("="*70)

print(
    results_df.to_string(
        index=False
    )
)

print("\n"+"="*70)
print("TRAINING COMPLETE")
print("="*70)
print("\nGenerated files:")
print(f"  {test_file_name}")
print(f"  {result_file}")
for filename in model_filenames.values():
    print(
        f"  {model_directory}/{filename}"
    )

