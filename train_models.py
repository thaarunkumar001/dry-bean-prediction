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

from sklear.metrics import(
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoed,
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
    df=pd.read_excle(data_file)
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
    ])
}




