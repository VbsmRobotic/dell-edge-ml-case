#!/usr/bin/env python3
"""
Hard Drive Anomaly Detection using One-Class SVM (Unsupervised, Real-Time Friendly)
"""

import os
import time
import numpy as np
import pandas as pd
import gdown
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
FILE_ID = "1f1SA_MzgA9kwSBvs-stfyGzJE1yPaLsd"
BASE_URL = "https://drive.google.com/uc?id=" + FILE_ID

# Define data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Define output directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "supervised")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define path to save model
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

train_days = ["2025-01-01.csv", "2025-01-02.csv", "2025-01-03.csv"]
test_day = "2025-01-08.csv"

selected_features = ['smart_1_raw', 'smart_5_raw', 'smart_9_raw', 'smart_194_raw', 'smart_197_raw']
columns_to_use = ['failure'] + selected_features

# ------------------------------------------------------------------------------
# STEP 1: DOWNLOAD DATA IF MISSING
# ------------------------------------------------------------------------------
# for file_name in train_days + [test_day]:
#     if not os.path.exists(file_name):
#         print(f"Downloading {file_name}...")
#         gdown.download(url=BASE_URL, output=file_name, quiet=False)
for file_name in train_days + [test_day]:
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"Downloading {file_name} to {file_path}...")
        gdown.download(url=BASE_URL, output=file_path, quiet=False)

# ------------------------------------------------------------------------------
# STEP 2: DATA LOADING + CLEANING
# ------------------------------------------------------------------------------
def load_data(files):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=columns_to_use)
            df['failure'] = pd.to_numeric(df['failure'], errors='coerce').fillna(0).astype(int)
            for col in selected_features:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return pd.concat(dfs, ignore_index=True)

print("Loading training (healthy only)...")
df_train = load_data(train_days)
df_train = df_train[df_train['failure'] == 0]  # use only normal samples

print("Loading test...")
df_test = load_data([test_day])

X_train = df_train[selected_features].values
X_test = df_test[selected_features].values
y_test = df_test['failure'].values

# ------------------------------------------------------------------------------
# STEP 3: NORMALIZATION
# ------------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------------------
# STEP 4: ONE-CLASS SVM TRAINING
# ------------------------------------------------------------------------------
print("Training One-Class SVM on normal behavior...")
svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)  # nu: expected outlier fraction
svm.fit(X_train_scaled)

# ------------------------------------------------------------------------------
# STEP 5: REAL-TIME STYLE PREDICTION
# ------------------------------------------------------------------------------
print("Predicting on test data (anomaly detection)...")
y_pred = svm.predict(X_test_scaled)  # +1 = inlier, -1 = outlier
y_pred_bin = (y_pred == -1).astype(int)  # 1 = anomaly (potential failure)

# ------------------------------------------------------------------------------
# STEP 6: EVALUATION
# ------------------------------------------------------------------------------
print("\nTest Evaluation (Ground Truth Used Only for Evaluation):")
print(classification_report(y_test, y_pred_bin, digits=4))

cm = confusion_matrix(y_test, y_pred_bin)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Failure'])
plt.title("Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "svm_confusion_matrix.png"))
plt.close()

# ------------------------------------------------------------------------------
# STEP 7: MODEL SAVING
# ------------------------------------------------------------------------------
# joblib.dump(svm, 'one_class_svm_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# Define paths for saving
svm_model_path = os.path.join(MODELS_DIR, "one_class_svm_model.pkl")
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

# Save the SVM model and scaler
print(f"Saving One-Class SVM model to '{svm_model_path}'...")
joblib.dump(svm, svm_model_path)

print(f"Saving scaler to '{scaler_path}'...")
joblib.dump(scaler, scaler_path)
print("Model saved as 'one_class_svm_model.pkl' and scaler as 'scaler.pkl'")

print("\n✅ Script completed (unsupervised + real-time-friendly model ready).")
