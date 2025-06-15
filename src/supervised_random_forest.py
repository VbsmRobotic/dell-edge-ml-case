#!/usr/bin/env python3
"""
Hard Drive Failure Prediction using Regularized Random Forest
This script trains a Random Forest classifier with anti-overfitting techniques
to predict hard drive failures using SMART attributes.
"""

import os
import time
import numpy as np
import pandas as pd
import gdown
import joblib
import matplotlib.pyplot as plt
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

# ------------------------------------------------------------------------------
# STEP 1: COMMAND-LINE ARGUMENT PARSING
# ------------------------------------------------------------------------------
# Create argument parser to handle command-line options
parser = argparse.ArgumentParser(description="Train Random Forest on SMART data")
# Add --debug flag for development/testing with smaller dataset
parser.add_argument('--debug', action='store_true', help='Enable fast mode with limited data for quick debugging')
# Parse arguments from command line
args = parser.parse_args()

# ------------------------------------------------------------------------------
# STEP 2: CONSTANTS AND DATASET CONFIGURATION
# ------------------------------------------------------------------------------
# Google Drive file ID for the dataset
FILE_ID = "1f1SA_MzgA9kwSBvs-stfyGzJE1yPaLsd"
# Construct download URL
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

# Define dataset files
train_days = ["2025-01-01.csv", "2025-01-02.csv", "2025-01-03.csv",
              "2025-01-04.csv", "2025-01-05.csv", "2025-01-06.csv"]  # 6 days of training data
val_days = ["2025-01-07.csv"]  # 1 day of validation data
test_day = "2025-01-08.csv"    # 1 day of test data

# Combine all required files
all_files = train_days + val_days + [test_day]

# Feature selection - SMART attributes known to be predictive of failure
selected_features = ['smart_1_raw', 'smart_5_raw', 'smart_9_raw', 'smart_194_raw', 'smart_197_raw']
# Columns to load from CSV files (target + features)
columns_to_use = ['failure'] + selected_features

# ------------------------------------------------------------------------------
# STEP 3: DOWNLOAD MISSING DATASET FILES
# ------------------------------------------------------------------------------
# Check and download any missing files from Google Drive
for file_name in all_files:
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"Downloading {file_name} to {file_path}...")
        gdown.download(url=BASE_URL, output=file_path, quiet=False)
# for file_name in all_files:
#     if not os.path.exists(file_name):
#         print(f"Downloading {file_name}...")
#         # Download file using gdown library
#         gdown.download(url=BASE_URL, output=file_name, quiet=False)

# ------------------------------------------------------------------------------
# STEP 4: DATA LOADING AND CLEANING FUNCTIONS
# ------------------------------------------------------------------------------
def load_filtered_data(file_list):
    """
    Load CSV files while handling errors and converting failure column
    Args:
        file_list: List of CSV files to load
    Returns:
        Concatenated DataFrame of all files
    """
    dfs = []  # List to store individual DataFrames
    for f in file_list:
        try:
            # Read CSV with only selected columns
            df = pd.read_csv(f, usecols=columns_to_use)
            # Convert 'failure' to numeric, handle errors, fill missing with 0, convert to int
            df['failure'] = pd.to_numeric(df['failure'], errors='coerce').fillna(0).astype(int)
            dfs.append(df)  # Add to list
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    # Combine all DataFrames into one
    return pd.concat(dfs, ignore_index=True)

def clean_data(df):
    """
    Clean data by converting features to numeric and handling infinite/NaN values
    Args:
        df: Input DataFrame
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()  # Create copy to avoid modifying original
    # Convert selected features to numeric
    for col in selected_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with missing values in features or target
    df.dropna(subset=selected_features + ['failure'], inplace=True)
    return df

# ------------------------------------------------------------------------------
# STEP 5: DATA LOADING AND PREPROCESSING
# ------------------------------------------------------------------------------
print("Loading training data...")
# Load and clean training data
train_df = clean_data(load_filtered_data(train_days))

# If debug mode enabled, use small subset for faster iteration
if args.debug:
    print("[DEBUG MODE] Using only 1000 samples for training.")
    train_df = train_df.sample(n=1000, random_state=42)  # Fixed seed for reproducibility
print(f"Training samples: {len(train_df)}")

print("Loading validation data...")
val_df = clean_data(load_filtered_data(val_days))

print("Loading test data...")
test_df = clean_data(load_filtered_data([test_day]))

# ------------------------------------------------------------------------------
# STEP 6: FEATURE/TARGET SEPARATION
# ------------------------------------------------------------------------------
# Prepare feature matrix (X) and target vector (y) for all datasets
X_train = train_df[selected_features]  # Training features
y_train = train_df['failure']          # Training target

X_val = val_df[selected_features]      # Validation features
y_val = val_df['failure']              # Validation target

X_test = test_df[selected_features]    # Test features
y_test = test_df['failure']            # Test target

# ------------------------------------------------------------------------------
# STEP 7: MODEL TRAINING WITH REGULARIZATION
# ------------------------------------------------------------------------------
print("\nTraining Regularized Random Forest to prevent overfitting...")
start_time = time.time()  # Start timer

# Configure Random Forest with anti-overfitting parameters
clf = RandomForestClassifier(
    n_estimators=100,          # Number of trees in the forest
    max_depth=8,               # Maximum depth of trees - limits complexity
    min_samples_split=10,      # Minimum samples required to split a node
    min_samples_leaf=5,        # Minimum samples required at each leaf node
    max_features='sqrt',       # Features to consider: sqrt(total features)
    class_weight='balanced',   # Adjust weights for imbalanced classes
    random_state=42,           # Seed for reproducibility
    n_jobs=-1                  # Use all available CPU cores
)
# Train the model
clf.fit(X_train, y_train)

# Calculate and print training time
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# ------------------------------------------------------------------------------
# STEP 8: METRIC TRACKING DURING INCREMENTAL TRAINING
# ------------------------------------------------------------------------------
print("\nTracking metrics vs. number of trees with regularization...")

# Lists to store metrics at different tree counts
train_accuracies = []  # Training accuracy at each tree count
val_accuracies = []    # Validation accuracy
train_losses = []      # Training loss (1 - accuracy)
val_losses = []        # Validation loss
val_precisions = []    # Validation precision
val_recalls = []       # Validation recall
val_f1s = []           # Validation F1 score

# Tree counts to evaluate (10, 20, ..., 100)
estimator_range = list(range(10, 110, 10))

# Train models with increasing tree counts
for n in estimator_range:
    # Configure model with current tree count (keeping other regularization params)
    model = RandomForestClassifier(
        n_estimators=n,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    # Train model with n trees
    model.fit(X_train, y_train)

    # Training metrics
    train_pred = model.predict(X_train)            # Predictions on training set
    train_acc = model.score(X_train, y_train)      # Training accuracy
    train_loss = 1 - train_acc                     # Training loss
    
    # Validation metrics
    val_pred = model.predict(X_val)                # Predictions on validation set
    val_acc = model.score(X_val, y_val)            # Validation accuracy
    val_loss = 1 - val_acc                         # Validation loss
    
    # Calculate precision, recall, F1 for validation set
    precision = precision_score(y_val, val_pred)   # Precision: TP/(TP+FP)
    recall = recall_score(y_val, val_pred)         # Recall: TP/(TP+FN)
    f1 = f1_score(y_val, val_pred)                 # F1: harmonic mean of precision/recall

    # Store metrics
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)
    
    # Print progress
    print(f"Trees: {n:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {f1:.4f}")

# ------------------------------------------------------------------------------
# STEP 9: MODEL SELECTION AND FINAL TRAINING
# ------------------------------------------------------------------------------
# Find index of best validation F1 score
best_idx = np.argmax(val_f1s)
print(f"\nSelected model with {estimator_range[best_idx]} trees (best validation F1: {val_f1s[best_idx]:.4f})")

# Train final model with optimal tree count
clf = RandomForestClassifier(
    n_estimators=estimator_range[best_idx],  # Use best-performing tree count
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
).fit(X_train, y_train)  # Train immediately

# ------------------------------------------------------------------------------
# STEP 10: VISUALIZATION OF TRAINING METRICS
# ------------------------------------------------------------------------------
# Create figure for accuracy/loss plot
plt.figure(figsize=(10, 6))
# Plot training accuracy (solid blue line)
plt.plot(estimator_range, train_accuracies, 'b-', label='Train Accuracy')
# Plot validation accuracy (solid green line)
plt.plot(estimator_range, val_accuracies, 'g-', label='Validation Accuracy')
# Plot training loss (dashed blue line)
plt.plot(estimator_range, train_losses, 'b--', label='Train Loss (1-Acc)')
# Plot validation loss (dashed green line)
plt.plot(estimator_range, val_losses, 'g--', label='Validation Loss (1-Acc)')
plt.xlabel("Number of Trees")
plt.ylabel("Metric Value")
plt.title("Train vs Validation Metrics with Regularization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "train_val_metrics.png"))  # Save plot
plt.close()  # Close figure to free memory

# Create figure for precision/recall/F1 plot
plt.figure(figsize=(10, 6))
# Plot validation precision (red line)
plt.plot(estimator_range, val_precisions, 'r-', label='Precision')
# Plot validation recall (blue line)
plt.plot(estimator_range, val_recalls, 'b-', label='Recall')
# Plot validation F1 score (green line)
plt.plot(estimator_range, val_f1s, 'g-', label='F1 Score')
# Add vertical line at best tree count
plt.axvline(x=estimator_range[best_idx], color='gray', linestyle='--', 
            label=f'Best: {estimator_range[best_idx]} trees')
plt.xlabel("Number of Trees")
plt.ylabel("Score")
plt.title("Validation Metrics vs Tree Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "val_metrics_vs_trees.png"))  # Save plot
plt.close()

# ------------------------------------------------------------------------------
# STEP 11: MODEL EVALUATION AND REPORTING
# ------------------------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    """
    Evaluate model performance and generate reports
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        label: Dataset name for reporting
    """
    # Print classification report with 4-digit precision
    print(f"\n{label} Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Create heatmap visualization
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Failure', 'Failure'], 
                yticklabels=['No Failure', 'Failure'])
    plt.title(f"{label} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    # Save confusion matrix plot
    # plt.savefig(f"{label.lower().replace(' ', '_')}_confusion_matrix.png")
    # Format filename and save in correct folder
    filename = f"{label.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

print("\nFinal Model Evaluation:")
# Evaluate on validation set
val_pred = clf.predict(X_val)
evaluate(y_val, val_pred, "Validation")

# Evaluate on test set (unseen during training/validation)
test_pred = clf.predict(X_test)
evaluate(y_test, test_pred, "Test")

# ------------------------------------------------------------------------------
# STEP 12: MODEL PERSISTENCE
# ------------------------------------------------------------------------------
# print("Saving model to 'random_forest_model.pkl'...")
# # Serialize and save trained model
# joblib.dump(clf, 'random_forest_model.pkl')

model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
print(f"Saving model to '{model_path}'...")
# Serialize and save the trained model
joblib.dump(clf, model_path)

# ------------------------------------------------------------------------------
# STEP 13: FEATURE IMPORTANCE VISUALIZATION
# ------------------------------------------------------------------------------
# Extract feature importances from trained model
importances = clf.feature_importances_
# Get indices that would sort importances in descending order
indices = np.argsort(importances)[::-1]

# Create feature importance plot
plt.figure(figsize=(8, 4))
plt.title("Feature Importances")
# Create bar chart of feature importances
plt.bar(range(len(selected_features)), importances[indices], color="skyblue")
# Set x-axis labels to feature names
plt.xticks(range(len(selected_features)), [selected_features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"))  # Save plot
plt.close()

print("\nScript completed successfully!")