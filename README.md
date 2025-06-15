📊 Hard Drive Failure Prediction using Random Forest
📌 Project Overview
This project implements a supervised machine learning pipeline using a Random Forest classifier to predict hard drive failures based on SMART (Self-Monitoring, Analysis, and Reporting Technology) attributes.

The model incorporates anti-overfitting regularization techniques and includes visualizations for monitoring performance across training, validation, and testing phases.

✅ Features
SMART feature selection for failure prediction

Regularization to prevent overfitting

Accuracy/loss tracking during training

Precision, Recall, and F1-score visualization

Feature importance analysis

Confusion matrix generation

Model persistence using joblib

Automatic dataset download via Google Drive

📂 Dataset
The dataset includes SMART attributes collected over several days:

Dataset Date Range  Duration
Training    Jan 1, 2025 – Jan 6, 2025   6 days
Validation  Jan 7, 2025 1 day
Testing Jan 8, 2025 1 day

Note: The dataset is automatically downloaded using gdown if not found locally.

🧩 Dependencies
Python 3.7+

Required packages:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn gdown joblib
▶️ How to Run
Clone the repository

Install the required dependencies

Run the script

bash
Copy
Edit
# Normal execution
python smart_rf.py

# Debug mode (uses a smaller dataset)
python smart_rf.py --debug
📁 Output Files
File Name   Description
train_val_metrics.png   Training vs. validation accuracy/loss
val_metrics_vs_trees.png    Precision/Recall/F1 vs. number of trees
validation_confusion_matrix.png Confusion matrix on validation set
test_confusion_matrix.png   Confusion matrix on test set
feature_importance.png  Feature importance bar chart
random_forest_model.pkl Serialized Random Forest model

⚙️ Code Explanation
1. Command-line Arguments
python
Copy
Edit
parser.add_argument('--debug', action='store_true', help='Enable fast mode')
Enables fast mode with a reduced dataset for debugging purposes.

2. Data Download
python
Copy
Edit
gdown.download(url=BASE_URL, output=file_name, quiet=False)
Downloads dataset files from Google Drive.

3. Data Loading and Cleaning
python
Copy
Edit
def load_filtered_data(file_list):
    # Loads and processes CSVs, selects columns, converts labels

def clean_data(df):
    # Handles NaNs, infs, and ensures numeric types
4. Model Configuration
python
Copy
Edit
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
Includes all regularization settings for robust training.

5. Metric Tracking and Visualization
python
Copy
Edit
# Tracking lists
train_accuracies = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

# Plot metrics after training
plt.plot(...)
6. Model Selection
python
Copy
Edit
best_idx = np.argmax(val_f1s)
# Retrain using best number of trees
7. Evaluation and Confusion Matrices
python
Copy
Edit
def evaluate(y_true, y_pred, label):
    # Prints report and plots confusion matrix
🧠 Anti-Overfitting Techniques
Technique   Parameter   Purpose
Tree Depth Limitation   max_depth=8 Limits tree complexity
Split Requirement   min_samples_split=10    Avoids over-specific splits
Leaf Size Control   min_samples_leaf=5  Ensures minimum leaf size
Feature Subsampling max_features='sqrt' Improves ensemble diversity
Class Weight Balancing  class_weight='balanced' Addresses class imbalance
Optimal Tree Selection  Based on validation F1  Balances model performance and complexity

📈 Interpreting Results
1. train_val_metrics.png
Solid lines: Accuracy (Train vs. Validation)

Dashed lines: Loss (1 - Accuracy)

Good model: Validation closely follows training accuracy

Overfitting: Large accuracy gap between train and validation

2. val_metrics_vs_trees.png
Precision (Red): Accuracy of predicted failures

Recall (Blue): Captures actual failures

F1 Score (Green): Balance of precision and recall

Dashed vertical line: Selected tree count

3. validation_confusion_matrix.png / test_confusion_matrix.png
Cell    Meaning
Top-left    True Negatives (correct non-failure)
Top-right   False Positives (false alarms)
Bottom-left False Negatives (missed failures)
Bottom-right    True Positives (correct failures)

4. feature_importance.png
Bar chart showing importance of each SMART attribute

Helps identify which metrics best predict failure

🧪 Expected Output
text
Copy
Edit
Loading training data...
Training samples: 15000
Loading validation data...
Loading test data...
Training Regularized Random Forest to prevent overfitting...
Training completed in 15.32 seconds.

Tracking metrics vs. number of trees with regularization...
Trees:  10 | Train Acc: 0.9820 | Val Acc: 0.9805 | Val F1: 0.7500
Trees:  20 | Train Acc: 0.9835 | Val Acc: 0.9810 | Val F1: 0.7692
...
Trees: 100 | Train Acc: 0.9845 | Val Acc: 0.9818 | Val F1: 0.7826

Selected model with 80 trees (best validation F1: 0.7852)

Validation Classification Report:
              precision    recall  f1-score   support
           0     0.9921    0.9990    0.9956      3912
           1     0.9000    0.7000    0.7857        10
    accuracy                         0.9913      3922
   macro avg     0.9460    0.8495    0.8906      3922
weighted avg     0.9910    0.9913    0.9910      3922

Saving model to 'random_forest_model.pkl'...
✅ Conclusion
This project provides a robust and interpretable machine learning pipeline for predicting hard drive failures using SMART attributes. Regularization techniques are integrated to improve generalization, and comprehensive visualizations guide evaluation and feature understanding. The model selection based on F1 score ensures a balanced trade-off between precision (avoiding false alarms) and recall (detecting real failures).