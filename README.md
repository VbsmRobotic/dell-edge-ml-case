# 📦 dell-edge-ml-case

This repository demonstrates both **supervised** and **unsupervised** anomaly detection techniques for edge-based machine learning applications in industrial settings.

Two main approaches have been implemented and are available in the [`develop_version1`](https://github.com/VbsmRobotic/dell-edge-ml-case.git) branch:

- ✅ In the supervised approach, a Random Forest algorithm is trained as a classifier using labeled data.
- ⚙️ In the unsupervised approach, a One-Class SVM is used for anomaly detection on normal-only data.

The branch contains:
- Model training scripts
- Data preprocessing
- Evaluation results
- A structured comparison between both approaches


## 📂 Repository Structure
```
develop_version1/
	└── src/
		├── supervised_random_forest.py # Random Forest supervised classification
		|
		├── unsupervised_oneclass_svm.py # One-Class SVM anomaly detection
		|
		├── data/ # Sample data files
		│ 	└── # (downloaded via gdown from Google Drive)
		|
		├── models/ # Saved models
		│ 	├── random_forest_model.pkl
		│ 	└── svm_model.pkl
		|
		└── results/ # Visualizations, evaluation metrics
			├── supervised/
			└── unsupervised/
```

---

## 📘 Methods Overview

### 🧠 1. Supervised Learning – Random Forest

| Aspect              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Type**            | Supervised                                                                  |
| **Training Data**   | Requires both features and labels (e.g., `failure = 0 or 1`)                 |
| **Goal**            | Learn to classify between “normal” and “failure”                             |
| **Model Output**    | A trained classifier that predicts failure labels                            |
| **Evaluation**      | Accuracy, Precision, Recall, F1 Score, Confusion Matrix                      |
| **Use Case**        | When failure labels are available in historical logs                         |
| **Pros**            | High accuracy if good labels are provided                                    |
| **Cons**            | Requires many labeled failure cases, which may be rare                      |

> ✅ **Best used when you have labeled failure data from historical logs**

---

### ⚙️ 2. Unsupervised Learning – One-Class SVM

| Aspect              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Type**            | Unsupervised (Semi-supervised Anomaly Detection)                            |
| **Training Data**   | Uses only healthy (normal) data                                              |
| **Goal**            | Learn the boundary of “normal” operation to detect anomalies                |
| **Model Output**    | An anomaly detector: determines whether a new datapoint is abnormal          |
| **Evaluation**      | Labels are optional and used only for evaluation during testing              |
| **Use Case**        | When failure data is rare or unlabeled                                       |
| **Pros**            | Works even without failure labels; robust for rare-event detection           |
| **Cons**            | May flag any deviation as anomaly, not necessarily a true failure            |
| **Speed**           | Lightweight and typically more real-time friendly                           |

> ✅ **Ideal for early-stage deployments where failure data is not available**

---

## 🔍 Real-World Analogy

- **Random Forest (Supervised):**  
  Like a doctor trained on many patients with known diseases and symptoms — diagnoses based on past cases.

- **One-Class SVM (Unsupervised):**  
  Like a security guard trained to recognize what’s normal — alerts when something unusual happens.

---

## 📝 Summary Comparison

| Feature                     | Random Forest (Supervised)        | One-Class SVM (Unsupervised)     |
|-----------------------------|-----------------------------------|----------------------------------|
| **Training Labels Required**| ✅ Yes                             | ❌ No                             |
| **Detection Goal**          | Explicit classification (normal/failure) | Anomaly detection only     |
| **Real-Time Suitability**   | ⚠️ Possible, but heavier           | ✅ Yes, lightweight               |
| **Handles Rare Failures**   | ❌ Needs sufficient examples       | ✅ Yes                            |
| **Model Interpretability**  | ✅ Easy to understand              | ⚠️ Can be harder to explain       |

---

