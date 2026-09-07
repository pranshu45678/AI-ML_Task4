# CodeAlpha AI & ML Internship — Task 4
## Classification with Logistic Regression

### Objective
Build a binary classifier using logistic regression.

### Dataset
Breast Cancer Wisconsin Diagnostic dataset from scikit-learn.
- Samples: 569
- Features: 30
- Binary target: malignant / benign

### Implementation
1. Choose a binary classification dataset.
2. Split into training and test sets.
3. Standardize features using StandardScaler.
4. Fit LogisticRegression.
5. Evaluate with confusion matrix, precision, recall and ROC-AUC.
6. Tune the classification threshold.
7. Plot the ROC curve and sigmoid function.

### Results
- Accuracy: 0.9825
- Precision: 0.9861
- Recall: 0.9861
- ROC-AUC: 0.9954

### Files
- `logistic_regression.py` — complete implementation
- `breast_cancer.csv` — dataset
- `confusion_matrix.png`
- `roc_curve.png`
- `sigmoid_curve.png`
- `threshold_analysis.csv`
- `results.txt`
- `interview_questions.txt`
- `requirements.txt`

### Run
```bash
pip install -r requirements.txt
python logistic_regression.py
```

### Sigmoid explanation
The sigmoid function converts a model score into a value between 0 and 1, which can be interpreted as a probability for the positive class. A threshold such as 0.5 converts that probability into a class prediction.
