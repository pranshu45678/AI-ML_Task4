import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score

# Load binary classification dataset
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target)

# Train/test split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=.2,random_state=42,stratify=y
)

# Standardize features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Logistic Regression
model=LogisticRegression(max_iter=5000,random_state=42)
model.fit(X_train,y_train)

# Probabilities and default threshold
prob=model.predict_proba(X_test)[:,1]
pred=(prob>=.5).astype(int)

print("Accuracy:",accuracy_score(y_test,pred))
print("Precision:",precision_score(y_test,pred))
print("Recall:",recall_score(y_test,pred))
print("ROC-AUC:",roc_auc_score(y_test,prob))
print("Confusion Matrix:")
print(confusion_matrix(y_test,pred))

# Confusion matrix visualization
ConfusionMatrixDisplay.from_predictions(
    y_test,pred,display_labels=data.target_names
)
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout(); plt.savefig("confusion_matrix.png",dpi=150); plt.show()

# ROC curve
fpr,tpr,_=roc_curve(y_test,prob)
auc=roc_auc_score(y_test,prob)
plt.figure(figsize=(7,5))
plt.plot(fpr,tpr,label=f"ROC-AUC = {auc:.4f}")
plt.plot([0,1],[0,1],"--",label="Random classifier")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve"); plt.legend()
plt.tight_layout(); plt.savefig("roc_curve.png",dpi=150); plt.show()

# Threshold tuning
for threshold in [.30,.40,.50,.60,.70]:
    p=(prob>=threshold).astype(int)
    print(
        f"Threshold {threshold:.2f}: "
        f"Accuracy={accuracy_score(y_test,p):.4f}, "
        f"Precision={precision_score(y_test,p,zero_division=0):.4f}, "
        f"Recall={recall_score(y_test,p,zero_division=0):.4f}"
    )

# Sigmoid function
import numpy as np
z=np.linspace(-8,8,400)
sigmoid=1/(1+np.exp(-z))
plt.figure(figsize=(7,5))
plt.plot(z,sigmoid)
plt.xlabel("z"); plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.axhline(.5,linestyle="--")
plt.tight_layout(); plt.savefig("sigmoid_curve.png",dpi=150); plt.show()
