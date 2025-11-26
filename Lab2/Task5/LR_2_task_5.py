import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

RANDOM_STATE = 0

iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeClassifier(tol=1e-2, solver="sag", random_state=RANDOM_STATE))
])
clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)

print("Accuracy:", accuracy_score(ytest, ypred))
print("Precision (weighted):", precision_score(ytest, ypred, average="weighted"))
print("Recall (weighted):", recall_score(ytest, ypred, average="weighted"))
print("F1 Score (weighted):", f1_score(ytest, ypred, average="weighted"))
print("Cohen Kappa:", cohen_kappa_score(ytest, ypred))
print("Matthews Corrcoef:", matthews_corrcoef(ytest, ypred))
print("\nClassification Report:\n", classification_report(ytest, ypred, target_names=target_names))

cm = confusion_matrix(ytest, ypred)
plt.figure(figsize=(5.2, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("RidgeClassifier — Confusion Matrix (Iris)")
plt.tight_layout()
plt.show()