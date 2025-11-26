import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

svm_clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))