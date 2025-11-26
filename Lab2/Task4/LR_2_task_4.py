import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

FILE = "Lab2\Task4\income_data.txt"
SEED = 42

def can_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False

def prepare_dataset(path: str):
    data, labels = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != 15 or "?" in parts:
                continue
            data.append(parts[:-1])
            labels.append(parts[-1])
    data = np.array(data, dtype=object)
    labels = np.array(labels)

    encoded = np.zeros(data.shape, dtype=float)
    for idx in range(data.shape[1]):
        col = data[:, idx]
        if all(can_float(v) for v in col):
            encoded[:, idx] = col.astype(float)
        else:
            le = LabelEncoder()
            encoded[:, idx] = le.fit_transform(col)
    return encoded, labels

def test_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
    }

X, y = prepare_dataset(FILE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

classifiers = {
    "LogReg": LogisticRegression(max_iter=1000),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "DecisionTree": DecisionTreeClassifier(random_state=SEED),
    "NaiveBayes": GaussianNB(),
    "SVM_RBF": SVC(kernel="rbf", gamma="scale", random_state=SEED),
}

scores = {name: test_model(clf, X_train, X_test, y_train, y_test)
          for name, clf in classifiers.items()}

print("\n=== Порівняння алгоритмів (за F1-score) ===")
for name, metrics in sorted(scores.items(), key=lambda kv: kv[1]["f1"], reverse=True):
    print(f"{name:15s} acc={metrics['accuracy']:.4f} "
          f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}")