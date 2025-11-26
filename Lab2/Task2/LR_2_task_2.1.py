import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_FILE = "Lab2\Task2\income_data.txt"
SEED = 42
ROW_LIMIT = None

def try_float(x: str) -> bool:
    """Перевіряє чи значення можна перетворити у float."""
    try:
        float(x)
        return True
    except:
        return False

start = time.time()
features, targets = [], []

with open(DATA_FILE) as fh:
    for idx, line in enumerate(fh):
        if ROW_LIMIT and idx >= ROW_LIMIT:
            break
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 15 or "?" in parts:
            continue
        features.append(parts[:-1])
        targets.append(parts[-1])

print(f"[INFO] Рядків після очищення: {len(features)} (час {time.time()-start:.2f}s)")

features = np.array(features, dtype=object)
targets = np.array(targets)
encoded = np.zeros(features.shape, dtype=float)
encoders = []

for col_idx in range(features.shape[1]):
    column = features[:, col_idx]
    if all(try_float(v) for v in column):
        encoded[:, col_idx] = column.astype(float)
        encoders.append(None)
    else:
        le = LabelEncoder()
        encoded[:, col_idx] = le.fit_transform(column)
        encoders.append(le)
X_train, X_test, y_train, y_test = train_test_split(
    encoded, targets, test_size=0.2, random_state=SEED, stratify=targets
)
print(f"[INFO] Розмір train: {X_train.shape}, test: {X_test.shape}")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="poly",
        degree=3,
        C=1.0,
        gamma="scale",
        random_state=SEED,
        cache_size=1000,
        verbose=False
    ))
])

train_start = time.time()
model.fit(X_train, y_train)
print(f"[INFO] Навчання завершено за {time.time()-train_start:.2f}s")
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision (weighted):", round(precision_score(y_test, y_pred, average="weighted"), 4))
print("Recall (weighted):", round(recall_score(y_test, y_pred, average="weighted"), 4))
print("F1 (weighted):", round(f1_score(y_test, y_pred, average="weighted"), 4))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))