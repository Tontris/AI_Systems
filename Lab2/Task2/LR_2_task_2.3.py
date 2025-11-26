import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = "Lab2\Task2\income_data.txt"
SEED = 42
ROW_LIMIT = None

def is_floatable(val: str) -> bool:
    """Перевіряє чи можна перетворити рядок у float."""
    try:
        float(val)
        return True
    except ValueError:
        return False

start = time.time()
records, targets = [], []

with open(DATA_FILE, encoding="utf-8") as fh:
    for idx, line in enumerate(fh):
        if ROW_LIMIT and idx >= ROW_LIMIT:
            break
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 15 or "?" in parts:
            continue
        records.append(parts[:-1])
        targets.append(parts[-1])

print(f"[INFO] Завантажено {len(records)} рядків за {time.time()-start:.2f} сек.")

records = np.array(records, dtype=object)
targets = np.array(targets)

encoded = np.zeros(records.shape, dtype=float)
encoders = []

for col_idx in range(records.shape[1]):
    column = records[:, col_idx]
    if all(is_floatable(v) for v in column):
        encoded[:, col_idx] = column.astype(float)
        encoders.append(None)
    else:
        le = LabelEncoder()
        encoded[:, col_idx] = le.fit_transform(column)
        encoders.append(le)

X_train, X_test, y_train, y_test = train_test_split(
    encoded, targets, test_size=0.2, random_state=SEED, stratify=targets
)
print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

svm_clf = SVC(
    kernel="sigmoid",
    C=1.0,
    gamma="scale",
    random_state=SEED
)

train_start = time.time()
svm_clf.fit(X_train, y_train)
print(f"[INFO] Навчання завершено за {time.time()-train_start:.2f} сек.")

y_pred = svm_clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision (weighted):", round(precision_score(y_test, y_pred, average="weighted"), 4))
print("Recall (weighted):", round(recall_score(y_test, y_pred, average="weighted"), 4))
print("F1 (weighted):", round(f1_score(y_test, y_pred, average="weighted"), 4))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))