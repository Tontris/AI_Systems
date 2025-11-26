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

FILE_PATH = "Lab2\Task2\income_data.txt"
SEED = 42
ROW_LIMIT = None

def can_be_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False

start_time = time.time()
data, labels = [], []

with open(FILE_PATH, encoding="utf-8") as file:
    for idx, line in enumerate(file):
        if ROW_LIMIT and idx >= ROW_LIMIT:
            break
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 15 or "?" in parts:
            continue
        data.append(parts[:-1])
        labels.append(parts[-1])

print(f"[INFO] Отримано {len(data)} рядків за {time.time()-start_time:.2f} сек.")

data = np.array(data, dtype=object)
labels = np.array(labels)

encoded_data = np.zeros(data.shape, dtype=float)
encoders = []

for col_idx in range(data.shape[1]):
    column = data[:, col_idx]
    if all(can_be_float(v) for v in column):
        encoded_data[:, col_idx] = column.astype(float)
        encoders.append(None)
    else:
        le = LabelEncoder()
        encoded_data[:, col_idx] = le.fit_transform(column)
        encoders.append(le)

X_train, X_test, y_train, y_test = train_test_split(
    encoded_data, labels, test_size=0.2, random_state=SEED, stratify=labels
)
print(f"[INFO] Навчальні дані: {X_train.shape}, Тестові дані: {X_test.shape}")

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=SEED
)

train_start = time.time()
svm_model.fit(X_train, y_train)
print(f"[INFO] Час навчання: {time.time()-train_start:.2f} сек.")

y_pred = svm_model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision (weighted):", round(precision_score(y_test, y_pred, average="weighted"), 4))
print("Recall (weighted):", round(recall_score(y_test, y_pred, average="weighted"), 4))
print("F1 (weighted):", round(f1_score(y_test, y_pred, average="weighted"), 4))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))