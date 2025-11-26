import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_FILE = "Lab5/traffic_data.txt"

def read_any_csv(path: Path) -> pd.DataFrame:
    """Зчитування CSV з будь-яким роздільником, очищення назв колонок та рядків."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def find_target_column(df: pd.DataFrame) -> str:
    """Автоматичний пошук цільової колонки (кількість авто)."""
    cols = df.columns.tolist()
    if "vehicles" in cols:
        return "vehicles"

    pat = re.compile(r"(veh|vehicle|volume|count|traffic)")
    candidates = [c for c in cols if pat.search(c)]
    for c in candidates:
        if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9:
            return c

    numeric_like = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    if numeric_like:
        return numeric_like[-1]

    raise ValueError(f"Не знайдено цільову колонку з кількістю авто. Колонки: {cols}")

def load_and_encode(path: Path):
    """Завантаження даних, визначення цільової змінної, кодування категоріальних ознак."""
    df = read_any_csv(path)
    target_col = find_target_column(df)

    y = pd.to_numeric(df[target_col], errors="coerce").values.astype(float)
    feature_cols = [c for c in df.columns if c != target_col]

    X_cols_cat, X_cols_num = [], []
    for c in feature_cols:
        ratio_num = pd.to_numeric(df[c], errors="coerce").notna().mean()
        if ratio_num > 0.9:
            X_cols_num.append(c)
        else:
            X_cols_cat.append(c)

    encoders = {}
    X_cat_parts = []
    for c in X_cols_cat:
        le = LabelEncoder()
        Xc = le.fit_transform(df[c].astype(str))
        encoders[c] = le
        X_cat_parts.append(Xc.reshape(-1, 1))
    X_cat = np.hstack(X_cat_parts) if X_cat_parts else np.empty((len(df), 0))

    X_num = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in X_cols_num}).values \
        if X_cols_num else np.empty((len(df), 0))

    X = np.hstack([X_cat, X_num])

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]

    print(f"Визначена цільова колонка: '{target_col}'")
    print(f"Категоріальні ознаки: {X_cols_cat if X_cols_cat else '—'}")
    print(f"Числові ознаки: {X_cols_num if X_cols_num else '—'}")
    print(f"Форма X: {X.shape}, y: {y.shape}")

    return X, y, encoders, X_cols_cat, X_cols_num

def evaluate_model(model, Xte, yte, y_pred):
    """Вивід метрик якості моделі."""
    print(f"R2={r2_score(yte, y_pred):.4f}  "
          f"MAE={mean_absolute_error(yte, y_pred):.4f}  "
          f"MSE={mean_squared_error(yte, y_pred):.4f}")

def main():
    X, y, encoders, cats, nums = load_and_encode(Path(DATA_FILE))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)

    model = ExtraTreesRegressor(n_estimators=300, max_depth=None, random_state=1)
    model.fit(Xtr, ytr)

    y_pred = model.predict(Xte)
    evaluate_model(model, Xte, yte, y_pred)

if __name__ == "__main__":
    main()