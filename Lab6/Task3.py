import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

URL = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"

def build_target_bins(y_price, bins=2):
    """
    Перетворення цін у категорії (класи).
    bins=2 → дешеві/дорогі (за медіаною).
    bins=3 → три категорії (за квантилями).
    """
    y = pd.Series(y_price).astype(float)
    if bins == 2:
        thr = y.median()
        labels = (y > thr).astype(int)
    elif bins == 3:
        q1, q2 = y.quantile([0.33, 0.66])
        labels = pd.cut(y, bins=[-np.inf, q1, q2, np.inf], labels=[0, 1, 2]).astype(int)
    else:
        raise ValueError("bins must be 2 or 3")
    return labels.values

def clean_renfe(df: pd.DataFrame):
    """Очищення датасету: видалення пропусків, дублікатів, некоректних цін."""
    df = df.copy()
    if "price" not in df.columns:
        raise ValueError(f"Не знайдено колонку 'price'. Поля: {df.columns.tolist()}")
    df = df.dropna(subset=["price"]).drop_duplicates()
    df = df[df["price"] > 0]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def encode_features(df: pd.DataFrame, num_cols, cat_cols, one_hot=True):
    """Кодування ознак: числові залишаються, категоріальні → OneHot або LabelEncoder."""
    X_num = df[num_cols].astype(float).values if num_cols else np.empty((len(df), 0))
    X_cat = None
    if cat_cols:
        if one_hot:
            oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat = oh.fit_transform(df[cat_cols])
        else:
            mats = []
            for c in cat_cols:
                le = LabelEncoder()
                mats.append(le.fit_transform(df[c]).reshape(-1, 1))
            X_cat = np.hstack(mats)
    X = X_num if X_cat is None else (np.hstack([X_num, X_cat]) if X_num.size else X_cat)
    return X

def main():
    parser = argparse.ArgumentParser(description="Naive Bayes для класифікації цін RENFE")
    parser.add_argument("--bins", type=int, default=2, choices=[2, 3], help="бінування ціни: 2 або 3 класи")
    parser.add_argument("--model", type=str, default="gauss", choices=["gauss", "multi"],
                        help="GaussianNB або MultinomialNB")
    parser.add_argument("--scale", action="store_true", help="масштабувати числові ознаки (StandardScaler)")
    args = parser.parse_args()

    df = pd.read_csv(URL)
    df = clean_renfe(df)

    # залишаємо тільки ті колонки, які реально є у датасеті
    cols = ["origin", "destination", "train_type", "train_class", "fare",
            "price", "insert_date", "start_date"]
    df = df[cols].dropna()

    y = build_target_bins(df["price"].values, bins=args.bins)

    num_cols = []  # duration і distance відсутні
    cat_cols = ["origin", "destination", "train_type", "train_class", "fare"]

    X = encode_features(df, num_cols, cat_cols, one_hot=True)

    if args.scale and len(num_cols) > 0:
        scaler = StandardScaler()
        X[:, :len(num_cols)] = scaler.fit_transform(X[:, :len(num_cols)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

    clf = GaussianNB() if args.model == "gauss" else MultinomialNB()
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)

    print("Confusion matrix:\n", confusion_matrix(yte, y_pred))
    print("\nClassification report:\n", classification_report(yte, y_pred, digits=4))

if __name__ == "__main__":
    main()