import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(path: Path):
    """Завантаження даних з файлу CSV (x1, x2, class)."""
    if not path.exists():
        raise FileNotFoundError(f"Файл {path} не знайдено. Перевірте шлях.")
    data = np.loadtxt(path, delimiter=',', dtype=float)
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y


def plot_decision_surface(ax, clf, X, y, title):
    """Побудова меж класифікації для ExtraTrees."""
    pad = 0.7
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25)

    markers = {0: 'o', 1: 's'}
    for lab, mk in markers.items():
        ax.scatter(X[y == lab, 0], X[y == lab, 1],
                   s=45, marker=mk, label=f'class {lab}', edgecolor='k')

    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()

def train_classifier(Xtr, ytr, balance: bool):
    """Створення та навчання ExtraTreesClassifier."""
    kwargs = dict(n_estimators=100, max_depth=7, random_state=1)
    if balance:
        clf = ExtraTreesClassifier(class_weight="balanced", **kwargs)
        suffix = " (balanced)"
    else:
        clf = ExtraTreesClassifier(**kwargs)
        suffix = " (unbalanced)"
    clf.fit(Xtr, ytr)
    return clf, suffix

def evaluate_model(clf, Xte, yte):
    """Оцінка моделі: матриця плутанини та звіт."""
    y_pred = clf.predict(Xte)
    print("Confusion matrix:\n", confusion_matrix(yte, y_pred))
    print("\nClassification report:\n", classification_report(yte, y_pred, digits=4))

def main():
    parser = argparse.ArgumentParser(description="LR_5_task_2 — дисбаланс класів (Завдання 2.2)")
    parser.add_argument("--data", type=str, default="Lab5/data_imbalance.txt", help="шлях до файлу з даними")
    parser.add_argument("--balance", type=str, choices=["off", "on"], default="off", help="використати балансування класів")
    parser.add_argument("--ignore", action="store_true", help="ігнорувати zero-division warnings")
    args = parser.parse_args()

    if args.ignore:
        warnings.filterwarnings("ignore")

    X, y = load_data(Path(args.data))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    clf, suffix = train_classifier(Xtr, ytr, balance=(args.balance == "on"))

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_decision_surface(ax, clf, Xtr, ytr, f"Decision boundary{suffix}")
    plt.tight_layout()
    plt.show()

    evaluate_model(clf, Xte, yte)

if __name__ == "__main__":
    main()