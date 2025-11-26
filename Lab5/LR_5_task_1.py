import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(path: Path):
    """Завантаження даних з файлу CSV (x1, x2, class)."""
    data = np.loadtxt(path, delimiter=',', dtype=float)
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y

def plot_points(ax, X, y, title):
    """Візуалізація точок різних класів."""
    markers = ['s', 'o', '^', 'x', 'D', 'P']
    for i, c in enumerate(np.unique(y)):
        ax.scatter(
            X[y == c, 0], X[y == c, 1],
            s=45, marker=markers[i % len(markers)],
            label=f'class {c}', edgecolor='k'
        )
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend(loc='best')

def plot_decision_surface(ax, clf, X, y, title):
    """Побудова меж класифікації для моделі."""
    pad = 0.7
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25)
    plot_points(ax, X, y, title)

def build_classifier(clf_type: str, n_estimators: int, max_depth: int, random_state: int):
    """Створення класифікатора RandomForest або ExtraTrees."""
    if clf_type == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        ), "Random Forest"
    else:
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        ), "Extra Trees (Extremely Randomized Trees)"

def evaluate_model(clf, Xte, yte, name: str):
    """Оцінка моделі: матриця плутанини, звіт, ймовірності."""
    y_pred = clf.predict(Xte)
    print(f"\n=== {name} ===")
    print("Confusion matrix:\n", confusion_matrix(yte, y_pred))
    print("\nClassification report:\n", classification_report(yte, y_pred, digits=4))

    test_pts = np.array([
        [Xte[:, 0].mean(), Xte[:, 1].mean()],
        [Xte[:, 0].min(), Xte[:, 1].min()],
        [Xte[:, 0].max(), Xte[:, 1].max()]
    ])
    proba = clf.predict_proba(test_pts)
    print("\nProbability estimates for test points:")
    for i, p in enumerate(proba):
        print(f"  pt{i+1} {test_pts[i]} -> {p}")

def main():
    parser = argparse.ArgumentParser(description="RF vs ExtraTrees (Завдання 2.1)")
    parser.add_argument("--data", type=str, default="Lab5/data_random_forests.txt", help="шлях до файлу з даними")
    parser.add_argument("--clf", type=str, choices=["rf", "erf"], default="rf", help="вибір класифікатора: rf або erf")
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=1)
    args = parser.parse_args()

    X, y = load_data(Path(args.data))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=args.random_state, stratify=y)

    clf, name = build_classifier(args.clf, args.n_estimators, args.max_depth, args.random_state)
    clf.fit(Xtr, ytr)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    plot_points(axs[0], X, y, "Вхідні дані (класи)")
    plot_decision_surface(axs[1], clf, Xtr, ytr, f"Межі класифікації — {name}")
    plt.tight_layout()
    plt.show()

    evaluate_model(clf, Xte, yte, name)

if __name__ == "__main__":
    main()