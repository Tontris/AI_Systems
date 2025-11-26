import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import __version__ as skl_version

def make_adaboost_regressor(base, n_estimators=200, learning_rate=0.1, random_state=1):
    """
    Створення AdaBoostRegressor з урахуванням різних версій scikit-learn.
    У нових версіях використовується параметр `estimator`, у старих — `base_estimator`.
    """
    try:
        return AdaBoostRegressor(
            estimator=base,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
    except TypeError:
        return AdaBoostRegressor(
            base_estimator=base,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )

def evaluate_model(model, Xte, yte):
    """Обчислення метрик R2, MAE, MSE."""
    y_pred = model.predict(Xte)
    r2 = r2_score(yte, y_pred)
    mae = mean_absolute_error(yte, y_pred)
    mse = mean_squared_error(yte, y_pred)
    print(f"R2={r2:.4f}  MAE={mae:.4f}  MSE={mse:.4f}")
    return y_pred

def plot_feature_importances(model, feature_names):
    """Візуалізація важливості ознак."""
    if not hasattr(model, "feature_importances_"):
        print("⚠️ Модель не має атрибуту feature_importances_.")
        return

    imp = model.feature_importances_
    imp = imp / imp.sum()
    order = np.argsort(imp)[::-1]

    plt.figure(figsize=(8, 4.5))
    plt.bar(range(len(imp)), imp[order])
    plt.xticks(range(len(imp)), [feature_names[i] for i in order],
               rotation=45, ha='right')
    plt.ylabel("Відносна важливість")
    plt.title("Важливість ознак (AdaBoostRegressor)")
    plt.tight_layout()
    plt.show()


def main():
    data = fetch_california_housing()
    X, y = data.data, data.target
    names = data.feature_names

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)

    base = DecisionTreeRegressor(max_depth=4, random_state=1)

    model = make_adaboost_regressor(base, n_estimators=200, learning_rate=0.1, random_state=1)
    print(f"scikit-learn version: {skl_version}")
    model.fit(Xtr, ytr)

    evaluate_model(model, Xte, yte)

    plot_feature_importances(model, names)

if __name__ == "__main__":
    main()