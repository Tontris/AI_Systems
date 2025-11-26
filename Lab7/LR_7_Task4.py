import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

def load_mapping(json_path: Path):
    """Завантаження словника {тикер: назва компанії} з JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    tickers = list(mapping.keys())
    names = [mapping[t] for t in tickers]
    return tickers, names

def try_download_prices(tickers, start, end, interval):
    """Спроба завантажити котирування через yfinance."""
    try:
        import yfinance as yf
    except Exception as e:
        print("[WARN] yfinance не встановлено або недоступне:", e)
        return None
    try:
        df = yf.download(tickers, start=start, end=end, interval=interval, progress=False)["Adj Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df.dropna(how="all")
    except Exception as e:
        print("[WARN] Не вдалось завантажити котирування:", e)
        return None

def make_features_from_prices(prices: pd.DataFrame):
    """Перетворення цін у матрицю ознак (логарифмічні доходності)."""
    rets = np.log(prices / prices.shift(1)).dropna(how="any")
    X = rets.values.T
    X = StandardScaler().fit_transform(X)
    return X, rets.index

def offline_synthetic(tickers, n_points=250, random_state=42):
    """Генерація синтетичних даних для офлайн‑режиму."""
    rng = np.random.default_rng(random_state)
    n = len(tickers)
    k_hidden = 3
    factors = rng.normal(0, 0.01, size=(k_hidden, n_points))
    loadings = rng.normal(0, 1, size=(n, k_hidden))
    eps = rng.normal(0, 0.005, size=(n, n_points))
    X = loadings @ factors + eps
    X = StandardScaler().fit_transform(X)
    dates = pd.date_range("2024-01-01", periods=n_points, freq="B")
    return X, dates

def run_affinity_propagation(X):
    """Запуск Affinity Propagation та повернення міток і екземплярів."""
    model = AffinityPropagation(random_state=1, damping=0.8)
    labels = model.fit_predict(X)
    exemplars = model.cluster_centers_indices_
    return labels, exemplars

def plot_clusters(X, labels, names, exemplars, out_path="affinity_clusters.png"):
    """Візуалізація кластерів у просторі перших двох головних компонент."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=80, alpha=0.9, edgecolors="k")
    if exemplars is not None:
        plt.scatter(Z[exemplars, 0], Z[exemplars, 1], c="black", s=220, marker="X", label="Exemplar")
    for i, nm in enumerate(names):
        plt.text(Z[i, 0] + 0.02, Z[i, 1] + 0.02, nm.split()[0], fontsize=9)
    plt.title("AffinityPropagation — підгрупи компаній")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Кластери", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Збережено графік: {out_path}")
    plt.show()

def print_cluster_members(labels, tickers, names, exemplars):
    """Вивід складу кластерів у консоль."""
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append((tickers[i], names[i]))
    print("\n=== Склад кластерів ===")
    for lab in sorted(clusters):
        star = " *" if exemplars is not None and lab in exemplars else ""
        print(f"\nКластер {lab}{star}:")
        for t, n in clusters[lab]:
            print(f"  {t:6s}  —  {n}")

def main():
    ap = argparse.ArgumentParser(description="AffinityPropagation для кластеризації компаній")
    ap.add_argument("--json", type=str, required=True, help="шлях до company_symbol_mapping.json")
    ap.add_argument("--start", type=str, default="2024-01-01")
    ap.add_argument("--end", type=str, default="2024-12-31")
    ap.add_argument("--interval", type=str, default="1d", help="1d, 1wk, 1mo ...")
    ap.add_argument("--offline", action="store_true", help="не завантажувати дані, згенерувати синтетичні")
    args = ap.parse_args()

    tickers, names = load_mapping(Path(args.json))

    if args.offline:
        X, dates = offline_synthetic(tickers)
        print("[INFO] Працюємо в офлайн‑режимі (синтетичні дані).")
    else:
        prices = try_download_prices(tickers, args.start, args.end, args.interval)
        if prices is None or prices.empty:
            print("[WARN] Перехід в офлайн‑режим (синтетичні дані).")
            X, dates = offline_synthetic(tickers)
        else:
            prices = prices.reindex(columns=[c for c in tickers if c in prices.columns])
            if prices.shape[1] != len(tickers):
                print("[WARN] Не всі тикери знайдено:", [t for t in tickers if t not in prices.columns])
            X, dates = make_features_from_prices(prices)
            print(f"[INFO] Котирування: {prices.shape[0]} днів, тикерів: {prices.shape[1]}")

    labels, exemplars = run_affinity_propagation(X)
    print_cluster_members(labels, tickers, names, exemplars)
    plot_clusters(X, labels, names, exemplars)

if __name__ == "__main__":
    main()