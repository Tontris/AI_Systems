import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

def load_data(path="Lab7/data_clustering.txt"):
    """Завантаження даних з файлу."""
    data = np.loadtxt(path, delimiter=",")
    print(f"Розмірність даних: {data.shape}")
    return data

def run_meanshift(data, quantile=0.2, n_samples=100):
    """Запуск алгоритму MeanShift та повернення результатів."""
    bandwidth = estimate_bandwidth(data, quantile=quantile, n_samples=n_samples)
    print(f"Оцінена ширина вікна (bandwidth): {bandwidth:.3f}")

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)

    labels = ms.labels_
    centers = ms.cluster_centers_
    n_clusters = len(np.unique(labels))

    return labels, centers, n_clusters

def plot_clusters(data, labels, centers, n_clusters):
    """Візуалізація кластерів та їх центрів."""
    plt.figure(figsize=(7, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', s=35, alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='Центри кластерів')
    plt.title(f"Кластеризація методом MeanShift (кластерів = {n_clusters})")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()

def main():
    data = load_data("Lab7/data_clustering.txt")
    labels, centers, n_clusters = run_meanshift(data, quantile=0.2, n_samples=100)
    plot_clusters(data, labels, centers, n_clusters)

    print(f"Кількість знайдених кластерів: {n_clusters}")
    print("Координати центрів кластерів:")
    for i, center in enumerate(centers, start=1):
        print(f"Кластер {i}: {center}")

if __name__ == "__main__":
    main()