import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data(path="Lab7/data_clustering.txt"):
    """Завантаження даних з файлу."""
    data = np.loadtxt(path, delimiter=",")
    print(f"Розмірність даних: {data.shape}")
    return data

def plot_raw_data(data):
    """Візуалізація вихідних даних."""
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], s=25, c='gray')
    plt.title("Вихідні дані для кластеризації (LR_7_task_1)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show()

def run_kmeans(data, n_clusters=5, random_state=42):
    """Запуск алгоритму K-means та повернення міток і центрів."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        random_state=random_state
    )
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def plot_clusters(data, labels, centers):
    """Візуалізація кластерів та центрів."""
    plt.figure(figsize=(7, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', s=30)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X')
    plt.title("Кластеризація методом K-середніх")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def main():
    data = load_data("Lab7/data_clustering.txt")
    plot_raw_data(data)

    labels, centers = run_kmeans(data, n_clusters=5, random_state=42)
    plot_clusters(data, labels, centers)

    print("Координати центрів кластерів:")
    for i, center in enumerate(centers, start=1):
        print(f"Кластер {i}: {center}")

if __name__ == "__main__":
    main()