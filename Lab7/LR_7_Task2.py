import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data
y = iris.target

n_clusters = 3
kmeans = KMeans(
    n_clusters=n_clusters,
    init='k-means++',
    n_init=20,
    random_state=42
)

labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Центри кластерів')
plt.title("Кластеризація набору Iris методом K-середніх")
plt.xlabel("Довжина чашолистка")
plt.ylabel("Ширина чашолистка")
plt.legend()
plt.show()

print("Координати центрів кластерів:")
for i, center in enumerate(centers, start=1):
    print(f"Кластер {i}: {center}")