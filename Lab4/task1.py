import numpy as np
import matplotlib.pyplot as plt

x = np.array([3, 1, 0, 5], dtype=float)
y = np.array([7, 4, 6, 8], dtype=float)

x_mean, y_mean = x.mean(), y.mean()
b1 = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
b0 = y_mean - b1*x_mean

X = np.column_stack([np.ones_like(x), x])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"Аналітично: b0={b0:.6f}, b1={b1:.6f}")
print(f"NumPy lstsq: b0={beta[0]:.6f}, b1={beta[1]:.6f}")

xx = np.linspace(x.min(), x.max(), 300)
yy = b0 + b1*xx
plt.scatter(x, y, label="Експериментальні точки")
plt.plot(xx, yy, label="Лінія МНК")
plt.title("Завдання 1. Перевірка методу найменших квадратів")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()