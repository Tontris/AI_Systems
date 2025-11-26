import numpy as np
import matplotlib.pyplot as plt

x = np.array([8, 7, 6, 5, 4, 3], dtype=float)
y = np.array([2, 4, 6, 8, 10, 12], dtype=float)

b1, b0 = np.polyfit(x, y, 1)
y_hat = b1 * x + b0

def regression_metrics(y_true, y_pred):
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    mse = np.mean((y_true - y_pred)**2)
    return r2, mse

r2, mse = regression_metrics(y, y_hat)

print(f"Рівняння регресії: y = {b0:.6f} + {b1:.6f}x")
print(f"R² = {r2:.6f},  MSE = {mse:.6f}")

xx = np.linspace(x.min(), x.max(), 400)
yy = b1 * xx + b0

plt.scatter(x, y, color="blue", label="Експериментальні дані")
plt.plot(xx, yy, color="red", label="Лінія регресії")
plt.title("Завдання 2. Лінійна регресія (Варіант 2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()