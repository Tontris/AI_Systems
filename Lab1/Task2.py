import numpy as np
import matplotlib.pyplot as plt

def activation(z: float) -> int:
    return int(z > 0)


def xor_model(a: int, b: int):
    neuron1 = activation(a - b - 0.5)
    neuron2 = activation(b - a - 0.5)
    output = activation(neuron1 + neuron2 - 0.5)
    return neuron1, neuron2, output

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

grid_x = np.linspace(0, 1, 201)
grid_y = np.linspace(0, 1, 201)
X, Y = np.meshgrid(grid_x, grid_y)
X_bin = (X > 0.5).astype(int)
Y_bin = (Y > 0.5).astype(int)
xor_map = ((X_bin | Y_bin) & ~(X_bin & Y_bin)).astype(int)

plt.figure(figsize=(7, 7), dpi=130)
plt.title("Карта рішень XOR", fontsize=15)
plt.pcolormesh(X, Y, xor_map, shading="nearest", cmap="coolwarm", alpha=0.8)
plt.scatter(inputs[:, 0], inputs[:, 1], c="black", s=80)
for x, y in inputs:
    label = int((x or y) and not (x and y))
    plt.text(x + 0.04, y + 0.04, f"y={label}", fontsize=11, weight="bold")
plt.xlabel("x₁", fontsize=13)
plt.ylabel("x₂", fontsize=13)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

x_vals = np.linspace(-0.1, 1.1, 200)
plt.figure(figsize=(9, 9), dpi=150)
plt.title("Вхідний простір XOR-перцептрона", fontsize=17)
plt.plot(x_vals, x_vals - 0.5, label="x₂ = x₁ - 0.5", linewidth=2)
plt.plot(x_vals, x_vals + 0.5, label="x₂ = x₁ + 0.5", linewidth=2)

for x, y in inputs:
    h1, h2, y_out = xor_model(x, y)
    plt.scatter(x, y, c="black", s=100)
    plt.text(x + 0.035, y + 0.035, f"h₁={h1}, h₂={h2}, y={y_out}",
    fontsize=13, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

plt.xlabel("x₁", fontsize=15)
plt.ylabel("x₂", fontsize=15)
plt.legend(fontsize=13)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

hidden_outputs = np.array([xor_model(x, y) for x, y in inputs])
hidden_layer = hidden_outputs[:, :2]
final_output = hidden_outputs[:, 2]

plt.figure(figsize=(7, 7), dpi=130)
plt.title("Простір прихованого шару", fontsize=15)
colors = ["blue" if y == 0 else "red" for y in final_output]
plt.scatter(hidden_layer[:, 0], hidden_layer[:, 1], c=colors, s=110, edgecolors="black")
for (h1, h2), y in zip(hidden_layer, final_output):
    plt.text(h1 + 0.03, h2 + 0.03, f"y={y}", fontsize=11,
    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

plt.plot(x_vals, 0.5 - x_vals, label="h₁ + h₂ = 0.5", linestyle="--", linewidth=2)
plt.xlabel("h₁", fontsize=13)
plt.ylabel("h₂", fontsize=13)
plt.legend(fontsize=12)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()