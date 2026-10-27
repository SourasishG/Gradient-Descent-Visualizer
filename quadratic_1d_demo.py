import numpy as np
import matplotlib.pyplot as plt

# Objective: Minimize f(w) = (w - 3)^2 + 2,  Derivative: f'(w) = 2 * (w - 3)
w_curve = np.linspace(-2, 8, 100)
plt.figure(figsize=(8, 4.5))
plt.plot(w_curve, (w_curve - 3.0) ** 2 + 2.0, "k-", alpha=0.4, label="f(w) = (w-3)^2 + 2")

# Run GD for small, optimal, and large learning rates
configs = [(0.1, "blue", "Small (0.1)"), (0.7, "green", "Optimal (0.7)"), (1.05, "red", "Large (1.05)")]
for lr, col, label in configs:
    w = -1.5
    path = [w]
    for _ in range(12):
        w = w - lr * 2.0 * (w - 3.0)  # update: w = w - lr * grad
        path.append(w)
    loss = [(val - 3.0) ** 2 + 2.0 for val in path]
    plt.plot(path, loss, "o-", color=col, label=label)

plt.plot(3.0, 2.0, "y*", markersize=14, markeredgecolor="black", label="Optimum (3, 2)")
plt.title("1D Gradient Descent on Quadratic Function", fontweight="bold")
plt.xlabel("w")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plots/00_1d_quadratic_demo.png", dpi=150)
plt.close()
print("Saved 1D demo in ~25 lines to plots/00_1d_quadratic_demo.png")
