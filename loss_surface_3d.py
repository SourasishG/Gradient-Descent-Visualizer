"""
================================================================================
3D LOSS SURFACE VISUALIZATION OF GRADIENT DESCENT
================================================================================
Visualizes the 3D 'bowl' (paraboloid) of the loss function L(w, b) and traces
the 3D trajectory of the model parameters descending into the minimum.
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    os.makedirs("plots", exist_ok=True)
    print("Generating 3D Loss Surface Visualization...")

    # Synthetic data
    np.random.seed(42)
    num_samples = 100
    X = 2 * np.random.rand(num_samples, 1) - 1
    y = 3.0 * X + 2.0 + np.random.randn(num_samples, 1) * 0.2

    # Grid of w and b
    w_vals = np.linspace(-2.5, 4.5, 80)
    b_vals = np.linspace(-1.5, 3.5, 80)
    W, B = np.meshgrid(w_vals, b_vals)

    # Compute loss over 3D grid
    X_flat = X.ravel()
    y_flat = y.ravel()
    preds = W[:, :, None] * X_flat + B[:, :, None]
    Z = np.mean((preds - y_flat) ** 2, axis=2) / 2.0

    # Run Gradient Descent and record 3D path: (w, b, loss)
    w = -2.0
    b = -1.0
    lr = 0.25
    path_w = [w]
    path_b = [b]

    def get_loss(w_val, b_val):
        error = (w_val * X + b_val) - y
        return float(np.mean(error ** 2) / 2.0)

    path_z = [get_loss(w, b)]

    for _ in range(30):
        error = (w * X + b) - y
        dw = float(np.mean(error * X))
        db = float(np.mean(error))
        w = w - lr * dw
        b = b - lr * db
        path_w.append(w)
        path_b.append(b)
        path_z.append(get_loss(w, b))

    # Create 3D Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(
        W, B, Z,
        cmap="coolwarm",
        alpha=0.65,
        edgecolor="none"
    )

    # Plot optimization path
    ax.plot(
        path_w, path_b, path_z,
        color="black",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="GD Path (Descending)"
    )

    # Highlight Start and Minimum
    ax.scatter([path_w[0]], [path_b[0]], [path_z[0]], color="red", s=80, label="Start Point")
    ax.scatter([3.0], [2.0], [get_loss(3.0, 2.0)], color="gold", s=120, edgecolors="black", marker="*", label="Minimum (w=3, b=2)")

    ax.set_title("3D Loss Surface and Gradient Descent Trajectory", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Parameter w (Slope)", fontsize=10, labelpad=8)
    ax.set_ylabel("Parameter b (Intercept)", fontsize=10, labelpad=8)
    ax.set_zlabel("Loss L(w, b)", fontsize=10, labelpad=8)
    ax.view_init(elev=32, azim=225)
    ax.legend(loc="upper left")

    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label="Loss Value")

    plt.tight_layout()
    plt.savefig("plots/04_loss_surface_3d.png", dpi=200)
    plt.close()
    print("Saved: plots/04_loss_surface_3d.png")


if __name__ == "__main__":
    main()

