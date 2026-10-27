"""
================================================================================
1D GRADIENT DESCENT DEMO (The Absolute Simplest Intro)
================================================================================
Objective: Find the minimum of f(w) = (w - 3)^2 + 2
Derivative: f'(w) = 2 * (w - 3)
Minimum is at: w = 3, where f(3) = 2
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def loss_function(w):
    return (w - 3.0) ** 2 + 2.0


def gradient(w):
    return 2.0 * (w - 3.0)


def run_1d_gd(learning_rate, num_steps, start_w):
    w = float(start_w)
    path = [w]
    losses = [loss_function(w)]

    for _ in range(num_steps):
        grad = gradient(w)
        w = w - learning_rate * grad
        path.append(w)
        losses.append(loss_function(w))

    return path, losses


def main():
    start_w = -2.0  # Initial random / chosen parameter
    steps = 15

    # Test three learning rates
    rates = [0.1, 0.7, 1.05]
    labels = ["Small LR (0.1) - Steady", "Optimal LR (0.7) - Fast", "Too Large LR (1.05) - Oscillating"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    plt.figure(figsize=(10, 5))

    # Plot the 1D parabolic curve
    w_curve = np.linspace(-3, 8, 200)
    plt.plot(w_curve, loss_function(w_curve), "k-", alpha=0.5, label="Loss Curve f(w)=(w-3)^2+2")

    for lr, label, col in zip(rates, labels, colors):
        path, _ = run_1d_gd(lr, steps, start_w)
        path_loss = [loss_function(val) for val in path]
        plt.plot(path, path_loss, marker="o", label=f"{label}", color=col, alpha=0.8)

    plt.plot(3.0, 2.0, "y*", markersize=16, markeredgecolor="black", label="Minimum (w=3, loss=2)")
    plt.title("1D Gradient Descent: Parameter Steps on Loss Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Parameter w", fontsize=11)
    plt.ylabel("Loss f(w)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/00_1d_quadratic_demo.png", dpi=200)
    plt.close()
    print("Saved: plots/00_1d_quadratic_demo.png")


if __name__ == "__main__":
    main()

