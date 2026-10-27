"""
================================================================================
GRADIENT DESCENT FROM SCRATCH & VISUALIZATION
================================================================================
Tasks Covered:
  1. Define a simple mathematical loss function (Mean Squared Error).
  2. Calculate its gradient analytically.
  3. Implement Gradient Descent from scratch using Python and NumPy.
  4. Initialize model parameters randomly.
  5. Update parameters using the gradient.
  6. Experiment with different learning rates (Small, Medium, Large).
  7. Implement Batch Gradient Descent (BGD).
  8. Implement Stochastic Gradient Descent (SGD).
  9. Implement Mini-Batch Gradient Descent (MBGD).
 10. Plot loss vs iterations and compare optimization methods.
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# DATASET GENERATION (Beginner-Friendly Synthetic Linear Data)
# ==============================================================================
# Model equation: y = w_true * x + b_true + noise
# True parameters: slope (w) = 3.0, intercept (b) = 2.0
np.random.seed(42)
num_samples = 100
X = 2 * np.random.rand(num_samples, 1) - 1  # X values between -1 and +1
noise = np.random.randn(num_samples, 1) * 0.2
y = 3.0 * X + 2.0 + noise


# ==============================================================================
# TASK 1: Define a Simple Mathematical Loss Function
# ==============================================================================
# We use Mean Squared Error (MSE):
#   Loss = (1 / (2 * m)) * sum((y_pred - y_true) ** 2)
# The factor of 1/2 simplifies the derivative when differentiating.
def compute_loss(X_data, y_data, w, b):
    m = len(y_data)
    y_pred = w * X_data + b
    error = y_pred - y_data
    loss = (1.0 / (2.0 * m)) * np.sum(error ** 2)
    return float(loss)


# ==============================================================================
# TASK 2: Calculate Its Gradient
# ==============================================================================
# The gradient tells us the direction of steepest ascent.
# By taking the negative of the gradient, we step toward the minimum.
#   d(Loss)/dw = (1 / m) * sum((y_pred - y_true) * X)
#   d(Loss)/db = (1 / m) * sum(y_pred - y_true)
def compute_gradients(X_batch, y_batch, w, b):
    batch_size = len(y_batch)
    y_pred = w * X_batch + b
    error = y_pred - y_batch
    dw = (1.0 / batch_size) * np.sum(error * X_batch)
    db = (1.0 / batch_size) * np.sum(error)
    return dw, db


# ==============================================================================
# TASK 3, 4, 5: Gradient Descent from Scratch with Random Init & Parameter Update
# ==============================================================================
def run_gradient_descent(X_data, y_data, learning_rate, num_iterations, w_start=None, b_start=None):
    # TASK 4: Initialize model parameters randomly if not specified
    if w_start is None:
        w = float(np.random.randn())
    else:
        w = float(w_start)

    if b_start is None:
        b = float(np.random.randn())
    else:
        b = float(b_start)

    loss_history = []
    param_history = [(w, b)]

    # Compute starting loss
    initial_loss = compute_loss(X_data, y_data, w, b)
    loss_history.append(initial_loss)

    for step in range(num_iterations):
        # Calculate gradients
        dw, db = compute_gradients(X_data, y_data, w, b)

        # TASK 5: Update parameters using the gradient
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record parameter path and loss
        param_history.append((w, b))
        current_loss = compute_loss(X_data, y_data, w, b)
        loss_history.append(current_loss)

    return w, b, loss_history, param_history


# ==============================================================================
# TASK 7: Batch Gradient Descent (BGD)
# Uses the ENTIRE dataset to compute the gradient at every single step.
# ==============================================================================
def batch_gradient_descent(X_data, y_data, learning_rate, num_epochs, w_init, b_init):
    w = float(w_init)
    b = float(b_init)
    loss_history = [compute_loss(X_data, y_data, w, b)]
    path = [(w, b)]

    for epoch in range(num_epochs):
        dw, db = compute_gradients(X_data, y_data, w, b)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        path.append((w, b))
        loss_history.append(compute_loss(X_data, y_data, w, b))

    return w, b, loss_history, path


# ==============================================================================
# TASK 8: Stochastic Gradient Descent (SGD)
# Updates parameters after EVERY SINGLE training sample (batch size = 1).
# Introduces stochastic fluctuations that help escape shallow local plateaus.
# ==============================================================================
def stochastic_gradient_descent(X_data, y_data, learning_rate, num_epochs, w_init, b_init):
    w = float(w_init)
    b = float(b_init)
    m = len(y_data)
    loss_history = [compute_loss(X_data, y_data, w, b)]
    path = [(w, b)]

    for epoch in range(num_epochs):
        # Shuffle data indices each epoch
        indices = np.random.permutation(m)
        for idx in indices:
            xi = X_data[idx:idx + 1]
            yi = y_data[idx:idx + 1]
            dw, db = compute_gradients(xi, yi, w, b)
            w = w - learning_rate * dw
            b = b - learning_rate * db
            path.append((w, b))

        loss_history.append(compute_loss(X_data, y_data, w, b))

    return w, b, loss_history, path


# ==============================================================================
# TASK 9: Mini-Batch Gradient Descent (MBGD)
# Divides data into small batches (e.g., 16 samples) and updates parameters per batch.
# Combines the stability of Batch GD with the computational speed of SGD.
# ==============================================================================
def mini_batch_gradient_descent(X_data, y_data, learning_rate, num_epochs, batch_size, w_init, b_init):
    w = float(w_init)
    b = float(b_init)
    m = len(y_data)
    loss_history = [compute_loss(X_data, y_data, w, b)]
    path = [(w, b)]

    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            batch_idx = indices[start:end]
            X_batch = X_data[batch_idx]
            y_batch = y_data[batch_idx]

            dw, db = compute_gradients(X_batch, y_batch, w, b)
            w = w - learning_rate * dw
            b = b - learning_rate * db
            path.append((w, b))

        loss_history.append(compute_loss(X_data, y_data, w, b))

    return w, b, loss_history, path


# ==============================================================================
# HELPER: Precompute Loss Grid for Contour & Surface Visualizations
# ==============================================================================
def make_loss_grid(X_data, y_data, w_range, b_range, num_points=120):
    w_grid, b_grid = np.meshgrid(w_range, b_range)
    X_flat = X_data.ravel()
    y_flat = y_data.ravel()
    # Vectorized computation of loss across the 2D grid
    preds = w_grid[:, :, None] * X_flat + b_grid[:, :, None]
    z_loss = np.mean((preds - y_flat) ** 2, axis=2) / 2.0
    return w_grid, b_grid, z_loss


# ==============================================================================
# TASK 6 & 10: Run Experiments and Generate High-Quality Visualizations
# ==============================================================================
def main():
    os.makedirs("plots", exist_ok=True)
    print("=" * 70)
    print("  GRADIENT DESCENT FROM SCRATCH - EXPERIMENTS & VISUALIZATION")
    print("=" * 70)

    # Common starting point for fair side-by-side comparison
    w_start = -1.5
    b_start = -0.5
    print(f"Dataset generated: {num_samples} points")
    print(f"Ground Truth Parameters: w = 3.0, b = 2.0")
    print(f"Starting Parameters:     w = {w_start}, b = {b_start}")
    print("-" * 70)

    # --------------------------------------------------------------------------
    # TASK 6: Experiment with Different Learning Rates
    # --------------------------------------------------------------------------
    print("\n[TASK 6] Running Learning Rate Experiments...")
    lr_configs = [
        {"rate": 0.03, "label": "Small LR (0.03) - Slow Progress", "color": "#1f77b4"},
        {"rate": 0.30, "label": "Optimal LR (0.30) - Smooth & Fast", "color": "#2ca02c"},
        {"rate": 1.85, "label": "Large LR (1.85) - Oscillations", "color": "#d62728"}
    ]

    lr_results = []
    iterations = 25
    for cfg in lr_configs:
        w_final, b_final, loss_hist, path = run_gradient_descent(
            X, y, learning_rate=cfg["rate"], num_iterations=iterations,
            w_start=w_start, b_start=b_start
        )
        lr_results.append({
            "rate": cfg["rate"],
            "label": cfg["label"],
            "color": cfg["color"],
            "loss_hist": loss_hist,
            "path": path,
            "final_w": w_final,
            "final_b": b_final
        })
        print(f"  Learning Rate {cfg['rate']:<5}: Final w = {w_final:6.3f}, b = {b_final:6.3f}, Loss = {loss_hist[-1]:.5f}")

    # --------------------------------------------------------------------------
    # Plot 1: Learning Rate Comparison (Loss Curves & Parameter Paths)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Loss vs Iterations
    for res in lr_results:
        ax1.plot(res["loss_hist"], marker="o", markersize=4, label=res["label"], color=res["color"], linewidth=2)
    ax1.set_title("Loss vs. Iterations for Different Learning Rates", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Iteration Step", fontsize=11)
    ax1.set_ylabel("Mean Squared Error Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Parameter Paths on Contour Map
    w_range = np.linspace(-2.5, 4.5, 120)
    b_range = np.linspace(-1.5, 3.5, 120)
    W_grid, B_grid, Z_loss = make_loss_grid(X, y, w_range, b_range)

    contour = ax2.contour(W_grid, B_grid, Z_loss, levels=25, cmap="viridis", alpha=0.7)
    ax2.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    for res in lr_results:
        p_arr = np.array(res["path"])
        ax2.plot(p_arr[:, 0], p_arr[:, 1], marker=".", markersize=5, label=res["label"], color=res["color"], linewidth=1.8)

    # Highlight Start and Minimum
    ax2.plot(w_start, b_start, "ro", markersize=9, label="Start (-1.5, -0.5)")
    ax2.plot(3.0, 2.0, "y*", markersize=14, markeredgecolor="black", label="True Optimum (3.0, 2.0)")
    ax2.set_title("Optimization Path on Loss Contours", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Parameter w (Slope)", fontsize=11)
    ax2.set_ylabel("Parameter b (Intercept)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig("plots/01_learning_rate_experiment.png", dpi=200)
    plt.close()
    print("  Saved: plots/01_learning_rate_experiment.png")

    # --------------------------------------------------------------------------
    # TASKS 7, 8, 9: Batch vs. Stochastic vs. Mini-Batch Gradient Descent
    # --------------------------------------------------------------------------
    print("\n[TASKS 7, 8, 9] Running BGD, SGD, and Mini-Batch GD...")
    epochs = 40
    np.random.seed(42)

    # Batch GD
    w_bgd, b_bgd, loss_bgd, path_bgd = batch_gradient_descent(
        X, y, learning_rate=0.25, num_epochs=epochs, w_init=w_start, b_init=b_start
    )
    print(f"  Batch GD:      Final w = {w_bgd:6.3f}, b = {b_bgd:6.3f}, Loss = {loss_bgd[-1]:.5f}")

    # Stochastic GD (SGD)
    w_sgd, b_sgd, loss_sgd, path_sgd = stochastic_gradient_descent(
        X, y, learning_rate=0.04, num_epochs=epochs, w_init=w_start, b_init=b_start
    )
    print(f"  Stochastic GD: Final w = {w_sgd:6.3f}, b = {b_sgd:6.3f}, Loss = {loss_sgd[-1]:.5f}")

    # Mini-Batch GD
    w_mb, b_mb, loss_mb, path_mb = mini_batch_gradient_descent(
        X, y, learning_rate=0.10, num_epochs=epochs, batch_size=16, w_init=w_start, b_init=b_start
    )
    print(f"  Mini-Batch GD: Final w = {w_mb:6.3f}, b = {b_mb:6.3f}, Loss = {loss_mb[-1]:.5f}")

    # --------------------------------------------------------------------------
    # TASK 10: Plot Loss vs Iterations & Compare the 3 Optimization Methods
    # --------------------------------------------------------------------------
    print("\n[TASK 10] Generating Method Comparison Plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Loss vs Epochs
    ax1.plot(loss_bgd, label="Batch GD (lr=0.25)", color="#1f77b4", linewidth=2.2)
    ax1.plot(loss_sgd, label="Stochastic GD (lr=0.04)", color="#ff7f0e", linewidth=1.8, linestyle="--")
    ax1.plot(loss_mb, label="Mini-Batch GD (lr=0.10, batch=16)", color="#2ca02c", linewidth=2)
    ax1.set_title("Loss vs. Epochs: Comparison of GD Methods", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Mean Squared Error Loss", fontsize=11)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Parameter Paths on Loss Surface Contours
    contour = ax2.contour(W_grid, B_grid, Z_loss, levels=25, cmap="cividis", alpha=0.7)
    ax2.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    # Plot paths
    p_bgd = np.array(path_bgd)
    p_sgd = np.array(path_sgd)
    p_mb = np.array(path_mb)

    # Subsample SGD path slightly for visual clarity while preserving fluctuation
    sgd_plot_indices = list(range(0, len(p_sgd), 5))
    sgd_plot_indices.append(len(p_sgd) - 1)
    p_sgd_sub = p_sgd[sgd_plot_indices]

    ax2.plot(p_bgd[:, 0], p_bgd[:, 1], "o-", color="#1f77b4", label="Batch GD Path", linewidth=2, markersize=4)
    ax2.plot(p_sgd_sub[:, 0], p_sgd_sub[:, 1], ".-", color="#ff7f0e", label="SGD Path (Noisy)", linewidth=1, markersize=3, alpha=0.85)
    ax2.plot(p_mb[::2, 0], p_mb[::2, 1], "x-", color="#2ca02c", label="Mini-Batch GD Path", linewidth=1.5, markersize=4)

    ax2.plot(w_start, b_start, "ro", markersize=9, label="Start (-1.5, -0.5)")
    ax2.plot(3.0, 2.0, "y*", markersize=15, markeredgecolor="black", label="True Optimum (3.0, 2.0)")

    ax2.set_title("Trajectory Comparison on Loss Contours", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Parameter w (Slope)", fontsize=11)
    ax2.set_ylabel("Parameter b (Intercept)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig("plots/02_optimization_methods_comparison.png", dpi=200)
    plt.close()
    print("  Saved: plots/02_optimization_methods_comparison.png")

    # --------------------------------------------------------------------------
    # Plot 3: Fitted Model Line on Data Points
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(X, y, color="#4c72b0", alpha=0.7, label="Training Data Points (with noise)")

    x_line = np.linspace(-1.1, 1.1, 100).reshape(-1, 1)
    y_init = w_start * x_line + b_start
    y_bgd = w_bgd * x_line + b_bgd
    y_true = 3.0 * x_line + 2.0

    ax.plot(x_line, y_init, "r--", linewidth=1.8, label=f"Initial Random Line (w={w_start:.1f}, b={b_start:.1f})")
    ax.plot(x_line, y_bgd, "g-", linewidth=2.4, label=f"Trained GD Line (w={w_bgd:.2f}, b={b_bgd:.2f})")
    ax.plot(x_line, y_true, "k:", linewidth=1.8, label="Ground Truth (w=3.0, b=2.0)")

    ax.set_title("Model Fit: Before vs. After Gradient Descent", fontsize=12, fontweight="bold")
    ax.set_xlabel("Input Feature x", fontsize=11)
    ax.set_ylabel("Target y", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("plots/03_model_fit.png", dpi=200)
    plt.close()
    print("  Saved: plots/03_model_fit.png")

    print("\n" + "=" * 70)
    print("ALL 10 TASKS SUCCESSFULLY COMPLETED! Check the 'plots' directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()

