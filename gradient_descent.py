import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Data Setup (True parameters: w = 3.0, b = 2.0) ---
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1) - 1
y = 3.0 * X + 2.0 + np.random.randn(m, 1) * 0.2

# --- Tasks 1 & 2: Loss Function (MSE) and Gradient ---
def compute_loss(X_data, y_data, w, b):
    error = (w * X_data + b) - y_data
    return float(np.mean(error ** 2) / 2.0)

def compute_gradients(X_batch, y_batch, w, b):
    error = (w * X_batch + b) - y_batch
    dw = float(np.mean(error * X_batch))
    db = float(np.mean(error))
    return dw, db

# --- Tasks 3, 5, 7, 8, 9: Unified Gradient Descent ---
# batch_size = m  -> Batch GD (Task 7)
# batch_size = 1  -> Stochastic GD (Task 8)
# batch_size = 16 -> Mini-Batch GD (Task 9)
def train_gd(lr, epochs, batch_size, w_init, b_init):
    w = float(w_init)
    b = float(b_init)
    losses = [compute_loss(X, y, w, b)]
    path = [(w, b)]

    for _ in range(epochs):
        indices = np.random.permutation(m)
        for start in range(0, m, batch_size):
            idx = indices[start:start + batch_size]
            dw, db = compute_gradients(X[idx], y[idx], w, b)
            w = w - lr * dw  # Task 5: update rule
            b = b - lr * db
            path.append((w, b))
        losses.append(compute_loss(X, y, w, b))

    return w, b, losses, path

# --- Task 4: Random Parameter Initialization ---
w_init = float(np.random.randn())
b_init = float(np.random.randn())
w_start, b_start = -1.5, -0.5  # Fixed start for fair visual comparison

# --- Task 6: Experiment with Different Learning Rates (Batch GD) ---
lr_results = {}
for lr in [0.03, 0.30, 1.85]:
    _, _, loss_hist, _ = train_gd(lr=lr, epochs=25, batch_size=m, w_init=w_start, b_init=b_start)
    lr_results[lr] = loss_hist

# --- Tasks 7, 8, 9: Run Batch GD, SGD, and Mini-Batch GD ---
_, _, loss_bgd, path_bgd = train_gd(lr=0.25, epochs=40, batch_size=m, w_init=w_start, b_init=b_start)
_, _, loss_sgd, path_sgd = train_gd(lr=0.04, epochs=40, batch_size=1, w_init=w_start, b_init=b_start)
_, _, loss_mb,  path_mb  = train_gd(lr=0.10, epochs=40, batch_size=16, w_init=w_start, b_init=b_start)

# --- Task 10: Plot Loss vs Iterations & Parameter Trajectories ---
os.makedirs("plots", exist_ok=True)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

# 1. Learning Rates Comparison
for lr, losses in lr_results.items():
    ax1.plot(losses, label=f"lr = {lr}")
ax1.set_title("1. Learning Rate Effects (Batch GD)", fontweight="bold")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

# 2. Optimization Methods Loss Comparison
ax2.plot(loss_bgd, label="Batch GD (batch=100)")
ax2.plot(loss_sgd, label="SGD (batch=1)", linestyle="--")
ax2.plot(loss_mb,  label="Mini-Batch (batch=16)")
ax2.set_title("2. Optimization Methods (Loss vs Epochs)", fontweight="bold")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss (log scale)")
ax2.set_yscale("log")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.6)

# 3. 2D Contour Optimization Paths
W_grid, B_grid = np.meshgrid(np.linspace(-2.5, 4.5, 80), np.linspace(-1.5, 3.5, 80))
Z = np.mean((W_grid[:, :, None] * X.ravel() + B_grid[:, :, None] - y.ravel()) ** 2, axis=2) / 2.0
ax3.contour(W_grid, B_grid, Z, levels=20, cmap="viridis")

p_bgd, p_sgd, p_mb = np.array(path_bgd), np.array(path_sgd), np.array(path_mb)
ax3.plot(p_bgd[:, 0], p_bgd[:, 1], "b-o", markersize=3, label="Batch GD")
ax3.plot(p_sgd[::5, 0], p_sgd[::5, 1], "r.-", alpha=0.5, label="SGD (Noisy)")
ax3.plot(p_mb[::2, 0], p_mb[::2, 1], "g-x", markersize=3, label="Mini-Batch")
ax3.plot(w_start, b_start, "ko", markersize=6, label="Start")
ax3.plot(3.0, 2.0, "y*", markersize=14, markeredgecolor="black", label="Optimum (3, 2)")
ax3.set_title("3. Optimization Paths on Loss Surface", fontweight="bold")
ax3.set_xlabel("w (slope)")
ax3.set_ylabel("b (intercept)")
ax3.legend(fontsize=8)
ax3.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("plots/gd_comparison.png", dpi=180)
plt.close()
print("All 10 tasks completed in ~75 lines! Plot saved to plots/gd_comparison.png")
