import numpy as np
import matplotlib.pyplot as plt

# 1. Data Setup (True: w=3, b=2)
np.random.seed(42)
X = 2 * np.random.rand(100, 1) - 1
y = 3.0 * X + 2.0 + np.random.randn(100, 1) * 0.2

# 2. 3D Grid & Loss Surface (Paraboloid Bowl)
W, B = np.meshgrid(np.linspace(-2.5, 4.5, 50), np.linspace(-1.5, 3.5, 50))
Z = np.mean((W[:, :, None] * X.ravel() + B[:, :, None] - y.ravel()) ** 2, axis=2) / 2.0

# 3. Track Gradient Descent 3D Path: (w, b, loss)
w, b, lr = -2.0, -1.0, 0.25
path = [(w, b, float(np.mean(((w * X + b) - y) ** 2) / 2.0))]

for _ in range(25):
    err = (w * X + b) - y
    w = w - lr * float(np.mean(err * X))
    b = b - lr * float(np.mean(err))
    path.append((w, b, float(np.mean(((w * X + b) - y) ** 2) / 2.0)))

# 4. Plot 3D Surface and Trajectory
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(W, B, Z, cmap="coolwarm", alpha=0.6)

p = np.array(path)
ax.plot(p[:, 0], p[:, 1], p[:, 2], "k-o", markersize=4, label="GD Path")
ax.scatter([3.0], [2.0], [np.min(Z)], color="gold", s=100, label="Optimum (3, 2)")

ax.set_title("3D Loss Surface & Gradient Descent Path", fontweight="bold")
ax.set_xlabel("w (slope)")
ax.set_ylabel("b (intercept)")
ax.set_zlabel("Loss")
ax.legend()

plt.tight_layout()
plt.savefig("plots/04_loss_surface_3d.png", dpi=150)
plt.close()
print("Saved 3D loss surface in ~35 lines to plots/04_loss_surface_3d.png")
