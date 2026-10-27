# Gradient Descent From Scratch: Complete Visual Guide & Implementation

A beginner-friendly, comprehensive guide and implementation of **Gradient Descent from scratch** using pure Python and NumPy. This repository covers all 10 foundational tasks of mathematical optimization, parameter dynamics, learning rates, and gradient descent variants (Batch, Stochastic, and Mini-Batch).

---

## 📖 Table of Contents
1. [The Problem: Why Optimization Matters](#-the-problem-why-optimization-matters)
2. [Mathematical Foundations](#-mathematical-foundations)
3. [Breakdown of All 10 Tasks](#-breakdown-of-all-10-tasks)
4. [Optimization Methods: Batch vs. SGD vs. Mini-Batch](#-optimization-methods-batch-vs-sgd-vs-mini-batch)
5. [The Role of Learning Rate](#-the-role-of-learning-rate)
6. [Code Architecture & Unified Design](#-code-architecture--unified-design)
7. [Repository Files & How to Run](#-repository-files--how-to-run)

---

## 🎯 The Problem: Why Optimization Matters

In supervised machine learning, we model relationships between inputs and outputs. For a linear regression model:
$$\hat{y}^{(i)} = w \cdot x^{(i)} + b$$

At the beginning, the parameters $w$ (weight / slope) and $b$ (bias / intercept) are unknown. If initialized randomly, the predictions will be completely inaccurate.

To make the model learn, we need:
1. **A Loss Function ($L$)**: A mathematical function that quantifies the difference between predictions $\hat{y}$ and true values $y$.
2. **An Optimization Algorithm**: A systematic procedure to update $w$ and $b$ to minimize $L$.

### Why not solve it analytically?
For simple linear regression, a closed-form formula exists (the *Normal Equation*: $\theta = (X^T X)^{-1} X^T y$). However:
- Matrix inversion scales as $O(d^3)$ with the number of features $d$. For large datasets, computing $(X^T X)^{-1}$ is computationally prohibitive.
- For non-linear models, logistic regression, and deep neural networks, analytical solutions do not exist.

**Gradient Descent** solves this by iteratively taking small downhill steps toward the minimum of the loss surface.

---

## 📐 Mathematical Foundations

### 1. The Loss Function (Mean Squared Error)
We use the Mean Squared Error (MSE) loss function:
$$L(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2 = \frac{1}{2m} \sum_{i=1}^{m} \left( w \cdot x^{(i)} + b - y^{(i)} \right)^2$$

- $m$: Number of training samples.
- The factor $\frac{1}{2}$ is intentionally added so that when differentiating, the exponent $2$ cancels out cleanly.

### 2. Analytical Gradient Derivation
The gradient vector $\nabla L = \left[ \frac{\partial L}{\partial w}, \frac{\partial L}{\partial b} \right]^T$ points in the direction of steepest increase. Stepping in the negative direction $-\nabla L$ guarantees moving toward the minimum.

Using the multivariable chain rule:
$$\frac{\partial L}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) \cdot x^{(i)}$$

$$\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)$$

### 3. The Parameter Update Rule
At each iteration step $t$:
$$w \leftarrow w - \alpha \cdot \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}$$

Where $\alpha > 0$ is the **learning rate** (step size).

---

## 🔍 Breakdown of All 10 Tasks

| Task # | Task Name | Description & Mathematical Role |
|---|---|---|
| **Task 1** | **Define Loss Function** | Formulate Mean Squared Error (MSE) to quantify model prediction error. |
| **Task 2** | **Calculate Gradient** | Derive partial derivatives $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$ analytically. |
| **Task 3** | **Implement from Scratch** | Build the optimization loop in pure Python/NumPy without ML libraries. |
| **Task 4** | **Random Parameter Initialization** | Initialize $w$ and $b$ using `np.random.randn()`. |
| **Task 5** | **Update Parameters** | Shift parameters opposite to the gradient scaled by learning rate $\alpha$. |
| **Task 6** | **Learning Rate Experiments** | Compare Small ($\alpha=0.03$), Optimal ($\alpha=0.30$), and Large ($\alpha=1.85$). |
| **Task 7** | **Batch Gradient Descent (BGD)** | Update parameters using the entire dataset ($m=100$) per step. |
| **Task 8** | **Stochastic Gradient Descent (SGD)**| Update parameters after every single training example ($k=1$). |
| **Task 9** | **Mini-Batch Gradient Descent** | Update parameters over small balanced batches ($k=16$). |
| **Task 10**| **Visualizations & Comparison** | Plot loss vs. epochs, 2D contour trajectories, and 3D loss surface bowl. |

---

## ⚖️ Optimization Methods: Batch vs. SGD vs. Mini-Batch

```
BGD (Batch Size = m):          ●───────────●───────────●───────────★ (Smooth, direct line)
SGD (Batch Size = 1):          ●───╲───╱───╲───╱───╲───╱───────────★ (Noisy, wandering walk)
Mini-Batch (Batch Size = 16):  ●─────╲─────●─────╱─────●───────────★ (Smooth with slight noise)
```

| Dimension | Batch Gradient Descent | Stochastic Gradient Descent (SGD) | Mini-Batch Gradient Descent |
|---|---|---|---|
| **Batch Size ($k$)** | Entire dataset ($k = m$) | Single sample ($k = 1$) | Small batch ($1 < k < m$, e.g. 16) |
| **Trajectory** | Direct, smooth line down steepest slope | Erratic, random zigzagging | Smooth curve with mild stochasticity |
| **Speed per Step** | Slowest (processes all data before update) | Fastest (updates after 1 sample) | Fast (vectorized matrix chunks) |
| **Hardware Efficiency** | Moderate | Poor (cannot leverage GPU/SIMD vectorization) | Optimal (fully utilizes parallel hardware) |
| **Saddle Point Escape** | Can get trapped in saddle points/plateaus | Fluctuations help shake free from plateaus | Mild fluctuations help escape bad plateaus |
| **Standard Use Case** | Small datasets | Online / streaming data | Standard default in Deep Learning |

---

## ⚡ The Role of Learning Rate

The learning rate $\alpha$ dictates the step magnitude taken in parameter space:
1. **Small ($\alpha = 0.03$)**: Steps are tiny. The trajectory hugs the contours and never overshoots, but requires an excessive number of iterations to converge.
2. **Optimal ($\alpha = 0.30$)**: Balances progress and stability, reaching the minimum smoothly in minimal epochs.
3. **Large ($\alpha = 1.85$)**: The step size exceeds the curvature radius of the valley. The parameters overshoot the bottom, bouncing back and forth across opposing valley walls.
4. **Divergent ($\alpha > 2.0$)**: The step size compounds exponentially ($|w^{(t+1)} - w^*| > |w^{(t)} - w^*|$), causing the loss to explode toward infinity (`NaN`).

Mathematically, for a quadratic surface with maximum curvature eigenvalue $\lambda_{\max}$, stability strictly requires:
$$0 < \alpha < \frac{2}{\lambda_{\max}}$$

---

## 🧩 Code Architecture & Unified Design

A core insight of this repository is that **Batch GD**, **SGD**, and **Mini-Batch GD** are not separate algorithms. They are the exact same gradient descent update rule executed over different `batch_size` values.

This allows us to write all three variants in a single, beginner-friendly 15-line function:

```python
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
            w = w - lr * dw
            b = b - lr * db
            path.append((w, b))
        losses.append(compute_loss(X, y, w, b))

    return w, b, losses, path
```

- When `batch_size = m` $\rightarrow$ **Batch Gradient Descent**
- When `batch_size = 1` $\rightarrow$ **Stochastic Gradient Descent (SGD)**
- When `batch_size = 16` $\rightarrow$ **Mini-Batch Gradient Descent**

---

## 📁 Repository Files & How to Run

### 1. Main Visualizer (`gradient_descent.py`)
Executes all 10 tasks and saves a 3-panel comparison figure (`plots/gd_comparison.png`).
```powershell
python gradient_descent.py
```
- **Panel 1**: Learning Rate Effects (Small vs. Optimal vs. Large).
- **Panel 2**: Optimization Methods Loss vs. Epochs (Logarithmic scale).
- **Panel 3**: Parameter Trajectories on 2D Contour Surface.

### 2. 3D Loss Surface (`loss_surface_3d.py`)
Generates the 3D paraboloid bowl with the parameter descent path (`plots/04_loss_surface_3d.png`).
```powershell
python loss_surface_3d.py
```

### 3. Minimal 1D Demo (`quadratic_1d_demo.py`)
Minimal 25-line script demonstrating gradient descent on $f(w) = (w - 3)^2 + 2$ (`plots/00_1d_quadratic_demo.png`).
```powershell
python quadratic_1d_demo.py
```

---

## 📌 Summary

- **Loss Function**: Quantifies prediction errors as a continuous surface.
- **Gradient**: Provides the exact downhill compass direction.
- **Learning Rate**: Controls stride length (must be tuned to avoid divergence).
- **Mini-Batch GD**: The optimal balance of convergence stability and parallel hardware speed.

### 3. Minimal 1D Demo (`quadratic_1d_demo.py`)
Minimal 25-line script demonstrating gradient descent on $f(w) = (w - 3)^2 + 2$ (`plots/00_1d_quadratic_demo.png`).
```powershell
python quadratic_1d_demo.py
```

---

## 📌 Summary

- **Loss Function**: Quantifies prediction errors as a continuous surface.
- **Gradient**: Provides the exact downhill compass direction.
- **Learning Rate**: Controls stride length (must be tuned to avoid divergence).
- **Mini-Batch GD**: The optimal balance of convergence stability and parallel hardware speed.

