# Gradient Descent From Scratch & Visualizer

A beginner-friendly, clean, and complete implementation of **Gradient Descent** from scratch using pure Python and NumPy (without machine learning libraries like scikit-learn or PyTorch).

---

## 🎯 Tasks Completed

1. **Simple Mathematical Loss Function**: Implemented Mean Squared Error (MSE) $L(w, b) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}_i - y_i)^2$.
2. **Gradient Calculation**: Derived and calculated the partial derivatives $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$.
3. **Scratch Implementation**: Pure Python and NumPy vectorization.
4. **Random Parameter Initialization**: Initialized weights and biases randomly (`np.random.randn()`).
5. **Parameter Update Rule**: Updated weights using $w \leftarrow w - \alpha \frac{\partial L}{\partial w}$ and $b \leftarrow b - \alpha \frac{\partial L}{\partial b}$.
6. **Learning Rate Experiments**: Tested Small ($\alpha = 0.03$), Optimal ($\alpha = 0.30$), and Large ($\alpha = 1.85$) learning rates.
7. **Batch Gradient Descent (BGD)**: Computes gradients using the entire dataset per update.
8. **Stochastic Gradient Descent (SGD)**: Updates parameters per individual training example.
9. **Mini-Batch Gradient Descent (MBGD)**: Computes gradients over batches of size 16.
10. **Visualization & Comparison**: Plotted loss curves, 2D contour paths, 3D loss surface bowl, and the fitted regression model.

---

## 📁 Repository Structure

```text
├── gradient_descent.py      # Core script executing Tasks 1 to 10 end-to-end
├── loss_surface_3d.py       # 3D surface plot of the paraboloid loss function
├── quadratic_1d_demo.py     # Minimal 1D quadratic function demo
├── plots/                   # High-resolution output visualizer plots
│   ├── 00_1d_quadratic_demo.png
│   ├── 01_learning_rate_experiment.png
│   ├── 02_optimization_methods_comparison.png
│   ├── 03_model_fit.png
│   └── 04_loss_surface_3d.png
└── README.md                # Project documentation
```

---

## 🚀 How to Run

### 1. Run the Full Gradient Descent Visualizer (Tasks 1-10)
```powershell
python gradient_descent.py
```

### 2. Run the 3D Loss Surface Visualizer
```powershell
python loss_surface_3d.py
```

### 3. Run the 1D Quadratic Demo
```powershell
python quadratic_1d_demo.py
```

---

## 📐 Mathematical Formulation

### 1. Model & Prediction
For a single feature $x$ with slope $w$ and intercept $b$:
$$\hat{y}^{(i)} = w \cdot x^{(i)} + b$$

### 2. Loss Function (Mean Squared Error)
$$L(w, b) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2$$

### 3. Gradients
$$\frac{\partial L}{\partial w} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) \cdot x^{(i)}$$
$$\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$

### 4. Parameter Update
$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

Where $\alpha$ is the learning rate.

---

## 📊 Summary of Optimization Methods

| Method | Batch Size | Pros | Cons | Trajectory on Loss Surface |
|---|---|---|---|---|
| **Batch GD** | Full dataset ($m$) | Direct, smooth, deterministic descent | Computationally heavy for large datasets | Straight path down the gradient |
| **Stochastic GD** | Single sample ($1$) | Fast updates, can escape shallow local minima | High variance, noisy zigzagging | Wanders with random fluctuations |
| **Mini-Batch GD** | Mini-batch (e.g. $16$) | Vectorized matrix efficiency, stable convergence | Requires tuning batch size | Balanced, smooth with slight stochasticity |

