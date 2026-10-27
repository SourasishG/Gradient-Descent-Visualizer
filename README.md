# Gradient Descent From Scratch & Visualizer

A beginner-friendly, concise implementation of **Gradient Descent from scratch** in pure Python and NumPy (~75 lines of code, no ternary operators).

---

## 🎯 All 10 Tasks Covered

1. **Simple Loss Function**: Mean Squared Error ($L = \frac{1}{2m} \sum (\hat{y}-y)^2$)
2. **Gradient Calculation**: Derived derivatives $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$
3. **Scratch Implementation**: Pure Python and NumPy
4. **Random Initialization**: `np.random.randn()`
5. **Parameter Update**: $w \leftarrow w - \alpha \frac{\partial L}{\partial w}$
6. **Learning Rate Experiments**: Small (0.03), Medium (0.30), and Large (1.85)
7. **Batch Gradient Descent**: `batch_size = m`
8. **Stochastic Gradient Descent**: `batch_size = 1`
9. **Mini-Batch Gradient Descent**: `batch_size = 16`
10. **Visualizations**: 3-panel plot of learning rates, loss curves, and 2D contour paths

---

## 🚀 Run

```powershell
python gradient_descent.py
```

The script outputs `plots/gd_comparison.png` containing all comparison plots.
