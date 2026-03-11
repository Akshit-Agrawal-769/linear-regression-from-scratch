import numpy as np
import matplotlib.pyplot as plt

# Generate correlated data
np.random.seed(0)
n = 200

X1 = np.random.randn(n)
X2 = X1 * 0.95 + np.random.randn(n) * 0.1   # highly correlated with X1
# X2 = np.random.randn(n)   # independent
y  = 3*X1 + 2*X2 + np.random.randn(n)

# Compute loss over grid of (w1, w2)
w1_vals = np.linspace(-5, 8, 100)
w2_vals = np.linspace(-5, 8, 100)

W1, W2 = np.meshgrid(w1_vals, w2_vals)
Loss = np.zeros_like(W1)

for i in range(len(w1_vals)):
    for j in range(len(w2_vals)):
        y_pred = W1[j,i]*X1 + W2[j,i]*X2
        Loss[j,i] = np.mean((y - y_pred)**2)

# Plot contour (top-down view of bowl)
plt.contour(W1, W2, Loss, levels=30)
plt.xlabel("w1")
plt.ylabel("w2")
plt.title("Loss Surface Contour (Correlated Features)")
plt.show()